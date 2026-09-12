"""Run the golden set through the agent and gate on quality.

    python -m evals.run_evals [--golden evals/golden.jsonl] [--report eval-report.json]
                              [--min-faithfulness 0.85] [--min-relevance 0.7]
                              [--min-precision 0.25] [--min-checks 0.85] [--pace 3] [--only g01,g02]

Exit codes: 0 pass, 1 a threshold was missed (the CI gate), 2 not runnable (no LLM key).
Each case runs through the real LangGraph agent (MCP servers, DB, web) and is scored by
`evals.judge` (LLM metrics) plus deterministic checks. `eval-report.json` is written always.
Never lower a threshold to make CI pass; fix the agent or a wrong golden expectation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evals import judge as J

RunCase = Callable[[dict], tuple[list[dict], dict, dict]]

@dataclass(frozen=True)
class Thresholds:
    faithfulness: float = 0.85
    relevance: float = 0.70
    precision: float = 0.25  # calibrated 2026-09-12: baseline 0.35 with k=6 over an 11-chunk corpus (see DECISIONS 24)
    checks: float = 0.85


@dataclass
class Report:
    passed: bool
    summary: dict
    cases: list[dict] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_golden(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _contexts(final_state: dict) -> list[str]:
    return [getattr(s, "text", "") for s in (final_state or {}).get("collected", []) or []]


RATE_LIMIT_PAUSE_S = 30.0
HARD_CHECKS = frozenset({"expect_grounded", "expect_web", "agent_error"})


def _is_rate_limit(e: BaseException) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return "429" in s or "rate limit" in s or "ratelimit" in s or "tokens per minute" in s


def _run_with_rate_limit_retry(run_case: RunCase, case: dict) -> tuple[list[dict], dict, dict]:
    """Free-tier per-minute caps: wait out the window once, then retry."""
    try:
        return run_case(case)
    except Exception as e:  # noqa: BLE001
        if not _is_rate_limit(e):
            raise
        time.sleep(RATE_LIMIT_PAUSE_S)
        return run_case(case)


def evaluate(cases: list[dict], run_case: RunCase, judge: J.Judge, th: Thresholds, pace: float = 0.0) -> Report:
    rows: list[dict] = []
    for i, case in enumerate(cases):
        if pace and i:
            time.sleep(pace)
        row: dict[str, Any] = {"id": case.get("id"), "kind": case.get("kind"), "question": case.get("question")}
        t0 = time.monotonic()
        try:
            events, answer, final = _run_with_rate_limit_retry(run_case, case)
        except Exception as e:  # noqa: BLE001
            row.update(error=f"{type(e).__name__}: {str(e)[:200]}", failed_checks=["agent_error"],
                       faithfulness=0.0, answer_relevance=0.0, context_precision=0.0, ms=int((time.monotonic() - t0) * 1000))
            rows.append(row)
            continue
        ans = str(answer.get("answer", ""))
        cits = list(answer.get("citations", []) or [])
        ctx = _contexts(final)
        if case.get("kind") == "metadata":
            # Deterministic route with no retrieved contexts: nothing for the judge to score.
            scores = J.CaseScores(faithfulness=None, answer_relevance=None, context_precision=None)  # type: ignore[arg-type]
        else:
            scores = J.score_case(str(case["question"]), ans, ctx, judge, case.get("reference"))
        failed = J.deterministic_checks(case, ans, cits)
        row.update(
            answer=ans[:600], citations=[c.get("label") for c in cits][:8], contexts=len(ctx),
            branches=len({e.get("branch") for e in events if e.get("branch") is not None}),
            faithfulness=scores.faithfulness, answer_relevance=scores.answer_relevance,
            context_precision=scores.context_precision, claims=scores.claims,
            failed_checks=failed, judge_error=scores.error, ms=int((time.monotonic() - t0) * 1000),
            judge_model_used=getattr(J, "last_judge_model", None),
        )
        rows.append(row)

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    checks_pass = round(sum(1 for r in rows if not r.get("failed_checks")) / len(rows), 4) if rows else 0.0
    # A grounded case answered from the web (or a web case with no web source) is a hard
    # failure regardless of the averages: that is exactly the regression the gate exists for.
    hard_failures = sum(1 for r in rows if any(f in HARD_CHECKS for f in (r.get("failed_checks") or [])))
    summary: dict[str, Any] = {
        "cases": len(rows),
        "hard_check_failures": hard_failures,
        "faithfulness": mean("faithfulness"),
        "answer_relevance": mean("answer_relevance"),
        "context_precision": mean("context_precision"),
        "checks_pass_rate": checks_pass,
        "agent_errors": sum(1 for r in rows if r.get("error")),
        "judge_errors": sum(1 for r in rows if r.get("judge_error")),
    }
    failed_th: list[str] = []
    for key, minimum in (("faithfulness", th.faithfulness), ("answer_relevance", th.relevance),
                         ("context_precision", th.precision), ("checks_pass_rate", th.checks)):
        val = summary[key]
        if val is None or val < minimum:
            failed_th.append(key)
    if hard_failures:
        failed_th.append("hard_checks")
    if not rows:
        failed_th.append("no_cases")
    summary["failed_thresholds"] = failed_th
    return Report(passed=not failed_th, summary=summary, cases=rows, thresholds=asdict(th))


# -- the real agent ------------------------------------------------------------------------
_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    """ONE loop for the whole run: the keep-alive MCP session lives on it. A fresh
    asyncio.run per case would kill the retrieval child between cases."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop


def agent_runner(session_id: str = "evals") -> RunCase:
    from app.agent import graph

    def run_case(case: dict) -> tuple[list[dict], dict, dict]:
        return _get_loop().run_until_complete(graph.run_agent_graph_full(session_id, str(case["question"])))

    return run_case


def shutdown_runner() -> None:
    global _loop
    if _loop is not None and not _loop.is_closed():
        try:
            from app.agent.toolbus import close_bus

            _loop.run_until_complete(close_bus())
        except Exception:  # noqa: BLE001
            pass
        _loop.close()
    _loop = None


def _print_table(rep: Report) -> None:
    print(f"{'id':5} {'kind':9} {'faith':>6} {'relev':>6} {'prec':>6} {'br':>2} checks")
    def fmt(v) -> str:
        return "-" if v is None else f"{v:.2f}"

    for r in rep.cases:
        checks = "ok" if not r.get("failed_checks") else ",".join(r["failed_checks"])
        print(f"{r['id']:5} {str(r.get('kind')):9} {fmt(r.get('faithfulness')):>6} {fmt(r.get('answer_relevance')):>6} "
              f"{fmt(r.get('context_precision')):>6} {r.get('branches') or 0:2} {checks}")
    print("summary:", json.dumps(rep.summary))
    print("PASS" if rep.passed else f"FAIL: below threshold on {rep.summary['failed_thresholds']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default="evals/golden.jsonl")
    ap.add_argument("--report", default="eval-report.json")
    ap.add_argument("--min-faithfulness", type=float, default=Thresholds.faithfulness)
    ap.add_argument("--min-relevance", type=float, default=Thresholds.relevance)
    ap.add_argument("--min-precision", type=float, default=Thresholds.precision)
    ap.add_argument("--min-checks", type=float, default=Thresholds.checks)
    ap.add_argument("--pace", type=float, default=3.0, help="seconds between cases (free-tier rate limits)")
    ap.add_argument("--only", default="", help="comma-separated case ids")
    args = ap.parse_args(argv)

    from app.config import settings

    if not settings.llm_configured:
        print("LLM_API_KEY is required for evals (set the repository secret LLM_API_KEY)", file=sys.stderr)
        return 2
    if not settings.db_configured:
        print("DATABASE_URL is required for evals", file=sys.stderr)
        return 2

    cases = load_golden(args.golden)
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        cases = [c for c in cases if c.get("id") in keep]
    th = Thresholds(args.min_faithfulness, args.min_relevance, args.min_precision, args.min_checks)
    print(f"evals: {len(cases)} cases; agent={settings.llm_model}; judge={J.effective_judge_model()}; "
          f"threshold={settings.relevance_distance_threshold}")

    try:
        rep = evaluate(cases, agent_runner(), J.groq_judge, th, pace=args.pace)
    finally:
        shutdown_runner()
    rep.summary["agent_model"] = settings.llm_model
    rep.summary["judge_model"] = J.effective_judge_model()
    rep.summary["git_sha"] = os.environ.get("GITHUB_SHA", "")
    Path(args.report).write_text(rep.to_json(), encoding="utf-8")
    _print_table(rep)
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
