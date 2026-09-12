"""Run the golden set through the agent and gate on quality.

    python -m evals.run_evals [--golden evals/golden.jsonl] [--report eval-report.json]
                              [--min-faithfulness 0.85] [--min-relevance 0.7]
                              [--min-precision 0.6] [--min-checks 0.85] [--pace 3] [--only g01,g02]

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

METRICS = ("faithfulness", "answer_relevance", "context_precision")


@dataclass(frozen=True)
class Thresholds:
    faithfulness: float = 0.85
    relevance: float = 0.70
    precision: float = 0.60
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


def evaluate(cases: list[dict], run_case: RunCase, judge: J.Judge, th: Thresholds, pace: float = 0.0) -> Report:
    rows: list[dict] = []
    for i, case in enumerate(cases):
        if pace and i:
            time.sleep(pace)
        row: dict[str, Any] = {"id": case.get("id"), "kind": case.get("kind"), "question": case.get("question")}
        t0 = time.monotonic()
        try:
            events, answer, final = run_case(case)
        except Exception as e:  # noqa: BLE001
            row.update(error=f"{type(e).__name__}: {str(e)[:200]}", failed_checks=["agent_error"],
                       faithfulness=0.0, answer_relevance=0.0, context_precision=0.0, ms=int((time.monotonic() - t0) * 1000))
            rows.append(row)
            continue
        ans = str(answer.get("answer", ""))
        cits = list(answer.get("citations", []) or [])
        ctx = _contexts(final)
        scores = J.score_case(str(case["question"]), ans, ctx, judge, case.get("reference"))
        failed = J.deterministic_checks(case, ans, cits)
        row.update(
            answer=ans[:600], citations=[c.get("label") for c in cits][:8], contexts=len(ctx),
            branches=len({e.get("branch") for e in events if e.get("branch") is not None}),
            faithfulness=scores.faithfulness, answer_relevance=scores.answer_relevance,
            context_precision=scores.context_precision, claims=scores.claims,
            failed_checks=failed, judge_error=scores.error, ms=int((time.monotonic() - t0) * 1000),
        )
        rows.append(row)

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    checks_pass = round(sum(1 for r in rows if not r.get("failed_checks")) / len(rows), 4) if rows else 0.0
    summary: dict[str, Any] = {
        "cases": len(rows),
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
    if not rows:
        failed_th.append("no_cases")
    summary["failed_thresholds"] = failed_th
    return Report(passed=not failed_th, summary=summary, cases=rows, thresholds=asdict(th))


# -- the real agent ------------------------------------------------------------------------
def agent_runner(session_id: str = "evals") -> RunCase:
    from app.agent.graph import run_agent_graph_full

    def run_case(case: dict) -> tuple[list[dict], dict, dict]:
        return asyncio.run(run_agent_graph_full(session_id, str(case["question"])))

    return run_case


def _print_table(rep: Report) -> None:
    print(f"{'id':5} {'kind':9} {'faith':>6} {'relev':>6} {'prec':>6} {'br':>2} checks")
    for r in rep.cases:
        prec = "-" if r.get("context_precision") is None else f"{r['context_precision']:.2f}"
        checks = "ok" if not r.get("failed_checks") else ",".join(r["failed_checks"])
        print(f"{r['id']:5} {str(r.get('kind')):9} {r.get('faithfulness', 0):6.2f} {r.get('answer_relevance', 0):6.2f} "
              f"{prec:>6} {r.get('branches', 0):2} {checks}")
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
    print(f"evals: {len(cases)} cases; agent={settings.llm_model}; judge={settings.eval_judge_model}; "
          f"threshold={settings.relevance_distance_threshold}")

    try:
        rep = evaluate(cases, agent_runner(), J.groq_judge, th, pace=args.pace)
    finally:
        try:
            from app.agent.toolbus import close_bus

            asyncio.run(close_bus())
        except Exception:  # noqa: BLE001
            pass
    rep.summary["agent_model"] = settings.llm_model
    rep.summary["judge_model"] = settings.eval_judge_model
    rep.summary["git_sha"] = os.environ.get("GITHUB_SHA", "")
    Path(args.report).write_text(rep.to_json(), encoding="utf-8")
    _print_table(rep)
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
