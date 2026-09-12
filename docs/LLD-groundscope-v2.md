# Low-Level Design — Groundscope v2 (Agent Harness)

**Version:** v2 (design) · **Date:** 2026-06-08 · app root `groundscope/`
**Derives from:** `docs/v2-agentic-design.md`, `docs/HLD-groundscope-v2.md`

This LLD specifies the modules that change or appear in v2. Modules unchanged from v1
(ingestion `extract.py`/`chunker.py`/`embedder.py`, `storage.py` core schema, sessions
cookie logic) are described in `LLD-groundscope.md` and only their deltas are noted here.

---

## 1. Tool bus — unify on MCP

### 1.1 MCP servers (Groundscope-owned)
Three local stdio servers, all registered in `mcp.json`, all bundled into the image.

**`mcp_servers/retrieval_server.py`** — `groundscope-retrieval` (also the PUBLIC server)
```python
mcp = FastMCP("groundscope-retrieval")

@mcp.tool()
def hybrid_search(session_id: str, query: str, limit: int = 6) -> str:
    """Hybrid (pgvector + BM25, RRF) retrieval over a session's docs + GLOBAL corpus.
    Returns JSON: {summary, score, sources:[{kind,label,detail,text}]}."""
    emb = get_embedder().embed([query])[0]
    hits, best = storage.hybrid_search(session_id, emb, query, limit=limit)
    sources = [{"kind":"doc","label":f"{h.file_name} p.{h.page_number}",
                "detail":f"p.{h.page_number}","text":h.text} for h in hits]
    return json.dumps({"summary": f"{len(hits)} chunks; best distance "
                       f"{best:.3f}" if best is not None else "0 chunks",
                       "score": best, "sources": sources})

@mcp.tool()
def metadata_query(session_id: str) -> str:
    """JSON list of documents in the session: [{file_name,pages,chunk_count}]."""
    return json.dumps({"documents": storage.list_documents(session_id)})
```
- **Structured JSON return** (not a plain string) so the orchestration gate can read
  `score`. The graph node `json.loads()` the tool result.
- `session_id` is a parameter but **the LLM never fills it** (see §1.3).

**`mcp_servers/web_server.py`** — `groundscope-web`
```python
@mcp.tool()
def web_search(query: str, limit: int = 4) -> str: ...        # Tavily, JSON sources
@mcp.tool()
def gemini_grounding(query: str) -> str: ...                  # Gemini "Grounding with
                                                             # Google Search" (optional key)
```

**`mcp_servers/util_server.py`** — `groundscope-utils` (exists; extend)
```python
@mcp.tool()
def calculator(expression: str) -> str: ...   # existing (AST eval)
@mcp.tool()
def current_datetime() -> str: ...            # existing
@mcp.tool()
def fetch_url(url: str, max_chars: int = 4000) -> str:        # new; httpx GET, text only
    """Fetch a URL and return readable text (truncated). For grounding on a known link."""
```

### 1.2 Registry (`agent/mcp_registry.py`) — unchanged shape, more servers
`load_mcp_tools()` already reads `mcp.json`, substitutes `python`→`sys.executable` for
local stdio servers, and returns LangChain tools via `MultiServerMCPClient.get_tools()`.
v2 just adds the three server entries; **no code change** to the loader. Returns `[]`
gracefully if a server fails, so the agent still runs.

`mcp.json` (v2):
```json
{
  "groundscope-retrieval": {"command":"python","args":["mcp_servers/retrieval_server.py"],"transport":"stdio"},
  "groundscope-web":       {"command":"python","args":["mcp_servers/web_server.py"],"transport":"stdio"},
  "groundscope-utils":     {"command":"python","args":["mcp_servers/util_server.py"],"transport":"stdio"}
}
```

### 1.3 Open vs. context tools — the tenant-isolation rule
Tools fall into two classes, enforced by **who invokes them**:

| Class | Tools | Invoked by | `session_id` |
|---|---|---|---|
| **Context** | `hybrid_search`, `metadata_query` | orchestration **node** (deterministic) | **server-injected** by the node; not in the LLM-visible tool schema |
| **Open** | `calculator`, `current_datetime`, `fetch_url`, `web_search`, `gemini_grounding` | LLM (ReAct) or node | n/a |

Implementation: context tools are wrapped so the node calls them with the request's
`session_id`; the version bound into `ChatOpenAI.bind_tools(...)` for the ReAct worker
has `session_id` **omitted from the schema** (partial-applied), so the model can request
"search my docs" but cannot target another tenant. This keeps a uniform MCP bus while
making isolation deterministic.

---

## 2. Supervisor graph (`agent/graph.py`) — fan-out

### 2.1 State
```python
class S(TypedDict, total=False):
    session_id: str
    question: str
    subqueries: list[str]          # NEW — planner decomposition
    branches: list[dict]           # NEW — per-subquery {sources, best}
    step: int
    collected: list                # aggregated sources
    best: Optional[float]          # min distance across branches
    web_ran: bool
    answer: str
    citations: list
```

### 2.2 Nodes
- **`planner_node`** — classify intent (deterministic `is_metadata` first). For a
  knowledge question, ask the LLM (`complete_json`) for a decomposition:
  `{"subqueries": ["...", "..."]}` (1 = simple, N = complex). Emits a `decision` trace.
- **`fanout`** — emit `Send("retrieval_worker", {"session_id", "subquery": q})` per
  subquery. LangGraph runs them **in parallel**; each emits trace events tagged with a
  `branch` index.
- **`retrieval_worker`** — calls the `groundscope-retrieval` MCP tool (session injected),
  parses JSON, returns `{branch, sources, best}`. May also call web/open tools.
- **`aggregate`** — reduce all `branches`: concat + dedupe sources by (file,page)/url;
  `best = min(branch.best for grounded branches)`.
- **`gate`** (`_gate`) — `grounded = best is not None and best <= threshold`.
  grounded or web-not-configured → `synth`; else → `web`.
- **`web_node`** — Tavily/Gemini fallback (existing logic), replaces weak doc context.
- **`synth_node`** — LLM answers from the delimited, untrusted SOURCES block; cites
  `[file p.N]` / `[Web: title — url]`; refuses if empty. Output guardrail screens it.
- **`hitl_gate`** (optional) — before a sensitive open-tool action, `interrupt()`;
  resume via checkpointer.

### 2.3 Edges
```
START → planner
planner → fanout            (knowledge)   |  → metadata  |  → tools(ReAct)
fanout → retrieval_worker × N  (Send, parallel)
retrieval_worker → aggregate
aggregate → gate → {synth | web}
web → synth
synth | metadata | tools → END
```

### 2.4 Streaming (correction vs. v1 LLD)
The graph uses an **injected** `StreamWriter` (`from langgraph.types import
StreamWriter`, a node parameter) — **not** `get_stream_writer` (which does not exist in
langgraph 0.2.39). `run_agent_graph(...).astream(stream_mode="custom")` yields the same
`{"kind":"trace"|"answer", "payload":...}` envelope. v2 trace payloads add
`branch:int|None`, `tokens:int|None`, `cost_usd:float|None`.

---

## 3. LLM router + cost (`agent/llm.py`)
Existing: tiered `_tiers()` (primary 70B → 8B → optional 3rd provider), `_Breaker`
(threshold 5, cooldown 60 s), `complete()` / `complete_json()`, `wrap_openai` for
LangSmith. **v2 delta:** capture `resp.usage` (prompt/completion tokens) and estimate
cost from a per-model rate table; return alongside the text via a light
`Completion(text, tokens, cost_usd)` dataclass (or a module-level accumulator keyed by
trace). The synth/planner nodes attach `tokens`/`cost_usd` to their trace events.

```python
_RATES = {  # USD per 1M tokens (free tier shows $0; table documents real cost)
  "llama-3.3-70b-versatile": (0.59, 0.79),
  "llama-3.1-8b-instant":    (0.05, 0.08),
}
```

---

## 4. Durable checkpointer
LangGraph **Postgres checkpointer** on Supabase, `thread_id = session_id` (tenant).
```python
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(settings.database_url)
checkpointer.setup()                         # creates checkpoint tables once
_graph = _build().compile(checkpointer=checkpointer)
# invoke with config={"configurable": {"thread_id": session_id}}
```
Enables crash/pause/HITL resume. `interrupt()` in `hitl_gate` pauses the run to the
checkpointer; a follow-up `/ask` with the same thread + a `resume` payload continues.

---

## 5. Guardrails (`agent/guards.py`)
- **`screen_input(question) -> Ok | Block(reason)`** — prompt-injection / PII heuristics
  (LLM Guard or a lightweight ruleset). Block → 400-style refusal trace.
- **`screen_output(answer, sources) -> Ok | Revise`** — checks the answer cites only
  provided sources (faithfulness guard) and contains no leaked PII.
- **Budgets** — `max_tool_rounds` (exists), `max_tokens_per_request`, `max_subqueries`.

---

## 6. Eval harness (`evals/`)
```
evals/
  golden.jsonl        # [{id, question, expect_grounded:bool, must_cite?:[...], notes}]
  run_evals.py        # loads golden set, runs the agent, scores, writes report
  metrics.py          # faithfulness / answer-relevance / context-precision (Ragas/DeepEval)
```
`run_evals.py` runs each golden case through the agent (langgraph engine), computes
metrics with a **free LLM judge** (Groq/Gemini), and emits `eval-report.json` +
a pass/fail against thresholds (e.g. faithfulness ≥ 0.85). Exit non-zero on regression
so CI fails. A small `make eval` target wraps it.

---

## 7. UI deltas (`static/index.html`)
- **`GET /tools`** new endpoint returns `{servers:[{name, tools:[{name,description}]}]}`
  from the loaded registry; a **Connected Tools panel** renders it (toggle/inspect).
- Trace renderer handles `branch` (group parallel sub-query steps) and shows the
  `cost_usd`/`tokens` on synthesis steps; a footer **cost line** per answer.
- Reskin to warm-dark + gold to match the portfolio (designed separately).

---

## 8. API (v2)
| Endpoint | Change |
|---|---|
| `POST /ask` (SSE) | + `branch`, `tokens`, `cost_usd` in trace; optional `{resume}` body for HITL |
| `GET /tools` | **new** — connected MCP servers + tools (for the panel) |
| `POST /ingest`, `GET /documents`, `GET /health` | unchanged |

---

## 9. CI/CD (`.github/workflows/ci.yml`)
```yaml
name: ci
on: [pull_request, push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check .                 # lint
      - run: pyright app mcp_servers       # typecheck
      - run: pytest -q                     # unit/integration
      - run: python evals/run_evals.py --threshold 0.85   # eval gate (uses CI secrets)
      - run: docker build -t groundscope .                # build proof
```
- Eval/LLM keys via GitHub Actions **secrets** (never committed).
- Push to `main` → Render auto-deploy (existing). PRs get the CI check + (optional)
  Render preview.

---

## 10. Storage deltas (`storage.py`)
- Core schema unchanged (`chunks`, `documents`, pgvector `<=>`, BM25 `tsvector` + RRF in
  `hybrid_search`).
- **RLS (multi-tenancy):** enable row-level security on `chunks`/`documents`, policy
  `session_id = current_setting('app.tenant')`; the node sets `app.tenant` per request.
  `GLOBAL` corpus readable by all.
- **Checkpointer tables:** created by `PostgresSaver.setup()` in the same DB.

---

## 11. Env vars (v2 additions)
Existing v1 set, plus: `GEMINI_API_KEY` (optional, for `gemini_grounding` + 3rd LLM
tier), `MAX_TOKENS_PER_REQUEST`, `MAX_SUBQUERIES`, `EVAL_THRESHOLD`. Langfuse vars
deprecated (observability is LangSmith: `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`).

---

## 12. Build order (maps to roadmap)
1. **Tool bus** — split native tools into the three MCP servers, structured returns,
   open/context wrapping. (Layer 2)
2. **Fan-out** — planner decomposition + `Send` workers + aggregate. (Layer 1)
3. **Eval harness + CI** — golden set, `run_evals.py`, GitHub Actions. (Layers 5–6)
4. **Checkpointer + HITL + guardrails.** (Layer 3)
5. **Cost accounting + `/tools` panel + reskin.** (Layers 2,4)
6. **RLS multi-tenancy.** (Layer 6)
