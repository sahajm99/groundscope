# Low-Level Design — Groundscope

**Version:** v1 (as built, 2026-06-07) · app root `groundscope/`

---

## 1. Config (`app/config.py`)
Pydantic-settings from `.env`. Key fields: `llm_api_key/base_url/model`,
`embed_provider/model/vector_dim`, `database_url`, `tavily_api_key`, `langfuse_*`,
`chunk_max_tokens=400`, `chunk_overlap_tokens=50`, `session_ttl_seconds=3600`,
`max_upload_mb=10`, `max_pages=120`, `rate_limit_per_min=12`, `global_daily_cap=500`,
`max_tool_rounds=2`, `relevance_distance_threshold=0.5`, `agent_engine="loop"`.
Helpers: `llm_configured`, `db_configured`, `web_search_configured`.

## 2. Ingestion
- **`extract.py`** — `extract_pages(bytes, name) -> list[Page(page_number, text)]`.
  PDF via PyMuPDF (`page.get_text`), DOCX via python-docx, TXT/MD direct. Raises
  `NoTextError` if no selectable text (scanned PDF) — no OCR in v1.
- **`chunker.py`** — `chunk_pages(pages, max=400, overlap=50) -> list[Chunk(idx, page, text)]`.
  Per-page whitespace-word windows (`step = max-overlap`), each chunk keeps its page number.
- **`embedder.py`** — `get_embedder()` singleton.
  - `LocalEmbedder` (fastembed `TextEmbedding`): `.embed(texts)`; `dim` probed empirically.
  - `OpenAICompatEmbedder`: batched (64) + exponential backoff, for Gemini/OpenAI.
  - **Startup assert:** `emb.dim == settings.vector_dim` (pgvector column is fixed-dim).

## 3. Storage (`app/storage.py`) — Supabase Postgres + pgvector
Tables:
```
chunks(id bigserial pk, session_id text, doc_id text, file_name text,
       page_number int, chunk_index int, text text, embedding vector(384))
documents(doc_id text pk, session_id text, file_name text, pages int,
          chunk_count int, uploaded_at timestamptz default now())
indexes: chunks(session_id), documents(session_id)
```
- `_connect(register=True)` — psycopg3, `autocommit`. `register_vector` only when needed
  (NOT for bootstrap, since the `vector` extension is created by `init_schema` first —
  avoids a chicken-and-egg "vector type not found").
- `_vec(list) -> np.float32 ndarray` — pgvector's adapter maps **numpy float32** to the
  `vector` type; a plain Python list is sent as `float8[]` and breaks the `<=>` operator.
- `add_document(session, doc_id, file, pages, rows)` — `executemany` insert chunks (+ `_vec`)
  + upsert a `documents` row.
- `vector_search(session, query_vec, limit=6) -> [Hit(text, file_name, page_number, distance)]`
  — `ORDER BY embedding <=> %s` (cosine distance), `WHERE session_id IN (session, 'GLOBAL')`.
- `list_documents(session)`, `purge_session(session)` (never deletes `GLOBAL`).

## 4. Tools (`app/agent/tools.py`)
Each returns grounding `Source(kind, label, detail, text)`.
- `vector_search(session, query, limit=6) -> (summary, [Source], best_distance)` —
  embeds the query, searches, wraps hits as `doc` Sources labelled `"<file> p.N"`.
- `metadata_query(session) -> (summary, [])` — `list_documents`, human-readable summary.
- `web_search(query, limit=4) -> (summary, [Source])` — Tavily; `web` Sources with `detail=url`.

## 5. Agent (`app/agent/loop.py` — default engine)
`run_agent(session, question)` async generator yielding
`{"kind":"trace", "payload": TraceEvent}` then `{"kind":"answer", "payload":{answer, citations}}`.
Flow:
1. **route** — `is_metadata(question)` keyword detector (`_META_PHRASES` + "how many … page/doc/file").
2. **metadata branch** → `metadata_query` → answer = summary.
3. **vector_search** (round 1) → emit `tool_result` with `score` = best distance.
4. **gate** — `grounded = sources and best <= relevance_distance_threshold (0.5)`.
   - not grounded & web configured → **web_search** (round 2); `collected = web sources only`.
   - grounded → emit "documents relevant" decision.
5. **synthesize | refuse** — if no sources → refusal; else LLM (`_SYNTH_SYS`) answers from a
   delimited SOURCES block (untrusted), citing `[file p.N]` / `[Web: title — url]`.
Every step mirrored to Langfuse via `obs.event`; `obs.finish(answer)` at each terminal.
Blocking LLM/tool calls run via `asyncio.to_thread`.

## 6. Agent (`app/agent/graph.py` — alt engine, `AGENT_ENGINE=langgraph`)
Same nodes as a `langgraph.StateGraph`: `route → (metadata|vector) → gate(vector→synth|web) →
web → synth`. Emits the identical `TraceEvent`s via LangGraph's custom stream writer
(`get_stream_writer`), consumed by `run_agent_graph(...).astream(stream_mode="custom")`.
Built but not yet runtime-verified — loop is the default.

## 7. Trace schema (`app/agent/trace.py`)
`TraceEvent{ step:int, type:tool_call|tool_result|decision|synthesis|refusal,
tool, input, summary, score:float|None, ms:int, links:[{title,url}]|None }`.
One shape consumed by both the SSE panel and Langfuse.

## 8. API
**`POST /ask`** (SSE, `text/event-stream`): events `trace` (TraceEvent) then `answer`
(`{answer, citations:[{label,kind,detail}]}`). Status: 200 stream · 400 bad/oversized input
· 429 rate/daily cap · 503 no LLM key. `Cache-Control: no-store`.
**`POST /ingest`** (multipart `file`): `{doc_id,file_name,pages,chunks}`; 413 size/pages,
422 no-text, 503 no DB.
**`GET /documents`**: `{documents:[{file_name,pages,chunk_count,uploaded_at}]}` (session-scoped).
**`GET /health`**: subsystem readiness + DB ping (also the keep-warm target).
**`GET /`**: static `index.html`, `no-store`.

## 9. Sessions / hardening (`app/sessions.py`)
`get_or_create_session` — `gs_session` UUID cookie (httponly, samesite=lax, max-age=TTL);
in-memory `_sessions` map (lost on restart). `client_ip` = `X-Forwarded-For` first hop.
`rate_limited(ip)` 12/min; `daily_cap_reached()` global 500/day; `purge_expired()` TTL sweep.

## 10. Observability (`app/observability.py`)
`start_trace(question)` → `_LangfuseTrace` (one `trace`, a `span` per `TraceEvent`,
`update(output=answer)` on finish) or `_NullTrace` no-op when keys absent.

## 11. Frontend (`static/index.html`)
Single page, inline CSS/JS. SSE over `fetch` POST: read `res.body`, **normalize `\r`**
(sse-starlette uses CRLF), split on `\n\n`, dispatch `trace`/`answer`. Renders the documents
bar (`GET /documents`), the live trace (with clickable web `links`), and answer citations.

## 12. Known limitations
In-memory sessions; demo caps (10MB/120pp); no OCR; single global relevance threshold;
LangGraph engine unverified; visual grounding deferred to v1.1.

## 13. Env vars
`LLM_API_KEY` (req), `LLM_BASE_URL`, `LLM_MODEL`, `EMBED_PROVIDER`, `EMBED_MODEL`,
`VECTOR_DIM`, `EMBED_API_KEY`/`EMBED_BASE_URL` (if openai provider), `DATABASE_URL` (req),
`TAVILY_API_KEY`, `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY`/`HOST`, `AGENT_ENGINE`,
`RELEVANCE_DISTANCE_THRESHOLD`, caps (`MAX_UPLOAD_MB`, `MAX_PAGES`, `RATE_LIMIT_PER_MIN`,
`GLOBAL_DAILY_CAP`, `SESSION_TTL_SECONDS`).
