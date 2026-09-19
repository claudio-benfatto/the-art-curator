# CLAUDE.md

Operational guidance for Claude Code working in this repo. Scope, rationale and build order live in [PLAN.md](PLAN.md).

## What this is

The Art Curator: an LLM art curator for Catalunya. Ingests venue/exhibition data, learns user taste implicitly from chat, replies with curated geographically-ordered itineraries. **v1 is a proof of concept**, not a product.

## Hard constraints

These are not style preferences. Violating any of them breaks a decision that was made deliberately.

### 1. Never persist third-party prose

The copyright posture is **facts from GRAF, prose from nobody**. GRAF's descriptive text and venue-site prose are copyrighted; facts are not.

- **Do not add a description/summary/body column to any GRAF-sourced table.** If a schema change seems to need one, it is the wrong schema change.
- `acf.desc_ca` / `desc_es` / `desc_en` from the GRAF API are **read-and-discard**. Never write them to Postgres, never put them in a prompt that produces stored output, never log them anywhere persistent.
- Venue-page text lands in `venue_pages.raw_text` with a **7-day TTL**: the main body `trafilatura` extracts (no markup, nav or boilerplate) from up to 8 exhibition/agenda pages per venue. That table is a processing cache, not a corpus. The purge job is not optional.
- Embeddings are computed **only from our own summaries**, never from `raw_text` or GRAF prose. Never feed third-party text into a persistent vector store (this rules out Bedrock Knowledge Bases over venue pages).
- What survives extraction is an **LLM-written original summary** plus `source_url`. The extraction prompt forbids reproducing spans longer than ~25 words.
- `tests/test_no_verbatim.py` enforces this. Do not skip or weaken it.

Exhibition/event **titles are persisted** — they are needed as dedup identifiers. Explicitly decided, not an oversight.

### 2. Every model call goes through `llm/client.py`

Token counts and cost are recorded *inside* the client wrapper. There is no code path that reaches a model without being recorded — that is what "observability is not an afterthought" means here architecturally.

- Never construct a model client (`AnthropicBedrockMantle`, `AnthropicBedrock`, boto3 `bedrock-runtime`) outside `llm/client.py`. CI greps for this.
- Never call `messages.create`, `converse` or `invoke_model` directly from application code.
- Every call — chat, extraction, embeddings, judge — writes a row to `llm_calls` with provider, model, tokens, cache hits, cost and latency.

### 3. The backend never knows what Telegram is

`api/` and `agent/` must not import or reference Telegram types. The bot is a client over `POST /chat`. A web client later should require zero backend changes.

Similarly, `queries/events.py` is the shared query module — it backs both the agent tools and the dev-time MCP server. Keep it free of agent or transport concerns.

### 4. Amazon Bedrock; chat is Claude-only

Bedrock gives multi-vendor models (incl. embeddings, which Anthropic doesn't offer) under one IAM/bill. **Chat stays on Claude** via the Messages-API endpoint; extraction, judge and embeddings may use any Bedrock model.

```python
from anthropic import AnthropicBedrockMantle
chat = AnthropicBedrockMantle(aws_region=settings.aws_region)  # SigV4 from the AWS credential chain
# non-Claude models: boto3 bedrock-runtime Converse / InvokeModel
```

Bedrock model IDs carry an `anthropic.` prefix. Env-driven:

```
AWS_REGION=eu-west-1
CHAT_MODEL=anthropic.claude-opus-5       # access is gated per account — check the console
EXTRACT_MODEL=anthropic.claude-haiku-4-5
EMBED_MODEL=                              # chosen by measurement in P3
```

Global endpoint by default. Regional (EU) endpoints cost **+10%** — only switch for a data-residency reason.

**Not available on Bedrock** (don't design against them): Message Batches endpoint, structured outputs on the Messages endpoint, MCP connector, programmatic tool calling, task budgets, server-side refusal `fallbacks` (use the SDK's client-side fallback), Models API, cache diagnostics. Bedrock's own batch inference exists but drops tool use/structured output and needs ~100+ records per job, so **extraction runs on-demand**.

**Settled — do not re-litigate:** a Claude Max subscription cannot back this application. Per claude.com/pricing, *"API access is separate and billed independently. The Max plan is for Claude's web, desktop, and mobile interfaces."* It remains the right tool for *developing* here (prompt iteration, gold-set construction) — just not for serving users.

### 5. Current Claude API shape

Training priors on these are stale — several changed in 2025–26.

- Opus 5: `thinking: {type: "adaptive"}`. **Never `budget_tokens`** — removed, returns 400.
- Opus 5: `output_config: {effort: "high"}` for chat. Inside `output_config`, not top-level.
- **Haiku 4.5 is different:** no `effort` (errors), no adaptive thinking. Extraction runs Haiku with thinking off.
- **No assistant prefill** — 400 on Opus 5. Use tool schemas or system-prompt instructions.
- **No structured outputs / `strict: true` on Bedrock's Messages endpoint.** Validate every tool input and extraction output with pydantic; on failure return a `tool_result` with `is_error: true` (chat) or retry once (extraction).
- Parse tool inputs with `json.loads()`. Never string-match the serialized input.

### 6. Cache layout

**One explicit breakpoint on the static system prefix, plus top-level automatic caching for the conversation tail.** Not manually-moved per-turn breakpoints.

```
system prompt → tool definitions → [explicit breakpoint] → conversation (auto-cached)
```

Never put `datetime.now()` or a per-request ID before the breakpoint. Verify with `usage.cache_read_input_tokens`; if it stays zero across turns, something in the prefix is varying.

Minimum cacheable prefix is model-dependent and **not monotonic across generations**:

- **Opus 5: 512 tokens** — our ~2.5k static prefix caches fine.
- **Haiku 4.5: 4096 tokens** — the extraction prompt is ~800, so it will **silently never cache**. No error, just `cache_creation_input_tokens: 0`. Cost estimates already assume this; don't hunt for a hit that cannot exist.

TTL is 5 minutes by default. The 1-hour TTL costs 2× on write and needs three reads to pay off — measure the start-to-start gap before switching.

### 7. The agent loop is capped at 4 tool iterations

Hard client-side count, enforced in `agent/loop.py`, with `iteration_count` on every trace. The cap bounds cost and latency; the metric is the evidence for whether the loop is needed at all. If traces show the overwhelming majority of exchanges are search → hydrate → answer, the loop can collapse to a workflow — but that is a decision made from data, not a guess.

Manual loop, ~60 lines. Not the beta Tool Runner, not Managed Agents, not the Agent SDK.

### 8. The model does no arithmetic

Dates and distances are computed in Postgres/PostGIS, where they are exact. `search_events` returns compact rows; `get_events` hydrates the shortlist. If the model is doing date math or distance estimation, the tool surface is wrong.

### 9. AWS infrastructure is Terraform, only

Every AWS resource (IAM roles and policies, RDS/Aurora, S3, scheduled jobs, networking) is declared in `infra/terraform/`. This starts in P0: the IAM policy that grants Bedrock access is the first resource.

- **No console or ad-hoc `aws` CLI creation** of anything that persists. Read-only CLI calls for inspection are fine.
- If a change is needed, change the `.tf` code and `terraform apply`. Never hand-edit a resource that Terraform manages. If drift is found, fix it by importing or reconciling in code, not by clicking.
- Remote state in S3 with locking. Never commit `*.tfstate`, `.terraform/` or `*.tfvars` containing secrets.
- Least-privilege IAM: the app's role gets `bedrock-mantle:CreateInference` / `bedrock:InvokeModel` scoped to the configured model ARNs only.
- The rare step Terraform cannot express (e.g. accepting Bedrock model-access terms, bootstrapping the state bucket) goes in `infra/README.md` as a numbered manual step. It is never done silently.

### 10. CI is GitHub Actions

All CI lives in `.github/workflows/`. No other CI system, and no checks that only run on someone's laptop.

- **On every PR:** `ruff`, `pytest` against a Postgres + PostGIS + pgvector service container, the model-client grep (constraint 2), `tests/test_no_verbatim.py` (constraint 1), and `terraform fmt -check` / `validate` / `plan` when `infra/` changes.
- **CI never calls a model.** Tests stub `llm/client.py`. A live Bedrock smoke test is a separate `workflow_dispatch` job, run on purpose because it costs money.
- **AWS auth from Actions uses GitHub OIDC** with a role declared in Terraform. Never store long-lived AWS keys as repo secrets.
- `terraform apply` runs only from `main`, behind a protected GitHub environment that needs manual approval.
- Pin third-party actions to a commit SHA.

## The GRAF API

Verified 2026-09-18. WordPress REST, no auth, `robots.txt` permits everything outside `/wp-admin/`.

| Endpoint | Count | Notes |
|---|---|---|
| `/wp-json/wp/v2/event-venues` | 568 | Venue taxonomy terms. Geo + address. |
| `/wp-json/wp/v2/users` | 158 | Venue profiles; 146 (92%) have `url` |
| `/wp-json/wp/v2/events` | 104 | **Live window, not an archive** |

Gotchas:

- **`longtitude` is misspelled in their API.** The field is literally `longtitude`, not `longitude`. Latitude is spelled correctly. Both are strings.
- The taxonomy's `rest_base` is `event-venues` (plural); the taxonomy *name* is `event-venue` (singular). Event objects carry `event-venues: [<term_id>]`.
- **`events` returns ~104 records regardless of date filters.** Passing `start=2020-01-01` changes nothing. This is why `graf_event_snapshots` exists and why nightly sync is load-bearing — history is unrecoverable if we miss it.
- Event `date` is `null`. Use `start` / `end` (ISO 8601 with offset).
- `per_page` maxes at 100; paginate and read `x-wp-total` / `x-wp-totalpages`.
- Some venue `url` values point at Instagram. Those venues are `crawl_enabled=false` and stay facts-only.
- Use `curl` over `urllib` when probing — Cloudflare rejects some Python UAs.

Useful ACF fields on events: `event_category`, `_event_free`, `_event_price-min`, `_event_price-max`, `_event_web_{ca,es,en}`, `event_contact_email`, `is_online_event`, `date_text`, `register_deadline`. The `title_*` fields are usable; the `desc_*` fields are **not** (constraint 1).

## Stack & conventions

Python 3.12+, FastAPI, SQLAlchemy 2.0 + Alembic, Postgres 16 + PostGIS + pgvector, `anthropic[bedrock]` + `boto3` for models, `pydantic-settings` for config, Terraform for AWS infra, `typer` for the CLI, `httpx` + `trafilatura` for crawling, `python-telegram-bot` for the client, OpenTelemetry + Langfuse for tracing, `pytest`. Docker Compose for local infra through P3; AWS (RDS/Aurora) after. Keep Postgres behind a connection string so the move is a config change.

- All configuration through `config.py` / env. No hardcoded model IDs or endpoints. Prices live in `pricing.yaml`.
- Async throughout the request path; the ingest CLI may be sync where it's simpler.
- Crawl politely: descriptive User-Agent, per-host delay, honour `robots.txt` via `urllib.robotparser`, cap ~8 pages per venue.
- Skip extraction when `content_hash` is unchanged. Re-extracting unchanged pages is the main avoidable cost.

## Commands

Managed with `uv`. Commands not yet implemented are the intended surface; update as they land.

```bash
uv sync                                             # create .venv from uv.lock
uv run ruff check . && uv run ruff format --check .
uv run python -m art_curator.cli config             # print effective settings

docker compose up -d db langfuse
uv run alembic upgrade head

uv run python -m art_curator.cli smoke              # traced test call, verifies cost recording
uv run python -m art_curator.cli sync-graf          # pull GRAF facts, snapshot events
uv run python -m art_curator.cli crawl --pilot      # crawl the ~20 pilot venues
uv run python -m art_curator.cli extract            # venue pages -> exhibitions
uv run python -m art_curator.cli embed              # summaries -> pgvector (P3)
uv run python -m art_curator.cli eval-extraction    # score against the gold set
uv run python -m art_curator.cli chat               # talk to the curator in the terminal

uv run pytest
docker compose up -d                                # full stack incl. api + bot
```

`mcp_server.py` is a dev-time MCP surface over `queries/events.py`, for interrogating the corpus from Claude Code while building the P2 gold set. It is **not** on the serving path.

## Current state

Nothing implemented yet. Next step is **P0** (scaffold, schema, config, instrumented client, telemetry) — see PLAN.md § Build order.

First tasks in P0, before writing code against them:

- Confirm Opus 5 model access in the Bedrock console (it is not open to every account), and that top-level automatic caching works on the Mantle endpoint (check `cache_read_input_tokens` on a second call).
- Verify the Langfuse SDK surface against live docs. If it has drifted, the OTel + Postgres layer stands alone and Langfuse can be dropped without data loss.

**P3 is the real checkpoint.** If the curator isn't good over 20 venues of data, scaling to 158 won't fix it. Don't build P4–P6 to avoid finding out.
