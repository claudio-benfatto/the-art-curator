# CLAUDE.md

Operational guidance for Claude Code working in this repo. Scope, rationale and build order live in [PLAN.md](PLAN.md) — this file is the set of things that are easy to get wrong.

## What this is

The Art Curator: an LLM art curator for Catalunya. Ingests venue/exhibition data, learns user taste implicitly from chat, replies with curated geographically-ordered itineraries. **v1 is a proof of concept**, not a product.

## Hard constraints

These are not style preferences. Violating any of them breaks a decision that was made deliberately.

### 1. Never persist third-party prose

The copyright posture is **facts from GRAF, prose from nobody**. See PLAN.md § Copyright posture for the reasoning.

- **Do not add a description/summary/body column to any GRAF-sourced table.** If a schema change seems to need one, it is the wrong schema change.
- `acf.desc_ca` / `desc_es` / `desc_en` from the GRAF API are **read-and-discard**. Never write them to Postgres, never put them in a prompt that produces stored output, never log them anywhere persistent.
- Venue-site HTML lands in `venue_pages` with a **7-day TTL**. That table is a processing cache, not a corpus. The purge job is not optional.
- What survives extraction is an **LLM-written original summary** plus `source_url`. The extraction prompt forbids reproducing spans longer than ~25 words.
- `tests/test_no_verbatim.py` enforces this. Do not skip or weaken it.

Exhibition/event **titles are persisted** — they are needed as dedup identifiers. Explicitly decided, not an oversight.

### 2. Every model call goes through `llm/client.py`

Token counts and cost are recorded *inside* the client wrapper. There is no code path that reaches a model without being recorded — that is what "observability is not an afterthought" means here architecturally.

- Never construct `Anthropic()` outside `llm/client.py`. CI greps for this.
- Never call `messages.create` directly from application code.
- Every call writes a row to `llm_calls` with tokens, cache hits, cost and latency.

### 3. The backend never knows what Telegram is

`api/` and `agent/` must not import or reference Telegram types. The bot is a client over `POST /chat`. A web client later should require zero backend changes.

Similarly, `queries/events.py` is the shared query module — it backs both the agent tools and the dev-time MCP server. Keep it free of agent or transport concerns.

### 4. First-party Anthropic API

```python
from anthropic import Anthropic
client = Anthropic()   # resolves ANTHROPIC_API_KEY from env
```

Canonical model IDs, no platform prefix. Env-driven so selection stays open:

```
CHAT_MODEL=claude-opus-5        # $5/$25 per MTok
EXTRACT_MODEL=claude-haiku-4-5  # $1/$5
```

**Settled — do not re-litigate:** a Claude Max subscription cannot back this application. Per claude.com/pricing, *"API access is separate and billed independently. The Max plan is for Claude's web, desktop, and mobile interfaces."* Max is a per-seat allowance for interactive human use on a rolling 5-hour window. It remains the right tool for *developing* here (prompt iteration, gold-set construction) — just not for serving users.

Bedrock was evaluated and dropped. First-party gives us the **Batch API** (halves extraction cost — extraction uses it for scheduled runs), plus MCP connector, programmatic tool calling and task budgets, none of which are available on Bedrock.

### 5. Current Claude API shape

Training priors on these are stale — several changed in 2025–26.

- `thinking: {type: "adaptive"}`. **Never `budget_tokens`** — removed on current models, returns 400.
- `output_config: {effort: "high"}` for chat, `"low"` for extraction. Inside `output_config`, not top-level.
- **No assistant prefill** — 400 on Opus 5. Use structured outputs or system-prompt instructions.
- Structured outputs are `output_config: {format: {...}}`, not the deprecated `output_format`.
- Parse tool inputs with `json.loads()`. Never string-match the serialized input.
- `strict: true` goes on the *tool definition*, not on `tool_choice`, and needs `additionalProperties: false` + `required`.
- Batch results arrive in **any order** — key by `custom_id`, never by position.

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

Python 3.12+, FastAPI, SQLAlchemy 2.0 + Alembic, Postgres 16 + PostGIS, `pydantic-settings` for config, `typer` for the CLI, `httpx` + `trafilatura` for crawling, `python-telegram-bot` for the client, OpenTelemetry + Langfuse for tracing, `pytest`. Docker Compose for local infra.

- All configuration through `config.py` / env. No hardcoded model IDs or endpoints. Prices live in `pricing.yaml`.
- Async throughout the request path; the ingest CLI may be sync where it's simpler.
- Crawl politely: descriptive User-Agent, per-host delay, honour `robots.txt` via `urllib.robotparser`, cap ~8 pages per venue.
- Skip extraction when `content_hash` is unchanged. Re-extracting unchanged pages is the main avoidable cost.
- Extraction runs synchronously during prompt iteration (fast feedback) and via the Batch API for scheduled runs (half price).

## Commands

Not yet implemented — this is the intended surface as of P0. Update as it lands.

```bash
docker compose up -d db langfuse
alembic upgrade head

python -m the_art_curator.cli smoke              # traced test call, verifies cost recording
python -m the_art_curator.cli sync-graf          # pull GRAF facts, snapshot events
python -m the_art_curator.cli crawl --pilot      # crawl the ~20 pilot venues
python -m the_art_curator.cli extract [--batch]  # venue pages -> exhibitions
python -m the_art_curator.cli eval-extraction    # score against the gold set
python -m the_art_curator.cli chat               # talk to the curator in the terminal

pytest
docker compose up -d                                # full stack incl. api + bot
```

`mcp_server.py` is a dev-time MCP surface over `queries/events.py`, for interrogating the corpus from Claude Code while building the P2 gold set. It is **not** on the serving path.

## Current state

Nothing implemented yet. Next step is **P0** (scaffold, schema, config, instrumented client, telemetry) — see PLAN.md § Build order.

First task in P0 is verifying the Langfuse SDK surface against live docs before writing code against it. If it has drifted, the OTel + Postgres layer stands alone and Langfuse can be dropped without data loss.

**P3 is the real checkpoint.** If the curator isn't good over 20 venues of data, scaling to 158 won't fix it. Don't build P4–P6 to avoid finding out.
