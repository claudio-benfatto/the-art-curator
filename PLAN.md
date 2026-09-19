# The Art Curator — v1 Plan

Scope, decisions and build order. Operational rules for working in the codebase are in [CLAUDE.md](CLAUDE.md).

**Status:** approved, nothing implemented. Next step is P0.
**Last updated:** 2026-09-18 (rev. 3)

---

## 1. Context

There is no good way to answer *"I have Saturday afternoon in Barcelona, I like video art and installation, I'm near Poblenou — what should I see?"*

The information exists, but it is scattered across 130+ venue websites and an aggregator agenda, in three languages, with no notion of who is asking. Search gives you a list; it doesn't give you a plan for an afternoon.

The Art Curator ingests venue and exhibition data for Catalunya, learns a user's taste implicitly from conversation, and answers with a curated, geographically-ordered itinerary.

**v1 is a proof of concept** — exploration, not a product. No GDPR/consent work, no public launch. The question it exists to answer: *is the curator actually any good?* Everything else is deferred until that has an answer.

---

## 2. Decision record

Decisions with the reasoning that produced them. The rationale matters more than the choice — it's what tells you whether a decision should be revisited when circumstances change.

| Area | Decision | Why | Rejected |
|---|---|---|---|
| **First client** | Telegram bot over a client-agnostic FastAPI backend | Chat is the native shape of the product. No auth, hosting or frontend work; identity comes free from the Telegram user ID. | Web-first (3–4× the effort before anyone can use it) |
| **Data sources** | GRAF REST API for facts + LLM extraction from venue sites for prose | Legal, not effort: GRAF's descriptive text is copyrighted. Facts aren't. | GRAF-only (copyright exposure on prose); other aggregators (needs entity resolution, deferred) |
| **Crawl scope** | Pilot of ~20 venues, spread across type and geography | Keeps the first iteration debuggable and lets extraction accuracy be *measured* before scaling. | All 158 (long-tail extraction failures would dominate the schedule) |
| **Retrieval** | `search_events` returns compact rows; `get_events` hydrates the shortlist | Makes shortlisting explicit and legible in a trace. Paired with the caching layout, roughly halves chat cost. | Stuffing 30–60 full records (breaks the cache economics; hides the model's reasoning) |
| **Agent architecture** | Capped agent loop — manual, max 4 tool iterations, `iteration_count` traced | Conversation is the product; refinement is the core interaction and a rigid workflow handles turn 1 well and turn 3 badly. The cap bounds cost and latency. | Deterministic workflow (cheaper, but degrades on follow-ups); multi-agent (splits a coherent job, multiplies context) |
| **LLM provider** | First-party Console API | Unlocks the Batch API (halves extraction), simpler auth, usage dashboard. Nothing in the plan needs AWS. | Bedrock (no Batch/MCP/PTC/task budgets); Claude Max (cannot back an application — see §3) |
| **Models** | Opus 5 chat, Haiku 4.5 extraction, both env-swappable | Curation is the quality-critical path; extraction is bulk schema-constrained work where a 5× cheaper model suffices. | One model for both (extraction 5× dearer for no quality gain) |
| **Observability** | OTel + `llm_calls` in Postgres + self-hosted Langfuse, from day one | Retrofitted telemetry never gets cost attribution right, and P3's "is it good?" needs numbers. | Adding it later (the question P3 answers becomes unanswerable) |
| **User profile** | Implicit, agent-written; no onboarding form | People lie on forms about taste. A profile that is a side effect of talking stays honest. | Explicit questionnaire (friction + unreliable); stateless (no personalisation to evaluate) |
| **Stack** | Python, FastAPI, Postgres + PostGIS, Docker Compose | Strongest scraping/LLM ecosystem; PostGIS makes "within 2km of me" exact and trivial. | TypeScript (weaker ingest tooling); SQLite (loses PostGIS) |
| **Languages** | Reply in the user's language; English canonical in storage | One storage language keeps dedup and search simple; translation at response time costs nothing extra. | Fixed language list (awkward for a trilingual source corpus) |
| **Routes** | Curated ordered itinerary + Google Maps link | Over a walkable district, nearest-neighbour ordering is within noise of a real routing solve. | Routing engine (paid dependency, disproportionate for v1) |
| **Audience** | PoC for the author and a few others | Defers GDPR, consent and the database-right question until the curator is proven. | Public launch (significant compliance work before any validation) |

### Decisions that changed after review

Three were revised after the initial plan was approved, each prompted by a specific question rather than a general re-think:

1. **Bedrock → first-party API.** Prompted by asking whether a Claude Max subscription could back the app. It cannot, which forced a real look at provider economics. Bedrock lacks Batches, MCP, programmatic tool calling and task budgets; nothing in the plan needed AWS. Dropping it removed a provider abstraction entirely and halved extraction cost.
2. **Stuffed candidates → compact + hydrate.** Prompted by the cost estimate showing chat dominates spend. See §6 for the correction on where the saving actually comes from.
3. **Observability promoted to P0.** Explicitly requested as a day-one requirement rather than a later phase.

---

## 3. Research findings

Verified 2026-09-18. These changed the shape of the project materially — the original plan assumed all data would come from scraping 130+ venue sites.

**graf.cat is a WordPress site with an open REST API.** The factual layer is available structured, geocoded and multilingual:

- `wp/v2/event-venues` — **568 venues** with address, city, province, postcode and **lat/lon**. Geocoding is solved for free.
- `wp/v2/events` — **104 current occurrences** with ISO `start`/`end`, venue ref, category, free flag, price range, external URL, and trilingual title/description.
- `wp/v2/users` — **158 venue profiles**; **146 (92%) publish a website URL** — the crawler's seed list.
- `robots.txt` permits everything outside `/wp-admin/`.

**`events` caps at ~104 records regardless of date filters.** It is a live window, not an archive. Snapshot daily from day one or historical depth is permanently unrecoverable.

**A Claude Max subscription cannot back the application.** Per [claude.com/pricing](https://claude.com/pricing): *"API access is separate and billed independently. The Max plan is for Claude's web, desktop, and mobile interfaces."* Max is a per-seat allowance for interactive human use on a rolling 5-hour window, with no per-user isolation — a bot serving other people is programmatic API usage. Max remains the right tool for *building* this: P2's extraction-prompt iteration and gold-set construction is interactive Claude Code work at zero marginal cost, and that is the expensive-to-get-right part.

API gotchas (misspelled `longtitude`, singular/plural taxonomy naming, Cloudflare UA filtering) are in [CLAUDE.md § The GRAF API](CLAUDE.md#the-graf-api).

---

## 4. Copyright posture

**Facts from GRAF, prose from nobody.** This is a schema constraint, not a display-layer rule.

**Persisted from GRAF** — venue name, address, city, province, postcode, lat/lon, website URL, event start/end, category, free flag, price range, external URL. Facts carry no copyright.

**Never persisted from GRAF** — `desc_ca` / `desc_es` / `desc_en`, venue bios, or any other prose. No column exists to hold them.

**Venue site prose** — fetched into `venue_pages` with a 7-day TTL and a purge job, used only as extraction input. What survives is an LLM-written abstractive summary plus `source_url`. The extraction prompt forbids reproducing spans longer than ~25 words. Everything surfaced carries attribution and links out to the venue.

**Residual risk, knowingly accepted.** The EU *sui generis* database right (Dir. 96/9/EC) protects substantial systematic extraction from a database even of unoriginal facts. Taking only facts reduces but does not eliminate this. Acceptable at PoC scale with no publication; **must be revisited before any public launch**.

**Titles are persisted.** Exhibition and event titles are needed as dedup identifiers and are short factual names, generally below the originality threshold. They are the field closest to the line, and this was decided explicitly. The alternative — storing a hash and matching on venue + date overlap alone — makes fuzzy matching impossible (*"Anima Mundi"* would no longer match *"Anima mundi. Pensament gràfic"*), so duplicates would surface to users.

---

## 5. Architecture

```
graf.cat REST API ─┐
                   ├─► ingest ─► Postgres/PostGIS ─► curator agent ─► FastAPI ─► Telegram
venue websites  ───┘   (crawl +   (facts +            (capped loop      /chat      bot
                        extract)   summaries)          + tools)          SSE
                                        │
                                        └─► OTel ─► llm_calls + Langfuse
```

**Layering rules:**
- The backend never knows what a Telegram message is. The bot is a thin client over `/chat`; a web UI later is a second client, not a rewrite.
- `queries/events.py` is the shared query module, backing both the agent tools and the dev-time MCP server.
- Every model call goes through `llm/client.py`, where cost is recorded. There is no uninstrumented path.

### Repository layout

```
the_art_curator/
  docker-compose.yml           # postgres+postgis, langfuse, api, bot, scheduler
  pyproject.toml
  pricing.yaml                 # per-model token prices; config, not code
  .env.example
  alembic/versions/
  src/the_art_curator/
    config.py                  # pydantic-settings; model IDs and knobs env-driven
    db/models.py, db/session.py
    llm/client.py              # instrumented Anthropic() wrapper — the only call site
    llm/pricing.py             # loads pricing.yaml, computes cost per call
    llm/extract.py             # page -> Exhibition[]; sync + batch paths
    queries/events.py          # shared query module (tools + MCP server)
    ingest/graf.py             # REST sync, facts only
    ingest/crawl.py            # robots check, page discovery, text extraction
    ingest/resolve.py          # dedupe venue-sourced vs GRAF-sourced
    agent/loop.py              # manual agent loop, 4-iteration cap
    agent/tools.py             # search_events, get_events, get_venue, remember, build_itinerary
    agent/profile.py           # profile read/write
    routes/itinerary.py        # ordering + maps link
    obs/telemetry.py           # OTel setup, span helpers, Langfuse exporter
    obs/metrics.py             # cache hit rate, cost rollups, latency, iteration counts
    eval/extraction.py         # gold-set scoring
    eval/curation.py           # offline LLM judge over the feedback dataset
    api/main.py                # POST /chat (SSE), GET /events, GET /venues
    clients/telegram_bot.py
    mcp_server.py              # dev-time MCP surface — NOT on the serving path
    cli.py                     # typer: smoke, sync-graf, crawl, extract, eval-extraction, chat
  tests/
```

### Data model

**Factual layer (GRAF-sourced)**

- `venues` — `graf_venue_id`, `graf_author_id`, `name`, `address`, `city`, `province`, `postcode`, `geom geography(Point,4326)`, `website_url`, `instagram_url`, `crawl_enabled`, `robots_allowed`, `last_crawled_at`, `crawl_status`
- `graf_event_snapshots` — `graf_event_id`, `occurrence_id`, `venue_id`, `title`, `starts_at`, `ends_at`, `category`, `is_free`, `price_min`, `price_max`, `external_url`, `source_url`, `first_seen_at`, `last_seen_at`. Upsert on `occurrence_id`, bump `last_seen_at`. **No description columns.**

**Derived layer (LLM-written)**

- `venue_pages` — `venue_id`, `url`, `fetched_at`, `http_status`, `content_hash`, `raw_text`. **Purged at 7 days.**
- `exhibitions` — `venue_id`, `title`, `starts_at`, `ends_at`, `artists[]`, `summary_en`, `themes[]`, `media[]`, `source_url`, `source_type`, `confidence`, `extracted_at`, `extraction_model`, `content_hash`
- `art_events` — openings, talks, guided tours; same shape plus `exhibition_id`

**User layer**

- `users` — `telegram_id`, `locale`, `created_at`
- `profile_facts` — append-only: `user_id`, `key`, `value`, `confidence`, `source_message_id`, `superseded_by`
- `conversations`, `messages`, `interactions` (saved/dismissed/visited), `itineraries`

**Telemetry layer**

- `llm_calls` — `trace_id`, `span_id`, `model`, `purpose` (chat/extract/judge), `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `cost_usd`, `latency_ms`, `stop_reason`, `user_id?`, `venue_id?`, `created_at`
- `feedback` — `trace_id`, `user_id`, `kind` (explicit/implicit), `score`, `comment?`, `created_at`

Append-only `profile_facts` makes taste drift visible rather than overwriting it, and reduces `/forget` to a single delete. Keeping `llm_calls` in Postgres rather than only in Langfuse means spend can be SQL-joined against users and venues.

---

## 6. LLM architecture

### Why a loop and not a workflow

Most single queries fit a fixed pipeline — parse intent → SQL → curate → respond — and a workflow would be cheaper, faster and easier to evaluate stage by stage. But **conversation is the product.** This is a curator you talk to, and refinement (*"no, something quieter"*) is the core interaction. A rigid workflow handles turn 1 well and turn 3 badly.

The constraint that makes a loop safe: **a hard cap of 4 tool iterations**, with `iteration_count` on every trace. If the traces show the overwhelming majority of exchanges are search → hydrate → answer, the loop can collapse to a workflow later — from data, not a guess.

**Manual loop, ~60 lines.** Not the beta Tool Runner, not Managed Agents, not the Agent SDK (a coding harness — wrong tool). The manual loop is where the instrumentation seam naturally lives.

**No multi-agent orchestration.** Curation is one coherent job; splitting it across a router and a curator discards the context that makes curation good, and each agent carries its own window. The only second models that earn a place are offline: Haiku 4.5 for extraction (a separate pipeline, not orchestration) and an LLM judge in the eval harness.

### Tool surface

| Tool | Returns |
|---|---|
| `search_events(date_from, date_to, lat, lon, radius_km, categories, free_only, limit)` | Compact rows — `{id, title, venue, city, date, one_line, distance_km}`, ~50 tok each |
| `get_events(ids)` | Full records — summary, artists, themes, source_url |
| `get_venue(venue_id)` | Venue detail, hours, website |
| `remember(key, value, confidence)` | Writes a learned profile fact |
| `build_itinerary(event_ids, start_time, start_lat, start_lon)` | Ordering + maps link |

Dates and distances are computed in Postgres, where they are exact. **The model never does date or distance arithmetic.**

### Caching layout

**One explicit breakpoint on the static system prefix, plus top-level automatic caching for the conversation tail.**

```
system prompt → tool definitions → [explicit breakpoint] → conversation (auto-cached)
```

**A correction worth recording:** the compact-search split does *not* save ~40% on its own. It adds a third round trip, and each call resends the conversation, so raw input tokens actually rise (~12.5k vs ~10.7k per exchange). The saving comes from the caching layout above — the two are a single decision, not two independent optimisations. The stronger argument for the split was always behavioural: it makes shortlisting explicit and legible in a trace.

Two facts to design against:

- **Opus 5's minimum cacheable prefix is 512 tokens** — our ~2.5k static prefix caches fine.
- **Haiku 4.5's is 4096 tokens.** The extraction prompt is ~800, so it will silently never cache (`cache_creation_input_tokens: 0`, no error). Extraction estimates already assume no caching.

TTL: 5 minutes by default. The 1-hour TTL costs 2× on write and needs three reads to pay off; measure the start-to-start gap before switching.

### Other behaviour

**Language** — reply in the user's language. Summaries stored in English; the model translates at response time.

**Profile** — the system prompt instructs a `remember` call when the user reveals a durable preference (liked/disliked media, mobility, budget, pace, languages, venues visited) and never for one-off context. Facts are injected after the cache breakpoint.

**Itinerary** — nearest-neighbour ordering from the start point, 20 min/km walking estimate plus dwell time, sanity check against opening hours, Google Maps multi-stop URL.

---

## 7. Observability

Three concerns, deliberately kept separate.

### Tracing

One trace per exchange:

```
chat.exchange          user_id, session_id, client, lang, iteration_count
├─ llm.call            model, effort, tokens{in,out,cache_read,cache_write}, cost_usd, ms, stop_reason
├─ tool.search_events  filters, candidate_count, db_ms
├─ tool.get_events     ids, count
├─ llm.call
└─ tool.build_itinerary stop_count, total_km
```

Ingest is traced too — `ingest.extract` carries `venue_id`, `content_hash`, tokens, cost, `exhibitions_found`, mean confidence. That is how you find the venue costing real money nightly because a rotating banner means its page never hashes stable.

### Metrics

| Metric | Why it earns its place |
|---|---|
| `cache_read_tokens / input_tokens` | **The silent 10×.** One `datetime.now()` before the breakpoint multiplies cost with no error raised. Needs a threshold alarm, not just a dashboard. |
| `cost_usd` by user / day / phase | The thing that quietly runs away |
| p95 exchange latency | Telegram users abandon around 10s; Opus 5 + three round trips + thinking gets close |
| `iteration_count` distribution | The evidence for or against collapsing the loop to a workflow |

Plus tokens-per-exchange (history-growth drift) and extraction cost per venue.

**The commitment that makes this non-retrofittable:** token and cost recording lives *inside* `llm/client.py`. No code path reaches a model without being recorded. Prices live in `pricing.yaml` — config, not code.

### Feedback loops

Two, on different timescales:

- **Extraction quality (P2)** — ~30 hand-checked pages as a gold set, held as a Langfuse dataset. Every prompt change scores precision/recall per field (title, dates, artists) plus the no-verbatim check. A regression suite, not a vibes check.
- **Curation quality (P3+)** — inline 👍/👎 per answer → score on the trace. Implicit signals: an immediate rephrase is negative, saving an event is positive; the `interactions` table covers the rest. These accumulate into a (query + profile + candidates) → (answer, score) dataset, scored by an offline LLM judge, turning *"is it good?"* into a number.

### Dev-time MCP server

A thin MCP wrapper over `queries/events.py`, used from Claude Code — **not on the serving path**. Building the P2 gold set means comparing extracted exhibitions against source pages; through `psql` that is miserable, through Claude Code with your own data as tools it is fast.

---

## 8. Cost model

Assumptions: ~2.5k tokens of page text per venue page; ~40 compact candidates + ~6 hydrated; ~2k tokens history; adaptive thinking on; cache read 0.1× / write 1.25× input.

**Extraction — Haiku 4.5 via the Batch API (50% off)**

| | | |
|---|---|---|
| Per page | 3.3k in / 0.6k out | $0.0032 |
| Initial run, 20 venues × 8 pages | 160 pages | **$0.50** |
| Nightly, ~10% pages changed | 16 pages | **~$1.50/mo** |
| *All 158 venues* | *1,264 pages* | *$4 initial, ~$12/mo* |

Crawling itself is free — `httpx`, no model. Batch suits nightly extraction exactly (latency-insensitive). Extraction runs synchronously during P2 prompt iteration, where fast feedback matters more than price.

**Chat — per exchange**

| Configuration | Cost |
|---|---|
| Opus 5, naive (stuff 40 full records, single breakpoint) | ~$0.10 |
| Opus 5, compact + hydrate + caching layout | **~$0.05** |
| Sonnet 5, same | ~$0.02 |

**Realistic PoC total: ~$16/month** — 300 exchanges on Opus 5 plus batch extraction. Chat dominates; extraction is noise.

Start on Opus 5 to establish the quality bar at P3, then measure whether Sonnet 5 holds it. That is a measurement, not a guess.

---

## 9. Ingestion

1. **`sync-graf`** — paginate `event-venues` (568), `users` (158), `events` (104). Join venues to authors by name similarity to attach website URLs. Facts only. Runs nightly; snapshots accumulate the history the API won't retain.
2. **`crawl`** — per enabled venue: check `robots.txt`, fetch homepage, discover candidate pages by URL/anchor heuristics (`exposicion`, `exhibition`, `exposicions`, `agenda`, `actual`, `current`, `activitats`), fetch ≤8 pages with a polite delay and descriptive User-Agent, extract main text with `trafilatura`, store with `content_hash`.
3. **`extract`** — skip unchanged `content_hash`. Haiku 4.5 with a `strict: true` schema emitting `Exhibition[]`: title, artists, dates, **original English summary**, themes, media, confidence. `--batch` for scheduled runs, synchronous for prompt iteration.
4. **`resolve`** — match venue-extracted exhibitions to GRAF snapshots on venue + date overlap + fuzzy title. Prefer the venue record for prose and detail; prefer GRAF for dates when the venue page is vague. Unmatched records from either side are kept.

**Pilot venues (~20)** — MACBA, Fundació Joan Miró, CaixaForum Barcelona, La Escocesa, ESPRONCEDA, àngels barcelona, ADN Galeria, Chiquita Room, ProjecteSD, Galeria Marc Domènech, RocioSantaCruz, Pigment Gallery, FUGA Gallery, Dilalica, ethall, Sala Parés, House of Chappaz, Galería Alegría (L'Hospitalet), #plantauno (L'Hospitalet), ACVic (Vic). Instagram-only venues are `crawl_enabled=false` and stay facts-only.

---

## 10. Build order

P0 absorbs the cross-cutting work so every later phase inherits it.

| Phase | Deliverable | Done when |
|---|---|---|
| **P0** | Compose (incl. Langfuse), schema, Alembic, config, instrumented client, OTel + `llm_calls` + `pricing.yaml` | A smoke-test call appears as a trace with correct cost attributed |
| **P1** | `sync-graf` | 568 venues with geometry, 146 author URLs joined, snapshots upserting idempotently |
| **P2** | `crawl` + `extract` (sync + batch) + gold-set eval + dev MCP server | Extraction scored against ~30 hand-checked pages, re-runnable on every prompt change |
| **P3** | Capped agent loop, `POST /chat`, `cli chat`, feedback capture | A real taste query returns a sensible curated answer; cache hit rate >80%; cost/exchange within 2× of estimate; `iteration_count` distribution recorded |
| **P4** | `build_itinerary` | Ordered itinerary with timings and a working maps link |
| **P5** | Telegram bot | Chat, location sharing, streamed replies, 👍/👎 capture, `/forget` |
| **P6** | Nightly scheduler + `venue_pages` purge | Data refreshes unattended; cache never exceeds 7 days |

**P3 is the real checkpoint** — now with numbers attached rather than impressions. If the curator isn't good with 20 venues of data, scaling to 158 won't fix it. Resist building P4–P6 to avoid finding out.

---

## 11. Verification

```bash
docker compose up -d db langfuse && alembic upgrade head

# P0 — instrumentation is unavoidable and correct
python -m the_art_curator.cli smoke
psql -c "select model, input_tokens, cache_read_tokens, cost_usd from llm_calls;"
# and the trace visible in the Langfuse UI

# P1 — facts land, no prose columns exist
python -m the_art_curator.cli sync-graf
psql -c "select count(*) from venues where geom is not null;"   # expect ~568
psql -c "\d graf_event_snapshots"                                # assert: no desc/summary column

# P2 — extraction scored, not eyeballed
python -m the_art_curator.cli crawl --pilot
python -m the_art_curator.cli extract --batch
python -m the_art_curator.cli eval-extraction     # precision/recall per field vs gold set
pytest tests/test_no_verbatim.py

# P3 — the assertions that matter
python -m the_art_curator.cli chat
#  > "I'm free Saturday afternoon near Poblenou, I like video art and installation,
#     nothing that needs a ticket. Build me a route."
psql -c "select avg(cache_read_tokens::float/nullif(input_tokens,0)) from llm_calls where purpose='chat';"  -- >0.8
psql -c "select sum(cost_usd)/count(distinct trace_id) from llm_calls where purpose='chat';"                -- ~0.05

# P5 — end to end
docker compose up -d   # message the bot, share a location, confirm a streamed itinerary
psql -c "select key, value, confidence from profile_facts order by created_at desc limit 10;"
```

Tests to write alongside: `sync-graf` idempotency (running twice changes no row counts), robots.txt refusal honoured, `content_hash` skip prevents re-extraction, PostGIS radius correctness against known coordinates, the no-verbatim check, cost computation against a known token count, the 4-iteration cap enforced, and a CI grep asserting no `Anthropic()` construction outside `llm/client.py`.

---

## 12. Open risks

- **Extraction accuracy across heterogeneous sites** is the main technical unknown. The P2 gold set exists to measure it early rather than discover it at P5.
- **Langfuse SDK surface is unverified.** Confirming it against live docs is the first task in P0, before code is written against it. If it has drifted, the OTel + Postgres layer stands alone and Langfuse can be dropped without data loss.
- **Three round trips per exchange** raises p95 latency. If it exceeds ~10s, the fallbacks in order are: lower `effort` to `medium`, then merge search and hydrate back into one call and accept the tokens.
- **GRAF's 104-event window** means coverage of long-running exhibitions depends on venue crawling working well.
- **~12 venues are Instagram-only** and stay facts-only until there's a fallback.
- **Database right exposure** is reduced but not zero (§4). Fine for a PoC; blocking for a launch.

---

## 13. Deferred

Explicitly out of scope for v1, recorded so they aren't rediscovered as gaps:

- GDPR: privacy policy, consent flow, data export. Required before any public launch.
- Web client and map rendering.
- Additional aggregators (Ajuntament de Barcelona agenda, museum APIs) and the entity resolution they'd need.
- Real routing engine (OSRM / Directions API).
- Instagram-only venue coverage.
- Venue outreach and opt-out list — the posture that would make this genuinely clean to publish.
- **Programmatic tool calling** — would collapse search + hydrate into one round trip, cutting latency. Available on the first-party API; deferred because it drags the code-execution container into the request path.
