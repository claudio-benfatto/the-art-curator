# The Art Curator — v1 Plan

Scope, decisions and build order. Operational rules (the things that are easy to get wrong) live in [CLAUDE.md](CLAUDE.md).

**Status:** P0 done, signed off 2026-09-23 — PRs 1–10 merged, Terraform applied, smoke green locally and from CI. Next: P1 (§ 8).
**Last updated:** 2026-09-23 (rev. 7)

---

## 1. Goal

Answer *"I have Saturday afternoon in Barcelona, I like video art and installation, I'm near Poblenou — what should I see?"* with a curated, geographically-ordered itinerary, learning taste implicitly from conversation.

**v1 is a proof of concept.** The one question it answers: *is the curator actually any good?* No GDPR work, no public launch.

---

## 2. Decisions

| Area | Decision | Why | Rejected |
|---|---|---|---|
| First client | Telegram bot over a client-agnostic FastAPI `/chat` | Chat is the product's native shape; identity for free | Web-first (3–4× effort before first use) |
| Data sources | GRAF REST API for facts; LLM extraction from venue sites for prose | GRAF prose is copyrighted, facts aren't | GRAF-only; other aggregators (need entity resolution) |
| Copyright | Facts from GRAF, prose from nobody — enforced in the schema | See CLAUDE.md § 1 | Display-layer filtering |
| Crawl scope | ~20 pilot venues | Measure extraction accuracy before scaling | All 158 up front |
| Retrieval | SQL hard filters (date, PostGIS radius) → compact rows → hydrate shortlist | Exact filtering; shortlisting legible in traces | Stuffing 30–60 full records |
| Semantic ranking | pgvector over our own summaries, optional `query` on `search_events` — **P3** | Taste and "more like X" queries; one SQL query with PostGIS | Bedrock Knowledge Bases for events (no geo radius; would persist third-party prose) |
| Agent | Manual loop, hard cap 4 tool iterations, `iteration_count` traced | Refinement is the core interaction; cap bounds cost | Fixed workflow; multi-agent |
| LLM provider | **Amazon Bedrock** | Multi-vendor models (incl. embeddings) under one IAM/bill; managed AWS services later | First-party API (Claude-only, no embeddings); Claude Max (cannot back an app) |
| Chat model | Claude Opus 4.6 via `AnthropicBedrock` (InvokeModel) — **Claude-only** | Newest Claude this account can call (§ 9); keeps Claude's native API shape | Opus 5 / Mantle (refused); model-agnostic chat via Converse |
| Other models | Extraction, judge, embeddings: any Bedrock model, env-selected | Bulk work; swap by measurement | — |
| Extraction mode | On-demand, not batch | Bedrock batch drops tool use/structured output, min ~100 records/job | Bedrock batch inference |
| Observability | OTel + `llm_calls` in Postgres + Langfuse, from P0 | Cost attribution can't be retrofitted | Adding later |
| User profile | Implicit, agent-written, append-only facts | Forms lie about taste | Onboarding questionnaire |
| Stack | Python, FastAPI, Postgres + PostGIS + pgvector | Scraping/LLM ecosystem; exact geo | TypeScript; SQLite |
| Hosting | Local Docker Compose through P3, AWS (RDS/Aurora) after | Don't pay for infra before the curator is proven | AWS from P0 |
| Cloud infra | Terraform only (`infra/terraform/`), from P0 | Reproducible, reviewable, no console drift | Console/ad-hoc CLI; CDK/CloudFormation |
| CI | GitHub Actions; OIDC to AWS; `plan` on PR, gated `apply` on `main` | Repo already on GitHub; no stored cloud keys | Other CI; long-lived AWS keys in secrets |
| Languages | Reply in user's language; English canonical in storage | Simple dedup and search | Fixed language list |
| Routes | Nearest-neighbour order + Google Maps link | Within noise of real routing on a walkable district | Routing engine |

---

## 3. Architecture

```
graf.cat REST API ─┐
                   ├─► ingest ─► Postgres ────────► curator agent ─► FastAPI ─► Telegram
venue websites  ───┘   crawl +   PostGIS+pgvector   capped loop      /chat SSE   bot
                       extract   facts+summaries    + tools
                          │                            │
                          └──── Amazon Bedrock ────────┘   (Claude chat; extraction/embeddings any model)
                                       │
                                       └─► llm/client.py ─► OTel ─► llm_calls + Langfuse
```

Layering rules: backend never knows about Telegram · `queries/events.py` backs both agent tools and the dev MCP server · every model call (incl. embeddings) goes through `llm/client.py`.

```
.github/workflows/         CI: lint, tests, invariant checks, terraform plan/apply
infra/terraform/           all AWS resources (IAM for Bedrock in P0; RDS, S3, schedulers in P7)
infra/README.md            manual steps Terraform can't express
src/art_curator/
  config.py                pydantic-settings; model IDs, region, knobs
  db/                      models.py, session.py
  llm/client.py            the only model call site — Mantle (Claude) + bedrock-runtime (others), records cost
  llm/pricing.py           pricing.yaml → cost per call
  llm/extract.py           page → Exhibition[]
  llm/embed.py             summary → vector (via client.py)
  queries/events.py        shared query module
  ingest/                  graf.py, crawl.py, resolve.py
  agent/                   loop.py, tools.py, profile.py
  routes/itinerary.py
  obs/                     telemetry.py, metrics.py
  eval/                    extraction.py, curation.py
  api/main.py              POST /chat (SSE), GET /events, GET /venues
  clients/telegram_bot.py
  mcp_server.py            dev-time only
  cli.py                   smoke, sync-graf, crawl, extract, embed, eval-extraction, chat
```

---

## 4. Data model

Each table lands in the phase that first writes it — no speculative schema.

| Layer | Tables | Phase |
|---|---|---|
| Facts (source) | `venues` (`source`, source-scoped ids, name, address, `geom`, website/instagram URL, crawl flags) · `event_snapshots` (one row per event: `source`, source-scoped ids, venue, title, category, free, price range, URLs, first/last seen) · `event_occurrences` (start/end per occurrence, first/last seen) — **no description columns**; `source` is `"graf"` today, other event sources are additive | P0 |
| Derived (LLM) | `venue_pages` (url, status, `content_hash`, `raw_text`; **7-day TTL**) | P0 |
| | `exhibitions` (venue, title, dates, artists, `summary_en`, themes, media, `embedding`, source_url, confidence, model, hash) · `art_events` (same + `exhibition_id`) | P2 (`embedding` P3) |
| User | `users` · `profile_facts` (append-only, `superseded_by`) · `conversations` · `messages` · `interactions` · `itineraries` | P3 (`itineraries` P4) |
| Telemetry | `llm_calls` (trace/span, provider, model, purpose `chat/extract/embed/judge/smoke`, tokens incl. cache, cost, latency, stop_reason) | P0 |
| | `feedback` | P3 |

---

## 5. Agent tools

| Tool | Returns |
|---|---|
| `search_events(date_from, date_to, lat, lon, radius_km, categories, free_only, query?, limit)` | Compact rows (~50 tok): id, title, venue, date, one-liner, `distance_km` |
| `get_events(ids)` | Full records |
| `get_venue(venue_id)` | Detail, hours, website |
| `remember(key, value, confidence)` | Profile fact |
| `build_itinerary(event_ids, start_time, start_lat, start_lon)` | Order + maps link |

The model does no date/distance arithmetic. `query` (semantic ranking) lands in P3.

---

## 6. Cost model

Assumptions: ~3.3k in / 0.6k out per extracted page; Bedrock global endpoint at Anthropic list price (verify in P0; regional EU endpoints +10%).

| Item | Estimate |
|---|---|
| Extraction, pilot initial (160 pages, Haiku 4.5 on-demand) | ~$1 |
| Extraction, pilot nightly (~10% changed) | ~$3/mo |
| Extraction, all 158 venues | ~$8 initial, ~$24/mo |
| Chat, Opus 4.6, compact + hydrate + caching | ~$0.05/exchange (same per-token price as Opus 5; caching starts only past 4096 tokens) |
| **PoC total** (300 exchanges + pilot extraction) | **~$18/mo** |

Embeddings are noise. Chat starts on Opus 4.6; measure a cheaper or newer model at P3, whichever the account can call by then.

---

## 7. Ingestion

1. **`sync-graf`** — venues (568), users (158), events (58–104, live window → snapshot nightly). Facts only.
2. **`crawl`** — robots check, homepage → candidate pages by keyword heuristics, ≤8 pages/venue, `trafilatura` main text, `content_hash`.
3. **`extract`** — skip unchanged hash; schema-validated `Exhibition[]` with original English summary.
4. **`embed`** — embed `summary_en` + themes for new/changed exhibitions (P3).
5. **`resolve`** — match venue-extracted ↔ GRAF on venue + date overlap + fuzzy title.

**Pilot venues:** MACBA, Fundació Joan Miró, CaixaForum Barcelona, La Escocesa, ESPRONCEDA, àngels barcelona, ADN Galeria, Chiquita Room, ProjecteSD, Galeria Marc Domènech, RocioSantaCruz, Pigment Gallery, FUGA Gallery, Dilalica, ethall, Sala Parés, House of Chappaz, Galería Alegría, #plantauno, ACVic.

---

## 8. Build order

| Phase | Deliverable | Done when |
|---|---|---|
| **P0** | Compose (Postgres+PostGIS+pgvector, Langfuse), schema (facts, `venue_pages`, `llm_calls`), config, Terraform (state backend + least-privilege Bedrock IAM + GitHub OIDC role), GitHub Actions CI, instrumented Bedrock client, OTel + `llm_calls` | Smoke call to Haiku 4.5 (runtime Converse) traced with correct cost — Opus 5 / Mantle blocked pending AWS, see § 9 |
| **P1** | `sync-graf` | 568 venues with geometry, 146 URLs joined, idempotent snapshots |
| **P2** | `crawl` + `extract` + gold-set eval + dev MCP server | Extraction scored against ~30 hand-checked pages, re-runnable |
| **P3** | Agent loop, `/chat`, `cli chat`, feedback, **pgvector ranking** | Sensible curated answers; cache hit >80% on exchanges past Opus 4.6's 4096-token minimum; cost within 2× estimate; `iteration_count` recorded; semantic ranking A/B'd against filters-only |
| **P4** | `build_itinerary` | Ordered itinerary + working maps link |
| **P5** | Telegram bot | Streamed replies, location, 👍/👎, `/forget` |
| **P6** | Nightly scheduler + `venue_pages` purge | Unattended refresh; cache ≤7 days |
| **P7** | Move to AWS (RDS/Aurora, scheduled jobs) via Terraform | `terraform apply` from clean builds the environment; app is a connection-string swap |

**P3 is the real checkpoint.** If the curator isn't good over 20 venues, scaling won't fix it.

### P0 breakdown

**Pre-work (manual, nothing committed):** model access — ✗ Opus 5 and Mantle blocked, worked around (§ 9) · caching two-call script (scratchpad only — constraint 2) — ✓ verified on Opus 4.6 · Langfuse SDK/OTLP vs live docs — ✓ OTLP direct · Bedrock list prices — list assumed, unverified.

```
Python:     1 scaffold ─┬─ 2 CI baseline
                        └─ 3 compose ─ 4 schema ─┐
            5 pricing ───────────────────────────┼─ 6 client ─ 7 telemetry ─┐
Terraform:  8 TF bootstrap + Bedrock IAM ─ 9 OIDC + TF CI ──────────────────┴─ 10 smoke
```

| # | PR | Done when |
|---|---|---|
| 1 | Scaffold: uv, `pyproject.toml`, `src/art_curator/`, `config.py`, typer `cli.py`, ruff/pytest, `.env.example` | `ruff` clean, config test green |
| 2 | CI baseline: ruff, pytest + Postgres service, model-client grep; SHA-pinned actions | Green; a planted client import fails the grep |
| 3 | Compose: db image (PostGIS + pgvector, pushed to GHCR for CI) + Langfuse | Healthy; both extensions load |
| 4 | Schema + Alembic: `llm_calls`, `venues`, `event_snapshots`, `venue_pages`; schema half of `test_no_verbatim.py` | Migrates locally + CI; invariant test green |
| 5 | `pricing.yaml` + `llm/pricing.py` (in/out/cache-write/cache-read) | Pure unit tests vs hand-computed costs |
| 6 | `llm/client.py`: Mantle + Converse, one `llm_calls` row per call, static-prefix breakpoint helper, test stub | Stubbed tests: row per call, correct cost |
| 7 | `obs/telemetry.py`: OTel spans, trace/span ids on `llm_calls`, Langfuse export behind a flag, **extract-purpose bodies masked** (constraint 1) | Span ids land in row; masking test green |
| 8 | Terraform: README manual steps, S3 backend + lock, Bedrock policy scoped to model ARNs, app role | fmt/validate/plan clean; applied once |
| 9 | GitHub OIDC: plan-only role, apply role, `plan` on `infra/` PRs, gated `apply` on `main` | PR posts plan; apply needs approval |
| 10 | `cli smoke` (Haiku, runtime Converse) + `workflow_dispatch` smoke job | P0 done-when met |

---

## 9. Open risks

- **Extraction accuracy** across heterogeneous sites — measured in P2, not discovered in P5.
- **Newest-model access on Bedrock** — AWS (2026-09-23): eligibility depends on account usage history, is reassessed as usage grows, and isn't configurable. Opus 5 refused everywhere; Mantle refused for every model, including Haiku 4.5 (which works on the runtime endpoint) and Sonnet 5 (agreement applied). **Worked around, not blocking:** chat on Opus 4.6, extraction on Haiku 4.5, both via runtime inference profiles. Revisit at P3; Claude Platform on AWS if quality demands a newer model.
- **No structured outputs on Bedrock's Messages endpoint** — tool inputs validated with pydantic, `is_error` + retry on failure.
- ~~**Langfuse SDK surface unverified**~~ — sidestepped 2026-09-22: no Langfuse SDK; plain OTel exports to its OTLP endpoint (`/api/public/otel/v1/traces`, verified against current docs). Not yet exercised against the running Compose instance.
- **p95 latency** with three round trips — fallbacks: lower `effort`, then merge search + hydrate.
- **GRAF's live event window** (58–104 events) — long-running shows depend on crawling.
- **~12 Instagram-only venues** stay facts-only.
- **EU database right** — reduced, not zero; blocking for any public launch.

---

## 10. Deferred

- GDPR, consent, data export — before any launch.
- Web client, map rendering, real routing engine.
- Additional aggregators + entity resolution.
- Instagram-only coverage; venue outreach / opt-out list.
- **External art-knowledge RAG** (artists, movements) via Bedrock Knowledge Bases — needs licensable sources first (Wikipedia/Wikidata with attribution: yes; museum/gallery texts: no).
- **Bedrock batch inference** — only if extraction volume passes ~100 pages/run and structured output isn't needed.
- **Programmatic tool calling** — not native on Bedrock; would need a self-hosted sandbox.
- **Claude Platform on AWS** — Anthropic-operated, full API parity, AWS billing. The fallback if Bedrock's Claude feature gaps start to bite.
