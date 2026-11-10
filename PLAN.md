# The Art Curator — v1 Plan

Scope, decisions and build order. Operational rules (the things that are easy to get wrong) live in [CLAUDE.md](CLAUDE.md).

**Status:** P0 done, signed off 2026-09-23. P1 built — PRs 11–16 (§ 8); done when the `GRAF` workflow runs green against live GRAF. P2 done: PRs 17–25 merged (24's baseline 2026-11-09; 25's judge not run live yet). P3 in progress: PR 26 built (§ 8).
**Last updated:** 2026-11-10 (rev. 13)

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
| Semantic ranking | pgvector over our own summaries, optional `query` on `search_events` — **P3, gated on search truncating** (§ 8) | Taste and "more like X" queries; one SQL query with PostGIS | Bedrock Knowledge Bases for events (no geo radius; would persist third-party prose) |
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
.github/workflows/         CI: lint, tests, invariant checks, terraform plan/apply; manual smoke + live GRAF
infra/terraform/           all AWS resources (IAM for Bedrock in P0; RDS, S3, schedulers in P7)
infra/README.md            manual steps Terraform can't express
src/art_curator/
  config.py                pydantic-settings; model IDs, region, knobs
  db/                      models.py, session.py
  llm/client.py            the only model call site — Mantle (Claude) + bedrock-runtime (others), records cost
  llm/pricing.py           pricing.yaml → cost per call
  llm/extract.py           page → Exhibition[]
  llm/embed.py             summary → vector (via client.py)
  queries/events.py        shared query module (incl. search_events, get_events, resolve_place)
  ingest/                  http.py, graf.py, matching.py, sync.py (P1) · crawl.py, discover.py, extraction.py (P2) · places.py, embedding.py (P3)
  agent/                   loop.py, tools.py, prompt.py, profile.py, store.py (conversations)
  routes/itinerary.py      P4
  obs/                     telemetry.py
  eval/                    extraction.py, runner.py (P2) · curation.py (P3)
  api/main.py              POST /chat (SSE), POST /feedback, DELETE /users/{id}/profile
  clients/telegram_bot.py  P5
  mcp_server.py            dev-time only
  cli.py                   smoke, sync-graf, discover, crawl, extract, eval-extraction · load-places, chat, eval-curation, embed
```

---

## 4. Data model

Each table lands in the phase that first writes it — no speculative schema.

| Layer | Tables | Phase |
|---|---|---|
| Facts (source) | `venues` (`source`, source-scoped ids, name, address, `geom`, website/instagram URL, crawl flags) · `event_snapshots` (one row per event: `source`, source-scoped ids, venue, title, category, free, price range, URLs, first/last seen) · `event_occurrences` (start/end per occurrence, first/last seen) — **no description columns**; `source` is `"graf"` today, other event sources are additive | P0 |
| Derived (LLM) | `venue_pages` (url, status, `content_hash`, `raw_text`; **7-day TTL**) | P0 |
| | `venue_pages.links` (a listing's numbered same-site links; same TTL as `raw_text`) | P2 |
| | `venue_pages.kind` (listing/detail) + extraction markers (status, hash, `EXTRACT_VERSION`, model, item count; no text) | P2 |
| | `venue_seeds` (venue, listing URL, page type, confidence, dated-item count, language, status, model, `discovered_at`; no free text) | P2 |
| | `exhibitions` (venue, title, dates, artists, `summary_en`, themes, media, source_url, confidence, model, hash) · `art_events` (same + `exhibition_id`) | P2 |
| | `embedding` + embed markers on both (model, hash of the embedded text) | P3, gated (§ 8) |
| Reference | `places` (name, aliases, kind barri/district/municipality, point; open data, CC BY) | P3 |
| User | `users` (`client`, `external_id`) · `conversations` (context block, `ended_at`) · `messages` (API content blocks as sent, `trace_id`, `iteration_count`) · `profile_facts` (append-only, `superseded_by`) · `feedback` | P3 |
| | `itineraries` | P4 |
| Telemetry | `llm_calls` (trace/span, provider, model, purpose `chat/discover/extract/embed/judge/smoke`, tokens incl. cache, cost, latency, stop_reason) | P0 |

`interactions` is dropped: `messages` + `feedback` cover what P3 and P5 read.

---

## 5. Agent tools

| Tool | Returns |
|---|---|
| `search_events(date_from, date_to, near? \| lat+lon?, radius_km?, kinds?, free_only?, query?, limit)` | Compact rows (~50 tok): prefixed id, kind, title, venue, start/end, `ends_in_days`, `distance_km`, `at_venue`, `is_free`, artists/media/themes; plus `total` matched. `near` lands in PR 27, `query` in 35 |
| `get_events(ids ≤ 10)` | `summary_en`, all facts, venue name/address/website, source/detail URL, price |
| `remember(topic, fact, replaces?)` | Profile fact id |
| `build_itinerary(event_ids, start_time, start_lat, start_lon)` | Order + maps link — P4 |

The model does no date/distance arithmetic: dates are ISO strings read off a calendar, places are names resolved in SQL. `get_venue` folded into `get_events` (§ 8, P3). `query` is gated (§ 8, P3).

---

## 6. Cost model

Assumptions: Bedrock global endpoint at Anthropic list price (verify in P0; regional EU endpoints +10%). The original extraction assumption, ~3.3k in / 0.6k out per page, was wrong by ~6×; the measured run replaces it.

**Extraction measured 2026-10-29** (`EXTRACT_VERSION` 1, 19 pilot listings, 25 calls): $0.72, **$0.038 a page**. Input 4.8k–24k tokens a call (median 9.7k, ~3.5k of it prompt + tool schema, so it never reaches Haiku's 4096-token caching minimum on its own). Output median 3.5k, and 4 calls cut off at 8192. Output was 64% of the cost: every show on a listing was described, and 82 of the 103 stored were over.

**Version 3, 2026-10-30: $0.015 a page across 45 pages, ~$0.022 a listing.** 45 of 45 ok (one after a retry), none cut off, no listing capped with nothing current; a second run made zero calls.

**Version 2, 2026-10-30** (facts only on listings, descriptions from a show's own page): listings $0.76 for 22 calls, no cheaper, and 2 still cut off at 16k. A listed item is still **~107 output tokens** (Pigment: 9,116 for 85 items); archive items had rarely carried a summary anyway. Listing cost scales with item count, and listings are archives: Pigment 77 of 79 over, ADN 78 of 81. **Detail pages: $0.0105 a page** (26 pages, 32 calls), which works. All 10 listings with rows run newest first, so version 3 keeps a listing's first 40 items in page order (§ 8, Extraction call). Expected, not measured: 40 exhibitions ≈ 4.3k output tokens, ~$0.03 with a 10k-token page; both lists full ≈ 10k, ~$0.06.

| Item | Estimate |
|---|---|
| Extraction, pilot initial | v1 $0.72 (19 listings); v2 $0.76 (listings) + $0.27 (26 detail pages); **v3 $0.69 for all 45 pages** (46 calls, none cut off), so ~$0.41 for the 19 listings at the v2 detail-page rate |
| Extraction, pilot nightly (~10% changed) | ~$0.07/night, ~$2/mo |
| Extraction, all 158 venues | ~8× the pilots: ~$5.5 initial, ~$16/mo nightly |
| Chat, Opus 4.6, compact + hydrate + caching | ~$0.05/exchange (same per-token price as Opus 5; caching starts only past 4096 tokens) — **likely low**: a first exchange of 3 calls is ~3k + 6k + 9k input tokens with nothing cached under 4096, ≈ $0.07–0.09 in + ~$0.04 out. PR 31 measures it |
| **PoC total** (300 exchanges + pilot extraction) | **~$18/mo** |

Embeddings are noise. Chat starts on Opus 4.6; measure a cheaper or newer model at P3, whichever the account can call by then.

---

## 7. Ingestion

1. **`sync-graf`** — venue terms (~570), profiles (158), events (live window, 56–104 → snapshot). Facts only. A term (space) is joined to a profile (organisation, carries the URL); the join and `is_pilot` are recomputed every run, never accumulated (CLAUDE.md § The GRAF API).
2. **`crawl`** — robots check, accepted seeds + detail URLs from extraction, ≤8 pages/site, all visible text, `content_hash` (§ 8).
3. **`extract`** — skip unchanged hash; schema-validated `Exhibition[]` with original English summary.
4. **`embed`** — embed `summary_en` + themes for new/changed exhibitions (P3).
5. **`resolve`** — match venue-extracted ↔ GRAF on venue + date overlap + fuzzy title.

**Pilot venues:** MACBA, Fundació Joan Miró, CaixaForum Barcelona, La Escocesa, ESPRONCEDA, àngels barcelona, ADN Galeria, Chiquita Room, ProjecteSD, Galeria Marc Domènech, RocioSantaCruz, Pigment Gallery, FUGA Gallery, Dilalica, ethall, Sala Parés, House of Chappaz, Galería Alegría, #plantauno, ACVic.
All 20 resolve; 18 have a crawlable URL — ProjecteSD (blank profile URL) and #plantauno (no profile) stay facts-only.

---

## 8. Build order

| Phase | Deliverable | Done when |
|---|---|---|
| **P0** | Compose (Postgres+PostGIS+pgvector, Langfuse), schema (facts, `venue_pages`, `llm_calls`), config, Terraform (state backend + least-privilege Bedrock IAM + GitHub OIDC role), GitHub Actions CI, instrumented Bedrock client, OTel + `llm_calls` | Smoke call to Haiku 4.5 (runtime Converse) traced with correct cost — Opus 5 / Mantle blocked pending AWS, see § 9 |
| **P1** | `sync-graf` | ~570 venue terms, ~556 with geometry (`0,0` → NULL); ≥145 joined to a website, ≤2 ambiguous; 20/20 pilots; a second run changes no ids, joins or `first_seen_at` |
| **P2** | `crawl` + `extract` + gold-set eval + dev MCP server | Extraction scored against ~30 hand-checked pages, re-runnable |
| **P3** | Agent loop, `/chat`, `cli chat`, profile, feedback, curation eval, pgvector ranking (gated) | `eval-curation` baseline (§ 8, P3 breakdown); cache hit >80% on calls past Opus 4.6's 4096-token minimum; cost within 2× estimate; `iteration_count` recorded; semantic ranking A/B'd, or skipped on measured evidence |
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

### P1 breakdown

```
11 http + config ─ 12 graf parse + scrub + fixtures ─┬─ 14 upsert (+0002) ─ 15 cli ─ 16 live job + docs
                   13 matcher (pure) ─────────────────┘
```

| # | PR | Done when |
|---|---|---|
| 11 | `ingest/http.py`: polite client, retries, `x-wp-totalpages` pagination; `HTTP_*` settings | Short page, 429 + `Retry-After`, 3× 500 tested with zero wall-clock sleep |
| 12 | `ingest/graf.py`: pydantic models, `scrub()` at capture, trimmed fixtures + `identity.json` | Fixture prose scan in `test_no_verbatim.py`, same commit as the first fixture |
| 13 | `ingest/matching.py`: slug → base-slug → name → guarded containment; `classify_url`; pilots | 156 joined / 1 ambiguous over `identity.json`; 20/20 pilots |
| 14 | `ingest/sync.py` `write_graf()` + migration 0002 (one profile, many terms) | Second run changes no ids or `first_seen_at`; nothing deleted |
| 15 | `cli sync-graf`: report, `--dry-run` (write + rollback), `--min-venues`, `--record` | Report printed; dry run writes nothing; unmatched pilot exits 1 |
| 16 | `graf.yml` (manual, live GRAF, two syncs + SQL floors) + docs | Green against live GRAF |

### P2 breakdown

**Scope:** the 18 crawlable pilots. **Done when:** `eval-extraction` scores extraction against ~30 hand-labelled pages from ≥12 venues (≥3 of them pages with no current show), prints per-field scores and cost, and a second run while the pages are unchanged gives the same scores within noise. Every crawlable pilot has an accepted seed (automatic or human-resolved), and discovery accuracy against the human labels is reported. The first scores are the **baseline, not a gate**. They decide whether P3 goes ahead over these 20 venues.

**Design decisions** (each is cheap to get right now and expensive to retrofit):

| Area | Decision | Why |
|---|---|---|
| Crawl unit | One crawl per distinct `website_url` among `is_pilot AND crawl_enabled` venues; pages hang off the pilot venue row | MACBA is four terms and one site; crawling per term fetches it four times |
| Robots | `robots.txt` fetched **through `PoliteClient`** (so UA, delay and retries apply), parsed with `robotparser.parse()`. 404 → allow all; 401/403 → disallow all; 5xx or fetch failure → skip the site this run. `Crawl-delay` raises the per-host delay, never lowers it | `RobotFileParser.read()` uses `urllib` with its own UA, which bypasses all of it |
| User-Agent | `art-curator/0.1 (+<repo URL>)`, so a webmaster can find out who we are | Descriptive UA is a CLAUDE.md requirement; the current value is just a name |
| Discovery | **A monthly two-pass Haiku step** that finds each site's *listing* pages (current/upcoming exhibitions, agenda). **Pass 1, links:** the homepage's same-site links (text, URL, nav/main/footer), numbered; Haiku returns ≤3 candidates **by number**, so it cannot invent a URL. **Pass 2, pages:** each candidate is fetched and its text goes to Haiku, which classifies it (`current_listing` / `agenda` / `past_archive` / `single_show` / `other`, count of dated items, language) and gives a confidence. GRAF's current titles and on-host `_event_web_*` URLs for the venue go into both prompts **as hints only**; nothing is accepted or rejected on them | Link structure varies too much across sites for keyword rules. A listing page rarely moves, so finding it monthly rather than nightly keeps cost and churn low |
| Routing | Code, not the model, routes each candidate: `current_listing`/`agenda` with high confidence → **accepted**; `other`/`past_archive` with high confidence → **rejected**; everything else, a site with no accepted candidate, or a proposal that differs from the seed already in use → **ambiguous**. Thresholds are calibrated against the human labels (PR 24) | Self-reported confidence is poorly calibrated; a categorical page type plus a coarse confidence is easier to check and tune |
| Human in the loop | Ambiguous sites are listed in the `discover` report. A person resolves them in committed **`ingest/seeds.yaml`** (`pin` or `reject` a URL per venue), reviewed in diffs like `PILOT_VENUES`. A human decision outlives the monthly pass: a pinned venue is not re-discovered unless its seed breaks, and a rejected URL is never proposed again | The machine handles the clear cases; the human sees only the cases it cannot decide |
| Seeds | `venue_seeds` holds the machine state: venue, URL, page type, confidence, status, model, `discovered_at`. **No free-text reasoning column**: pass 2 sees page text, so anything it writes in prose could quote it | Structured fields only keep third-party text out (constraint 1) |
| Detail pages | Not discovered by link rules. Extraction of a listing page returns each show's `detail_url` (a fact), and the next crawl fetches detail pages for shows not over and not yet fetched. A detail page is where a show is described (`summary_en`, themes, media); a listing records facts only. The model names it **by link number** from the page's numbered links, as in discovery. `raw_text` carries no URLs, so `crawl` stores an `ok` listing's links in `venue_pages.links` (`[{number, url, text}]`), purged with `raw_text` and tied to it by a CHECK. Detail pages store none. ≤8 pages per site per crawl, listings first; one language prefix per site | The step that understands the page picks the links. One language avoids extracting the same show three times |
| Broken seeds | The crawl report flags a seed that 404s, redirects to the homepage, or yields zero items after previously yielding some. That venue is re-discovered on the next `discover` run instead of waiting a month | A site redesign becomes a visible broken seed, not a silent gap |
| Text | **All visible text** (`pages.page_text`: `html2txt` with its cleaning off, after removing scripts, styles and other never-rendered elements), whitespace-normalised, capped at ~24k chars per page. `content_hash` = sha256 of that text, not of the HTML | Main-text extraction drops the dates, which sit in headers and sidebars: MACBA listing 48 → 0, àngels 458 → 0, MACBA/ADN/Rocío detail pages → 0 (probe, 2026-10-03). `html2txt`'s default cleaning drops any div whose class contains "footer": FUGA's listing 587 → 129 words, Sala Parés 520 → 125 (2026-10-07). HTML carries nonces, so its hash would change every fetch |
| Unusable sites | Fewer than ~50 words after extraction → status `thin` (JS-rendered); 403 / challenge → `blocked`. Both are recorded and reported, and the venue stays facts-only. **No headless browser in P2; never bypass a bot challenge** | Surveyed 2026-10-03: 5 of 18 unusable (§ 9). Accepted as coverage gaps |
| TTL | `cli purge-pages` lands with `crawl`, and `crawl` runs it first every time. P6 only schedules it | `raw_text` is first written in P2; the 7-day TTL has to hold from the first write, not from P6 |
| Extraction call | Haiku 4.5 over Converse, thinking off, temperature 0. One forced tool (`toolChoice`) carries `Exhibition[]` and `ArtEvent[]`. Two prompts by page kind: a **listing** records facts only (title, artists, dates, `at_venue`, link, confidence) for its **first 40 items a list in page order**, more dropped not failed, 16k output tokens; one that hits the cap with nothing current is reported (the page may run oldest first), a **show's own page** adds `summary_en`, themes and media (≤40 items, 8k). Pydantic-validated, one retry on failure, shared with the verbatim guard: ≤2 calls per page. An answer cut off at `maxTokens` fails the page without a retry. botocore retries off and a 300 s read timeout, so every request is one `llm_calls` row | No structured outputs on Bedrock (§ 9). A forced tool is the closest equivalent. Describing every listed show cost $0.038 a page and cut 4 of 19 pilot listings off at 8k tokens. Facts alone are still ~107 tokens an item, and listings are archives that run newest first (§ 6) |
| Dates | The model transcribes dates **as written**, `{day, month, year?}`. Code infers a missing year from `fetched_at` (the nearest year where end ≥ start) and expands month-only dates to month bounds | Year inference is arithmetic (constraint 8), and Catalan pages routinely omit the year |
| Verbatim guard | At runtime, the longest shared word run between `summary_en` and the page text is capped at **20 words** (margin under the prompt's ~25). On a breach, retry once. On a second breach, or a retry that comes back invalid, keep the facts, store `summary_en = NULL` and count it in the report. The extraction half of `test_no_verbatim.py` plants a copied span and checks it is caught | The prompt asks; the guard enforces |
| Re-extraction | A page is re-extracted when `(content_hash, EXTRACT_VERSION, model)` differs from what it was last extracted with. `EXTRACT_VERSION` is a constant, bumped by hand when the prompt or schema changes; a test pins a hash of both to it | Unchanged pages are the main avoidable cost; a prompt change must still invalidate them |
| Dedup | Upsert `exhibitions` on `(venue_id, title_key)`, where `title_key` is the normalised title; `art_events` adds the start date (`NULLS NOT DISTINCT`). A show seen again takes each field's newest non-empty value, except `summary_en`: the longer stays. `first_seen_at` / `last_seen_at` as for snapshots; nothing is deleted | Index and detail pages describe the same show. A detail page is fetched once, so a whole-record "richer wins" would let it shadow later corrections on the listing |
| Test fixtures | Crawl tests use **hand-written synthetic HTML** (`tests/fixtures/crawl/*.html`), never recorded pages. `test_no_verbatim.py` requires a `<!-- synthetic -->` marker and a size cap on every HTML fixture | A recorded page is third-party prose in git (constraint 1) |
| Gold set | **Labels only, facts only**: url, `content_hash`, and the expected `{kind, title, dates, artists, at_venue}` per item, in `eval/gold/extraction.yaml`; `confirmed` once a person checked it, and `--draft` writes unconfirmed ones. An archive-like listing may be `partial` (its first 10 of each kind): recall and fields, not precision. No page text and no summaries. Eval reads `raw_text` from `venue_pages` or re-crawls; a page whose hash no longer matches its label is **stale**, reported and skipped | The page text cannot be kept past 7 days, so the gold set decays as venues update. That is the price of constraint 1, and re-labelling is part of the job |
| Summary quality | Not in the gold set. Scored by an optional faithfulness judge (PR 25) and bounded by the verbatim guard | A hand-written reference summary measures wording, not truth |
| Dev MCP | `queries/events.py` + `mcp_server.py` expose venues, page **metadata** (status, hash, word count, extraction state), exhibitions, events and `llm_calls` cost. **Never `raw_text`.** Labelling reads the live page in a browser | Keeps third-party text out of conversation transcripts |

```
17 http: text + robots ─ 18 discover (stubbed) ─ 19 0004 + cli discover + seeds.yaml ─ 20 0005 + cli crawl + purge-pages ─┐
21 extract: schema, prompt, dates, guard (stubbed) ────────────────────────────────────────────────────────────────────────┴─ 21a 0006 page links ─ 22 0007 + cli extract ─┬─ 23 queries + dev MCP
                                                                                                                                                                          └─ 24 gold set + eval ─ 25 judge (optional)
```

| # | PR | Done when |
|---|---|---|
| 17 | `ingest/http.py`: `get_text` (per-request `Accept`, content-type check, body-size cap), `RobotsPolicy` (fetch via `PoliteClient`, cached per host, `Crawl-delay`); descriptive UA default; `trafilatura` dependency | Robots 404 / 403 / 5xx / disallow / `Crawl-delay` tested with zero wall-clock sleep |
| 18 | `ingest/discover.py`: link collection (same host, numbered, nav/main/footer), pass 1 and pass 2 prompts with GRAF hints, forced tools, pydantic validation, routing rules. Migration 0003: `discover` added to `llm_calls.purpose` (masked bodies, like extract). Synthetic HTML fixtures; marker check in `test_no_verbatim.py` | Stubbed: invented link number rejected; each routing outcome covered; no GRAF prose in either prompt |
| 19 | Migration 0004: `venue_seeds`. `ingest/seeds.yaml` overrides. `cli discover` (`--venue`, `--dry-run`, report listing accepted / rejected / ambiguous) | Run on the live pilots: every crawlable site has a status; ambiguous ones resolved in `seeds.yaml`. **This run is the site survey** |
| 20 | Migration 0005: `venue_pages.status`. `cli crawl --pilot`: pinned seeds, else accepted ones (detail URLs from PR 22), visible text + hash, `ok/thin/blocked/robots/error` statuses, `venue_pages` upsert by url, broken-seed flags (gone, redirects home), `--dry-run`, per-site report. `cli purge-pages` | Second crawl of unchanged pages changes no hashes; every page older than 7 days has `raw_text` NULL |
| 21 | `llm/extract.py`: pydantic `Exhibition` / `ArtEvent` (with `detail_url`, `at_venue`), versioned prompt, forced tool, one retry, date resolution, verbatim guard; extraction half of `test_no_verbatim.py`. `llm/tools.py`: the forced-tool pieces, shared with discovery | Stubbed: bad JSON → retry → ok; copied span → `summary_en` NULL; year inference table-tested |
| 21a | Migration 0006: `venue_pages.links`. `crawl` stores an `ok` listing's numbered links; `purge-pages` NULLs them with the text | Links are stored for listings only, never without `raw_text` (CHECK), and are gone after 7 days |
| 22 | Migration 0007: `exhibitions`, `art_events`, extraction markers on `venue_pages`; classified in `test_no_verbatim.py`. `cli extract` (skip unchanged, `--venue`, `--limit`, `--dry-run`, report with cost from `llm_calls`). `extract` passes a listing's stored links; crawl follows `detail_url` | Second run on unchanged pages makes zero model calls (met 2026-10-29). Version 3 on the pilots: no listing cut off, a listing costs less than v1's measured $0.038, and the cost is recorded in § 6 (met 2026-10-30: ~$0.022 a listing) |
| 23 | `queries/events.py` (read-only, async, no agent/transport concerns) + `mcp_server.py` (`mcp` in the dev group) | Usable from Claude Code; no tool returns `raw_text` (tested) |
| 24 | `eval/extraction.py` (pure scorer) + `cli eval-extraction` + labelling ~30 pages and the correct listing URL per venue | Done-when above. Reports item recall and precision (title match), date and artist accuracy on matched items, false positives on negative pages, stale count and cost. For discovery: accuracy vs the human-labelled listings, the ambiguous rate (human load), and **accepted-but-wrong** (the dangerous case) | Met 2026-11-09: 36 pages, 14 venues, 10 with no current show; runs agree except on one page (baseline below) |
| 25 | *(optional)* Summary faithfulness judge, `purpose=judge`, `JUDGE_MODEL` defaulting to the chat model (already in IAM and `pricing.yaml`). `llm/judge.py` + `eval-extraction --judge`; unsupported claims must be passages of the summary, so no page text is printed | Per-summary supported/unsupported in the eval report. Built; not run live yet |

PR 21 depends only on the client and can run in parallel with 17–20. `discover` runs by hand in P2; P6 schedules it monthly alongside the nightly crawl. PR 24 is mostly human work: Claude Code can draft labels through the MCP server and the live page, but "hand-checked" means a person confirms every item.

**Provisional quality bar** (an input to the P3 decision, not a gate): item recall ≥ 0.8, precision ≥ 0.9, date accuracy ≥ 0.9 on matched items, zero items invented on negative pages.

**Baseline, 2026-11-09** (`EXTRACT_VERSION` 3, Haiku 4.5; gold set: 36 pages, 14 venues, 200 items, 12 listings `partial`, 2 empty):

| Metric | Score | Bar |
|---|---|---|
| Recall | 0.98 (197/200) | met |
| Precision (24 non-partial pages) | 1.00 (60/60) or 0.78 (60/77) | met in 2 of 3 runs |
| Dates / artists / at_venue / kind, on matched | 0.98 / 0.99 / 0.99 / 1.00 | met |
| Invented on empty pages | 0 on 2 | met |
| Discovery | 14 of 16 correct, 2 ambiguous, **0 accepted-but-wrong** | — |
| Cost | $0.61 a run (36 pages, 39–40 calls) | — |

- **Run-to-run noise is one page.** Three runs scored every page alike except MACBA's `un-museu-fora-del-museu`, one show of 17 works shown around the city: 1 item in two runs, 18 in the third (and in the draft). At temperature 0. Those 17 extras are the whole precision swing.
- **Repeated dates are merged:** La Escocesa's events on separate days come back as one run (11-05 → 12-01), so 3 misses and 2 date errors. The prompt's "record each one once" is the likely cause; fixing it bumps `EXTRACT_VERSION`.
- Smaller: a year typo on the page drops a run's dates (FUGA, `resolve_dates` refuses end < start), an artist named only in the title is left out (Miró), one `at_venue` (Chiquita Room).
- Discovery calibration is too thin to retune: `agenda medium` 0 of 1, `other high` 1 of 2 were listings; `current_listing high` 16 of 16.

**Cost:** discovery ~$0.02 per site per month (≈5k tokens for pass 1, ≈4k per candidate for pass 2), so ~$0.40/month for the pilots and ~$3/month at 158 venues. First pilot extraction: $0.72 for 19 listings on version 1 (§ 6). Each eval run $0.61 over the 36-page gold set (measured), plus ~$1 with the Opus judge. Crawling is free.

### P3 breakdown

**Scope:** what P2 stored for the pilots, plus GRAF's live events (region-wide, facts only). The gold set alone holds 36 current exhibitions and 24 current events over 12 venues; GRAF adds 56–104. **Done when:** `eval-curation` runs ~20 scripted conversations and reports: zero recommended items the tools did not return, zero recommended items outside the asked dates, area or price, and a person judges ≥ 16 of 20 sensible; cache reads > 80% of input on calls whose prefix passes 4096 tokens; cost per exchange within 2× § 6; `iteration_count` on every trace, with its distribution; the semantic A/B, or the truncation measurement that skipped it. As in P2, the first numbers are **evidence for P4, not a merge gate**.

**Design decisions:**

| Area | Decision | Why |
|---|---|---|
| Search surface | One `search_events` over `exhibitions`, `art_events` and GRAF occurrences, one row shape, ids prefixed by source (`x:12`, `a:7`, `g:301`) | The model ranks one list; `get_events` must know which table an id names |
| Compact rows | id, kind, title, venue, start/end, `ends_in_days`, `distance_km`, artists, media, themes. **No summary.** Plus `total` matched, so truncation shows | ~50 tokens a row; the summary is what hydrating is for |
| Hydrate | `get_events(ids ≤ 10)`: `summary_en`, all facts, venue name/address/website, `detail_url`/`source_url`, GRAF price and free flag. **`get_venue` folds into it** | A separate venue tool costs a round and repeats what hydrate carries; there are no opening hours to return |
| GRAF ↔ extracted | A GRAF event is hidden when an extracted item at a venue of the same profile has the same normalised title (GRAF's or its `title_en`) and overlapping dates. Fuzzy `resolve` (§ 7) stays deferred until traces show duplicates | One exact rule in SQL; the fuzzy one is guesswork without traces |
| GRAF window | Search only occurrences the latest sync saw: a superseded occurrence or an event that left the window early is not listed | Both are what `last_seen_at` is for (CLAUDE.md § The GRAF API) |
| Runs and place | Exhibition: open start and end when missing. Event: no end = its start day; no date = never searched. Off-venue or online: no distance, never in a radius | The same rules as the `not_over_on` queries; a venue's point is not where an off-site show is |
| Price | `free_only` drops only what is *known* paid; extracted rows have `is_free` NULL | Venue pages give no price; dropping unknowns would empty the pilots |
| Kinds | One vocabulary: `exhibition` + `EVENT_KINDS`; GRAF `event_category` ids mapped in code (`graf.CATEGORY_KINDS`, 10 ids read off `class_list` 2026-11-10; unknown → `other`) | The filter means the same thing on every source; GRAF serves no category taxonomy over REST |
| Location | `near: <place>` resolved in SQL against `places` (Barcelona barris and districts, Catalan municipalities) and venue names, or `lat`/`lon` when the client sends a location (P5). Unknown or ambiguous → `is_error` with candidates | Geocoding from model memory is distance estimation (CLAUDE.md § 8) |
| Dates | Tools take ISO dates only. Each conversation opens with today and a 14-day calendar (weekday → date) in `SITE_TIMEZONE`. A conversation ends after 6 h idle or at local midnight | "Saturday" is a lookup, not arithmetic; a calendar cannot outlive its day |
| Prompt layout | System prompt fully static. Per-conversation context (calendar, profile snapshot, client location) is the first user turn, frozen | The static prefix stays shareable across users if chat moves to a 512-token-minimum model (CLAUDE.md § 6) |
| History | `messages` keeps the API content blocks exactly as sent and received, thinking blocks and signatures included, and the loop replays them unchanged | Opus 4.5+ keeps prior-turn thinking in context; any edit to history breaks the cache from that point |
| Loop | Manual, ≤ 4 tool rounds (CLAUDE.md § 7). After the 4th, one last call with `tool_choice: none`, so a reply always comes. `iteration_count` = tool rounds, on the trace and the assistant message | The cap bounds cost without leaving the user unanswered |
| Thinking / effort | `thinking: adaptive`; effort from `CHAT_EFFORT` (start `medium`), fixed per deployment; the eval sweeps it | A per-request change invalidates the messages cache |
| SDK | Pin `anthropic[bedrock]<2` | 2.0.0 sends API version `2026-10-21`, which Bedrock accepts only for Claude 4.7+: chat on Opus 4.6 would break on a routine lock upgrade |
| Profile | `remember(topic, fact, replaces?)` → `profile_facts`, append-only; `topic` from a small enum (media, themes, area, budget, language, access, other). The snapshot, with fact ids, is rendered at conversation start; a mid-conversation `remember` shows through its tool result and is not re-rendered | Re-rendering would edit the prefix |
| Forget | `DELETE /users/{id}/profile` deletes the user's facts and conversations. Lands in P3 | P5 stays client-only |
| Identity | `users(client, external_id)` unique, `client` an opaque string; one bearer token per deployment (`API_TOKEN`), API bound to localhost until P7 | The backend never learns what the client is (CLAUDE.md § 3) |
| Streaming | `LlmClient.stream_message`: one `llm_calls` row when the stream ends or breaks. SSE events: `status` (tool running), `text` deltas, `done` (message id, `iteration_count`). Thinking is not forwarded. Needs `bedrock:InvokeModelWithResponseStream`, which the app role lacks | A row on every path, as for unstreamed calls |
| Semantic ranking | **Gated:** built only if `eval-curation` shows `search_events` truncating (`total` > returned) in a real share of scenarios. Then: embed `summary_en` + themes + media (GRAF rows have no summary, so none); the model writes `query` in English, so the embedding model need not be multilingual; `query` reorders the filtered set, unembedded rows last | A few dozen current extracted items plus GRAF's window fit in one result; ranking cannot beat "the model sees everything" until they stop fitting |
| Curation eval | `eval/gold/curation.yaml`: ~20 scripted conversations (persona, turns, constraints, relative dates only), placed where the pilots are. Per exchange: titles in the reply matched to tool results (grounding), hard constraints checked in SQL on the matched ids, a judge (`JUDGE_MODEL`) for taste fit and invented items, and iteration count, cost, cache share and latency. Transcripts go to a file outside the repo for the human read. Runs on the live DB, so arms are compared on the same day | Expected answers rot as shows close; constraints do not |

```
26 search_events + get_events ─┬─ 28 agent core + cli chat ─ 29 conversations ─ 30 profile ─ 31 curation eval ─┬─ 35 embeddings (gated) ─ 36 sign-off
27 places ─────────────────────┘                                                                                │
32 Terraform: streaming + model candidates ─ 33 streamed client ─ 34 /chat API ─────────────────────────────────┘
```

| # | PR | Done when |
|---|---|---|
| 26 | `queries/events.py`: `search_events` + `get_events` (three sources, prefixed ids, dates and distance in SQL, `total`, GRAF overlap rule, kind mapping); both as MCP tools, listed in `test_no_verbatim.py` | DB tests: radius, date overlap, shows with no end date, GRAF suppression, `total` under truncation |
| 27 | Migration 0008: `places`. `cli load-places` from open data (source and CC BY attribution in the loader), `resolve_place` (case and accents ignored; venue names count) | "Poblenou", "poblenou", "MACBA" and "Vic" resolve; an ambiguous name returns candidates |
| 28 | `agent/`: tool schemas + pydantic validation, static system prompt, conversation-start context, the loop (cap, final `tool_choice: none`, `cached_tail` on every call, thinking + effort, trace with `iteration_count`). `cli chat` in-process, in memory, `--today`. SDK pin | Stubbed: cap holds; bad input → `is_error`; breakpoint on the last block every call; thinking blocks replayed unchanged. Live: the § 1 question answered over the pilots |
| 29 | Migration 0009: `users`, `conversations`, `messages`, `feedback`; classified in `test_no_verbatim.py`. Replay from the DB, conversation expiry; `cli chat` resumes, `/new`, `/good`, `/bad` | Turn n+1's request starts with turn n's, byte for byte (tested). Live: a resumed turn within 5 min reads the cache |
| 30 | Migration 0010: `profile_facts`. `agent/profile.py`: `remember`, snapshot, supersede, forget | A fact remembered in one conversation is in the next one's context; forget leaves no rows |
| 31 | `eval/curation.py` + `eval/gold/curation.yaml` + `cli eval-curation` (`--effort`, `--model`, `--arm`) | Baseline in § 8; two runs agree within noise; truncation share and cost per exchange reported |
| 32 | Terraform: `InvokeModelWithResponseStream` on the chat profile; IAM + `pricing.yaml` for two embedding candidates and one cheaper chat candidate; the Smoke workflow calls each | Gated CI apply green (the first one since the 09-23 laptop apply: read the plan); Smoke green through OIDC |
| 33 | `LlmClient.stream_message` (recorded stream, row on a broken stream) + a recorded-stream stub; the loop yields events | Stubbed: one row per stream, a broken one included; cost equals the unstreamed path's |
| 34 | `api/`: `POST /chat` (SSE), `POST /feedback`, `DELETE /users/{id}/profile`; bearer token; Compose `api`; `cli chat --api URL` | The same conversation in-process and over HTTP; boundary test covers `api/` |
| 35 | *(gated)* `LlmClient.embed` (purpose `embed`), migration 0011: `embedding` + markers, `cli embed` (skip unchanged), model chosen by a ~20-query retrieval check, `query` on `search_events` | A/B reported by `eval-curation`; decision recorded |
| 36 | Sign-off (docs): chat cost, cache share, latency, iteration distribution into § 6 and § 8; decide loop vs workflow, chat model and effort | P3 done-when met |

PR 32 depends on nothing and can go first; 33–34 do not block the eval, which runs in-process. PR 31 is the checkpoint: everything after it is plumbing or optional. Writing the scenarios is human work, like P2's labels.

**Cost:** an eval run ≈ 20 scenarios × ~3 exchanges × ~$0.10 ≈ $6, plus ~$1 of judge; an effort sweep or A/B multiplies it. Embeddings are noise (§ 6).

---

## 9. Open risks

- **Extraction accuracy** across heterogeneous sites — measured in P2, not discovered in P5.
- **Unusable venue sites** — surveyed by hand 2026-10-03 (curl, our UA): **5 of 18** crawlable pilots stay facts-only. `blocked` (Cloudflare challenge): Fundació Joan Miró, CaixaForum. `thin` (JS-rendered): Dilalica, Sala Parés, Espronceda. Above the ~4 threshold; decided to accept the gaps rather than add a headless fetch. Robots allows our UA on all 18. P2 accuracy is measured over the other 13.
- **The gold set decays** — page text cannot outlive the 7-day TTL, so labels are pinned to a `content_hash` and go stale when a venue updates its page. Expect to re-label a few pages per month while extraction is being tuned.
- **Newest-model access on Bedrock** — AWS (2026-09-23): eligibility depends on account usage history, is reassessed as usage grows, and isn't configurable. Opus 5 refused everywhere; Mantle refused for every model, including Haiku 4.5 (which works on the runtime endpoint) and Sonnet 5 (agreement applied). **Worked around, not blocking:** chat on Opus 4.6, extraction on Haiku 4.5, both via runtime inference profiles. Revisit at P3; Claude Platform on AWS if quality demands a newer model.
- **No structured outputs on Bedrock's Messages endpoint** — tool inputs validated with pydantic, `is_error` + retry on failure.
- ~~**Langfuse SDK surface unverified**~~ — sidestepped 2026-09-22: no Langfuse SDK; plain OTel exports to its OTLP endpoint (`/api/public/otel/v1/traces`, verified against current docs). Not yet exercised against the running Compose instance.
- **p95 latency** with three round trips — fallbacks: lower `effort`, then merge search + hydrate.
- **Coverage bounds the checkpoint** — 5 of 18 sites are facts-only, and the current shows sit at 12 venues. A scenario asking for an area with no pilot venue measures GRAF, not the curator; scenarios are placed where the data is, and that limit is stated with the P3 result.
- **Venue text reaches chat through our summaries** — `summary_en` is written by Haiku from venue pages, so a page could plant instructions in it. `remember` is the only tool that writes. The system prompt treats tool results as data, and profile facts are listed in the eval transcript for review.
- **Anthropic SDK 2.x** — sends an API version Bedrock refuses for Opus 4.6 and Haiku 4.5 (Messages path). Pinned below 2 in PR 28; lifted only with a move to a 4.7+ model.
- **GRAF's live event window** (56–104 events) — long-running shows depend on crawling.
- **Event history accrues only where `sync-graf` runs against a kept database** — locally, today; the `GRAF` workflow's DB is thrown away. Events that leave the window between runs are lost until P6 schedules it.
- **GRAF coverage** — 414 of 570 terms join no profile, so have no URL; 23 of 146 URL-bearing profiles join no term (mostly festivals/associations, but also `Galeria N2` ↔ `N2 Galeria`, `Fundació Vila Casas` ↔ `Museu Can Framis`). No pilot affected; an alias table if P2 needs them. Only 1 joined venue is Instagram-only.
- **The join is partly heuristic and recomputed each run** — 28 containment joins, and a GRAF rename can silently move a URL between venues. The `sync-graf` report lists both for review.
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
