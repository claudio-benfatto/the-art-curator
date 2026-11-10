"""A small invented corpus with one of everything `queries/events.py` reads, for
`tests/test_queries.py` and the MCP half of `tests/test_no_verbatim.py`.

One place, a barri near the second space, is there for `search_events(near=...)`.

The GRAF events cover each rule `search_events` applies to them: a twin of an extracted show at
the same venue, one at another space of the same organisation under its English title, a
namesake whose dates do not overlap, a superseded occurrence and an event that left the window.

The listing's text and its link text carry markers that stand for third-party prose. No query may
return either (CLAUDE.md § 1).
"""

import asyncio
from typing import Any
from unittest.mock import patch

from mcp import Client
from mcp.types import CallToolResult, Tool
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from art_curator import mcp_server

HOST = "https://galeria-exemple.test"
LISTING_URL = f"{HOST}/ca/exposicions/"
DETAIL_URL = f"{HOST}/ca/exposicions/prova-u"
CHANGED_URL = f"{HOST}/ca/agenda"
THIN_URL = f"{HOST}/ca/"

PAGE_MARKER = "frase-de-la-pagina-7781"
LINK_MARKER = "text-de-l-enllac-7781"
LISTING_TEXT = f"Exposicions {PAGE_MARKER} Prova U del 12 de setembre al 30 de novembre"
LINKS = f'[{{"number": 1, "url": "{DETAIL_URL}", "text": "{LINK_MARKER}"}}]'
HASH = "a" * 64

# Extraction markers that match the page's own hash: extracted, and nothing has moved since.
_MARKS = "'ok', '{h}', 3, 'haiku', {items}, '2026-10-30 10:00+00'"

CORPUS = [
    """
    INSERT INTO venues (id, source_venue_id, source_profile_id, name, slug, address, city, geom,
                        website_url, crawl_enabled, is_pilot)
    OVERRIDING SYSTEM VALUE VALUES
      (1, 101, 501, 'Galeria Exemple', 'galeria-exemple', 'Carrer Inventat 1', 'Barcelona',
       ST_GeogFromText('POINT(2.1700 41.3800)'), 'https://galeria-exemple.test/', true, true),
      (2, 102, NULL, 'Sala 100% Lluny', 'sala-lluny', NULL, 'Girona', NULL, NULL, false, false),
      -- A second space of the same organisation, ~3.35 km from the first.
      (3, 103, 501, 'Annex Exemple', 'annex-exemple', NULL, 'Barcelona',
       ST_GeogFromText('POINT(2.2000 41.4000)'), 'https://galeria-exemple.test/', false, false)
    """,
    # ~0.2 km from Annex Exemple and ~3.5 km from Galeria Exemple.
    """
    INSERT INTO places (kind, code, name, aliases, within, geom)
    VALUES ('barri', '68', 'el Poblenou', '{Poblenou}', 'Sant Martí',
            ST_GeogFromText('POINT(2.2024 41.4000)'))
    """,
    f"""
    INSERT INTO venue_seeds (venue_id, url, page_type, confidence, dated_items, language, status,
                             model)
    VALUES (1, '{LISTING_URL}', 'current_listing', 'high', 4, 'ca', 'accepted', 'haiku')
    """,
    # A listing extracted from its current text; a detail page past the purge; a page whose text
    # changed after it was extracted; a page that was never usable.
    f"""
    INSERT INTO venue_pages (id, venue_id, url, kind, status, http_status, content_hash, raw_text,
                             links, fetched_at, extract_status, extracted_hash, extract_version,
                             extract_model, extracted_items, extracted_at)
    OVERRIDING SYSTEM VALUE VALUES
      (1, 1, '{LISTING_URL}', 'listing', 'ok', 200, '{HASH}', '{LISTING_TEXT}', '{LINKS}',
       '2026-10-30 09:00+00', {_MARKS.format(h=HASH, items=2)}),
      (2, 1, '{DETAIL_URL}', 'detail', 'ok', 200, '{"b" * 64}', NULL, NULL,
       '2026-10-01 09:00+00', {_MARKS.format(h="b" * 64, items=1)}),
      (3, 1, '{CHANGED_URL}', 'listing', 'ok', 200, '{"c" * 64}', 'agenda nova', NULL,
       '2026-10-30 09:00+00', {_MARKS.format(h="d" * 64, items=0)}),
      (4, 1, '{THIN_URL}', 'listing', 'thin', 200, NULL, NULL, NULL, '2026-10-30 09:00+00',
       NULL, NULL, NULL, NULL, NULL, NULL)
    """,
    # On 2026-10-31: Prova U is on, Arxiu is over, Col·lecció has no end date. The stand is the
    # gallery's at a fair, not at the gallery.
    f"""
    INSERT INTO exhibitions (id, venue_id, title, title_key, artists, start_date, end_date,
                             at_venue, detail_url, summary_en, confidence, source_url,
                             content_hash, model, extract_version)
    OVERRIDING SYSTEM VALUE VALUES
      (1, 1, 'Prova U', 'prova u', '{{Artista Inventada}}', '2026-09-12', '2026-11-30', true,
       '{DETAIL_URL}', 'Drawings on paper, made for this show.', 'high', '{DETAIL_URL}',
       '{"b" * 64}', 'haiku', 3),
      (2, 1, 'Arxiu', 'arxiu', '{{}}', '2026-06-01', '2026-09-01', true, NULL, NULL, 'high',
       '{LISTING_URL}', '{HASH}', 'haiku', 3),
      (3, 1, 'Col·lecció', 'colleccio', '{{}}', '2026-01-01', NULL, true, NULL, NULL, 'medium',
       '{LISTING_URL}', '{HASH}', 'haiku', 3),
      (4, 1, 'Stand a la fira', 'stand a la fira', '{{}}', '2026-11-06', '2026-11-08', false,
       NULL, NULL, 'high', '{CHANGED_URL}', '{"c" * 64}', 'haiku', 3)
    """,
    f"""
    INSERT INTO art_events (id, venue_id, title, title_key, start_date, start_time, at_venue,
                            confidence, source_url, content_hash, model, extract_version, kind,
                            exhibition_id)
    OVERRIDING SYSTEM VALUE VALUES
      (1, 1, 'Xerrada', 'xerrada', '2026-09-10', NULL, true, 'high', '{LISTING_URL}',
       '{HASH}', 'haiku', 3, 'talk', NULL),
      (2, 1, 'Visita', 'visita', '2026-11-05', '19:00', true, 'high', '{LISTING_URL}',
       '{HASH}', 'haiku', 3, 'guided_tour', 1),
      (3, 1, 'Taller', 'taller', NULL, NULL, true, 'low', '{LISTING_URL}', '{HASH}', 'haiku',
       3, 'workshop', NULL)
    """,
    # The latest sync ran at 03:00 UTC on 30 October; Retirat and one Nit Oberta occurrence were
    # last seen by the one before. Category 103 is an exhibition, 189 an opening, NULL other.
    """
    INSERT INTO event_snapshots (id, source_event_id, venue_id, title, title_en,
                                 source_category_id, is_free, price_min, source_url, web_url_ca,
                                 last_seen_at)
    OVERRIDING SYSTEM VALUE VALUES
      (1, 9001, 1, 'Prova U', NULL, 103, true, NULL, 'https://graf.test/e/9001', NULL,
       '2026-10-30 03:00+00'),
      (2, 9002, NULL, 'Fira', NULL, NULL, false, 12.50, 'https://graf.test/e/9002', NULL,
       '2026-10-30 03:00+00'),
      (3, 9003, 3, 'Assaig U', 'PROVA Ú.', 103, NULL, NULL, 'https://graf.test/e/9003', NULL,
       '2026-10-30 03:00+00'),
      (4, 9004, 3, 'Nit Oberta', NULL, 189, true, NULL, 'https://graf.test/e/9004',
       'https://annex-exemple.test/nit', '2026-10-30 03:00+00'),
      (5, 9005, 1, 'Retirat', NULL, 103, NULL, NULL, 'https://graf.test/e/9005', NULL,
       '2026-10-20 03:00+00'),
      (6, 9006, 1, 'Arxiu', NULL, 103, false, NULL, 'https://graf.test/e/9006', NULL,
       '2026-10-30 03:00+00')
    """,
    # 23:30 UTC on 31 October is 1 November in Barcelona; 18:00 UTC in November is 19:00 there.
    """
    INSERT INTO event_occurrences (id, event_id, source_occurrence_id, starts_at, ends_at,
                                   last_seen_at)
    OVERRIDING SYSTEM VALUE VALUES
      (1, 1, 1, '2026-09-12 10:00+00', '2026-10-31 23:30+00', '2026-10-30 03:00+00'),
      (2, 2, 2, '2026-10-01 10:00+00', NULL, '2026-10-30 03:00+00'),
      (3, 3, 3, '2026-10-15 09:00+00', '2026-11-30 18:00+00', '2026-10-30 03:00+00'),
      (4, 4, 4, '2026-11-07 18:00+00', '2026-11-07 21:00+00', '2026-10-30 03:00+00'),
      (5, 4, 5, '2026-11-06 18:00+00', '2026-11-06 21:00+00', '2026-10-20 03:00+00'),
      (6, 5, 6, '2026-11-01 10:00+00', '2026-11-30 18:00+00', '2026-10-20 03:00+00'),
      (7, 6, 7, '2026-11-03 10:00+00', '2026-11-20 18:00+00', '2026-10-30 03:00+00')
    """,
    """
    INSERT INTO llm_calls (created_at, provider, model, purpose, input_tokens, output_tokens,
                           cost_usd, latency_ms, error_type)
    VALUES ('2026-10-07 10:00+00', 'converse', 'haiku', 'discover', 5000, 200, 0.006, 900, NULL),
           ('2026-10-30 10:00+00', 'converse', 'haiku', 'extract', 9000, 3000, 0.024, 9000,
            NULL),
           ('2026-10-30 10:05+00', 'converse', 'haiku', 'extract', 100, 0, 0.0001, 300,
            'ClientError')
    """,
]


def call_tools(url: URL, calls: list[tuple[str, dict[str, Any]]]) -> list[CallToolResult]:
    """Call MCP tools over the in-memory transport, against the database at `url`."""

    async def main() -> list[CallToolResult]:
        engine = create_async_engine(url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        try:
            with patch.object(mcp_server, "get_sessionmaker", lambda: sessions):
                async with Client(mcp_server.mcp) as client:
                    return [await client.call_tool(name, args) for name, args in calls]
        finally:
            await engine.dispose()

    return asyncio.run(main())


def list_tools() -> list[Tool]:
    """The server's tools as a client sees them. Needs no database."""

    async def main() -> list[Tool]:
        async with Client(mcp_server.mcp) as client:
            return (await client.list_tools()).tools

    return asyncio.run(main())
