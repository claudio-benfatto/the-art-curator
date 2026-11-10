"""Load `places` from open data: Barcelona's barris and districts, and Catalonia's municipalities.

`search_events(near=...)` measures from a place's point, so "near Poblenou" is a lookup in SQL and
never a guess at coordinates (CLAUDE.md § 8, PLAN.md § 8 Location).

Sources, and the attribution both licences ask for:

- **Barris and districts:** Ajuntament de Barcelona, *Divisions administratives de la ciutat de
  Barcelona* (Open Data BCN, dataset `20170706-districtes-barris`), CC BY 4.0. Each area comes as a
  WGS84 polygon; the point stored is its centroid, worked out by PostGIS when the row is written.
  A district's centroid can sit far from its best-known streets (Sants-Montjuïc's is on the
  mountain), which is why barris are loaded too.
- **Municipalities:** *Municipis Catalunya Geo*, Institut d'Estadística de Catalunya (IDESCAT) and
  Institut Cartogràfic i Geològic de Catalunya (ICGC), on the Generalitat de Catalunya's open data
  portal under the *Llicència oberta d'ús d'informació – Catalunya*, which asks for the same
  attribution as CC BY. The point is the municipal seat, not the centre of the territory: Vic's is
  in the town, not in its fields.

Names, codes and points are facts, so nothing here is prose (CLAUDE.md § 1).

A place also answers to its **aliases**: its name without the article ("Poblenou" for "el
Poblenou"), and the few names in `EXTRA_ALIASES` that people use and no dataset carries. IDESCAT
files 131 municipalities under the noun ("Garriga, la"); the article is put back in front first.
Matching ignores case and accents; that happens in SQL, in `queries/events.resolve_place`.

The table is replaced whole on every load, in one transaction. A download that is short, broken or
points outside Catalonia fails before anything is written.
"""

import csv
import io
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from geoalchemy2 import Geography
from sqlalchemy import Text, bindparam, cast, delete, func, insert
from sqlalchemy.ext.asyncio import AsyncSession

from art_curator.db.models import PLACE_KINDS, Place
from art_curator.ingest.http import FetchError, PoliteClient

_BCN = "https://opendata-ajuntament.barcelona.cat/data/dataset/808daafa-d9ce-48c0-925a-fa5afdb1ed41"
BARRIS_URL = f"{_BCN}/resource/b21fa550-56ea-4f4c-9adc-b8009381896e/download"
DISTRICTS_URL = f"{_BCN}/resource/576bc645-9481-4bc4-b8bf-f5972c20df3f/download"
MUNICIPALITIES_URL = (
    "https://analisi.transparenciacatalunya.cat/resource/9aju-tpwc.csv"
    "?$select=codi,nom,nom_comarca,latitud,longitud&$order=codi&$limit=5000"
)

# Barcelona has had 73 barris in 10 districts since 2009, and Catalonia 947 municipalities since
# 2018. Fewer means a truncated or changed download, not a redrawn map.
MIN_PLACES = {"barri": 73, "district": 10, "municipality": 940}

# A generous box around Catalonia. A point outside it is a swapped axis or a projected coordinate.
_LONGITUDES = (0.0, 3.5)
_LATITUDES = (40.4, 43.0)

# Names people use for a barri that no dataset carries. Ours, and facts: what a name refers to.
EXTRA_ALIASES = {
    "Sant Pere, Santa Caterina i la Ribera": ("el Born",),
    "el Barri Gòtic": ("Gòtic",),
    "el Poble-sec": ("Poblesec",),
    "la Vila Olímpica del Poblenou": ("la Vila Olímpica",),
}

# Catalan articles, and Aranese ones for the Val d'Aran ("Es Bòrdes").
_ARTICLE = re.compile(r"^(?:el|la|els|les|es|eth|era)\s+|^l['’]", re.IGNORECASE)
_ARTICLE_AFTER = re.compile(r"^(?P<noun>.+), (?P<article>el|la|els|les|es|eth|era|l['’])$", re.I)


class PlacesError(Exception):
    """A download that cannot be loaded as it is."""


@dataclass(frozen=True)
class PlaceRecord:
    kind: str  # one of PLACE_KINDS
    code: str
    name: str
    within: str | None
    aliases: tuple[str, ...]
    wkt: str  # WGS84, longitude first: an area's polygon, or a municipal seat's point


def aliases(name: str, extra: Sequence[str] = ()) -> tuple[str, ...]:
    """Other names `name` answers to: each of `extra`, and each of them and `name` without its
    article. Never `name` itself."""
    found: list[str] = []
    for each in (name, *extra):
        for alias in (each, _ARTICLE.sub("", each)):
            if alias != name and alias not in found:
                found.append(alias)
    return tuple(found)


def article_first(name: str) -> str:
    """IDESCAT files a municipality under its noun, "Hospitalet de Llobregat, l'"; the name is
    "l'Hospitalet de Llobregat"."""
    match = _ARTICLE_AFTER.match(name)
    if match is None:
        return name
    noun, article = match["noun"], match["article"]
    return f"{article}{noun}" if article.endswith(("'", "’")) else f"{article} {noun}"


def _rows(text: str, columns: set[str]) -> list[dict[str, str]]:
    """CSV rows keyed by lower-case header. The district file capitalises one header."""
    reader = csv.DictReader(io.StringIO(text.removeprefix("﻿")))
    reader.fieldnames = [name.strip().lower() for name in reader.fieldnames or []]
    if missing := columns - set(reader.fieldnames):
        raise PlacesError(f"columns {sorted(missing)} missing; found {reader.fieldnames}")
    return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _in_catalonia(longitude: float, latitude: float) -> bool:
    return (
        _LONGITUDES[0] <= longitude <= _LONGITUDES[1] and _LATITUDES[0] <= latitude <= _LATITUDES[1]
    )


def _area_wkt(wkt: str, name: str) -> str:
    """A polygon as served, checked on its first vertex: the file also carries projected (ETRS89)
    coordinates, and taking that column by mistake puts every barri in the Atlantic."""
    first = re.search(r"\(\(+\s*(-?[\d.]+)\s+(-?[\d.]+)", wkt)
    if not wkt.startswith(("POLYGON", "MULTIPOLYGON")) or first is None:
        raise PlacesError(f"{name}: not a WKT polygon")
    if not _in_catalonia(float(first.group(1)), float(first.group(2))):
        raise PlacesError(f"{name}: polygon starts outside Catalonia, at {first.groups()}")
    return wkt


def parse_barris(text: str) -> list[PlaceRecord]:
    rows = _rows(text, {"codi_barri", "nom_barri", "nom_districte", "geometria_wgs84"})
    return [
        PlaceRecord(
            kind="barri",
            code=row["codi_barri"],
            name=row["nom_barri"],
            within=row["nom_districte"],
            aliases=aliases(row["nom_barri"], EXTRA_ALIASES.get(row["nom_barri"], ())),
            wkt=_area_wkt(row["geometria_wgs84"], row["nom_barri"]),
        )
        for row in rows
    ]


def parse_districts(text: str) -> list[PlaceRecord]:
    rows = _rows(text, {"codi_districte", "nom_districte", "geometria_wgs84"})
    return [
        PlaceRecord(
            kind="district",
            code=row["codi_districte"],
            name=row["nom_districte"],
            within="Barcelona",
            aliases=aliases(row["nom_districte"]),
            wkt=_area_wkt(row["geometria_wgs84"], row["nom_districte"]),
        )
        for row in rows
    ]


def parse_municipalities(text: str) -> list[PlaceRecord]:
    records = []
    for row in _rows(text, {"codi", "nom", "nom_comarca", "latitud", "longitud"}):
        try:
            longitude, latitude = float(row["longitud"]), float(row["latitud"])
        except ValueError as exc:
            raise PlacesError(f"{row['nom']}: no usable point ({exc})") from exc
        if (longitude, latitude) == (0, 0):
            continue  # "999998 No consta": a placeholder for "not recorded", not a municipality
        if not _in_catalonia(longitude, latitude):
            raise PlacesError(f"{row['nom']}: seat outside Catalonia, at {latitude}, {longitude}")
        name = article_first(row["nom"])
        records.append(
            PlaceRecord(
                kind="municipality",
                code=row["codi"],
                name=name,
                within=row["nom_comarca"] or None,
                aliases=aliases(name),
                wkt=f"POINT({longitude} {latitude})",
            )
        )
    return records


@dataclass(frozen=True)
class Source:
    kind: str
    url: str
    parse: Callable[[str], list[PlaceRecord]]
    attribution: str


SOURCES = (
    Source(
        "barri",
        BARRIS_URL,
        parse_barris,
        "Barris: Ajuntament de Barcelona, Open Data BCN, CC BY 4.0",
    ),
    Source(
        "district",
        DISTRICTS_URL,
        parse_districts,
        "Districts: Ajuntament de Barcelona, Open Data BCN, CC BY 4.0",
    ),
    Source(
        "municipality",
        MUNICIPALITIES_URL,
        parse_municipalities,
        "Municipalities: IDESCAT and ICGC, Generalitat de Catalunya, "
        "Llicència oberta d'ús d'informació – Catalunya",
    ),
)


async def fetch(client: PoliteClient) -> list[PlaceRecord]:
    """Every source, parsed. Raises FetchError or PlacesError; nothing is written either way."""
    records: list[PlaceRecord] = []
    for source in SOURCES:
        page = await client.get_text(source.url, accept="text/csv", media_types=None)
        if page.status != 200:
            raise FetchError(f"{source.url}: HTTP {page.status}")
        if page.truncated:
            raise PlacesError(f"{source.url}: download cut off at the size cap")
        try:
            records += source.parse(page.text)
        except PlacesError as exc:
            raise PlacesError(f"{source.kind} ({source.url}): {exc}") from exc
    return records


def problems(records: Sequence[PlaceRecord]) -> list[str]:
    """Why `records` must not replace the table: too few of a kind, a code twice, or an
    `EXTRA_ALIASES` name the download no longer has. Empty when they can."""
    found = []
    counts = Counter(record.kind for record in records)
    for kind in PLACE_KINDS:
        if counts[kind] < MIN_PLACES[kind]:
            found.append(f"{counts[kind]} {kind} rows, expected at least {MIN_PLACES[kind]}")
    repeated = Counter((record.kind, record.code) for record in records)
    found += [
        f"{kind} code {code} appears {n} times" for (kind, code), n in repeated.items() if n > 1
    ]
    barris = {record.name for record in records if record.kind == "barri"}
    found += [
        f"no barri named {name!r} for EXTRA_ALIASES" for name in EXTRA_ALIASES.keys() - barris
    ]
    return found


async def write_places(session: AsyncSession, records: Sequence[PlaceRecord]) -> int:
    """Replace every row with `records`. Never commits. Each point is computed by PostGIS: an
    area's centroid, or a seat's point as it is."""
    await session.execute(delete(Place))
    if not records:
        return 0
    point = func.ST_Centroid(func.ST_GeomFromText(bindparam("wkt", type_=Text), 4326))
    statement = insert(Place).values(
        kind=bindparam("kind"),
        code=bindparam("code"),
        name=bindparam("name"),
        within=bindparam("within"),
        aliases=bindparam("aliases"),
        geom=cast(point, Geography("POINT", srid=4326)),
    )
    await session.execute(
        statement,
        [
            {
                "kind": r.kind,
                "code": r.code,
                "name": r.name,
                "within": r.within,
                "aliases": list(r.aliases),
                "wkt": r.wkt,
            }
            for r in records
        ],
    )
    return len(records)


def report(records: Sequence[PlaceRecord]) -> list[str]:
    counts = Counter(record.kind for record in records)
    lines = [f"{kind:<13} {counts[kind]}" for kind in PLACE_KINDS]
    return lines + [f"{'source':<13} {source.attribution}" for source in SOURCES]
