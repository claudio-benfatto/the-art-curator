"""`graf.fetch` and `graf.record`: the capture, scrubbed at the door, and the fixtures cut from it.

No database and no network: `tests/graf_stub.py` serves the recorded fixtures.
"""

import asyncio
import copy
import json

import pytest

from art_curator.ingest import graf
from art_curator.ingest.graf import Capture, banned_keys
from art_curator.ingest.http import FetchError, open_client
from art_curator.ingest.sync import GrafSnapshot
from tests.graf_stub import BASE_URL, FIXTURES, POLICY, FakeGraf, fixture, no_sleep


def fetch(fake: FakeGraf, *, deep_occurrences: bool = False) -> Capture:
    async def _run() -> Capture:
        async with open_client(POLICY, transport=fake.transport, sleep=no_sleep) as client:
            return await graf.fetch(client, BASE_URL, deep_occurrences=deep_occurrences)

    return asyncio.run(_run())


def test_fetch_pages_through_every_collection():
    fake = FakeGraf()  # pages of 3: venues and events take several
    capture = fetch(fake)

    assert capture.venues == fixture("venues.json")
    assert capture.profiles == fixture("profiles.json")
    assert capture.events == fixture("events.json")
    assert fake.paths().count("event-venues") == 3  # 8 terms
    assert not any("occurrences" in p for p in fake.paths())


def test_fetch_scrubs_prose_before_anything_sees_it():
    venues = copy.deepcopy(fixture("venues.json"))
    venues[0]["description"] = "Un text llarg i amb drets d'autor."
    events = copy.deepcopy(fixture("events.json"))
    events[0]["acf"]["desc_ca"] = "Prosa."
    events[0]["content"] = {"rendered": "<p>Prosa.</p>"}

    capture = fetch(FakeGraf(venues=venues, events=events))

    assert banned_keys([capture.venues, capture.profiles, capture.events]) == set()
    assert capture.venues[0]["name"] == venues[0]["name"]


def test_deep_occurrences_adds_what_the_list_did_not_serve():
    served = {"occurrence_id": 22821, "start": "2026-11-18T19:00:00+01:00"}
    recurring = {"occurrence_id": 99999, "start": "2026-11-25T19:00:00+01:00"}
    fake = FakeGraf(
        occurrences={
            59371: [
                {**o, "event_id": 59371, "end": None, "_links": {"self": []}}
                for o in (served, recurring)
            ]
        }
    )

    capture = fetch(fake, deep_occurrences=True)

    assert sum("occurrences" in p for p in fake.paths()) == 4  # one request per post
    list_row = next(r for r in fixture("events.json") if r["id"] == 59371)
    # Each occurrence is the post's list row with that occurrence swapped in, and nothing else
    # of the occurrence's own payload; where both endpoints serve one, the per-event copy wins.
    assert [r for r in capture.events if r["id"] == 59371] == [
        {**list_row, **o, "end": None} for o in (served, recurring)
    ]
    events = GrafSnapshot.parse(capture).events
    assert sorted(
        e.occurrence.source_occurrence_id for e in events if e.source_event_id == 59371
    ) == [22821, 99999]


def test_deep_occurrences_agree_with_the_list_by_default():
    # The stub derives /occurrences from the list rows, as live GRAF did on 2026-10-02.
    plain = GrafSnapshot.parse(fetch(FakeGraf()))
    deep = GrafSnapshot.parse(fetch(FakeGraf(), deep_occurrences=True))
    assert deep.events == plain.events


def test_deep_occurrences_refuse_a_non_list():
    fake = FakeGraf(occurrences={60390: {"code": "rest_post_invalid_id"}})
    with pytest.raises(FetchError, match="expected a JSON array"):
        fetch(fake, deep_occurrences=True)


# --- record ---------------------------------------------------------------------------------------


def test_record_round_trips(tmp_path):
    capture = fetch(FakeGraf())
    paths = graf.record(capture, tmp_path / "graf")

    assert sorted(p.name for p in paths) == [
        "events.json",
        "identity.json",
        "profiles.json",
        "venues.json",
    ]
    replayed = Capture(
        **{
            name: json.loads((tmp_path / "graf" / f"{name}.json").read_text())
            for name in graf.ENDPOINTS
        }
    )
    assert GrafSnapshot.parse(replayed) == GrafSnapshot.parse(capture)


def test_identity_reproduces_the_committed_fixture_byte_for_byte(tmp_path):
    committed = fixture("identity.json")
    capture = Capture(venues=committed["terms"], profiles=committed["profiles"], events=[])
    graf.record(capture, tmp_path)

    assert (tmp_path / "identity.json").read_text() == (FIXTURES / "identity.json").read_text()


def test_record_refuses_prose_and_writes_nothing(tmp_path):
    # `fetch` scrubs, so this only happens if a capture is built some other way. Still: git is
    # forever.
    venues = [{**fixture("venues.json")[0], "description": "prosa"}]
    with pytest.raises(ValueError, match="description"):
        graf.record(Capture(venues=venues, profiles=[], events=[]), tmp_path / "out")
    assert not (tmp_path / "out").exists()
