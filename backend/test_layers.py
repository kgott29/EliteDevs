"""
test_layers.py: Step 7 tests. Put it next to main.py.

    pip install pytest httpx
    pytest test_layers.py -m "not live" -v     # fast, no API calls
    pytest test_layers.py -m live -v -s        # real Groq + Hindsight (uses fresh test banks)
"""

import json
import os
import time

import pytest

# Use a throwaway bank. This must be set BEFORE main/memory are imported.
os.environ["BANK_ID"] = f"brightlane-test-{int(time.time())}"

from fastapi.testclient import TestClient  # noqa: E402

import llm  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402

GOOD_PLAN = json.dumps([
    {"d": d, "f": "How-to", "c": "Blog", "t": f"Idea for {d}", "w": "because"}
    for d in llm.DAYS
])


def patch_llm(monkeypatch, replies):
    """Make llm._complete return replies in order (an Exception item is raised)."""
    seen = []
    it = iter(replies)

    async def fake(messages, temperature=0.4):
        seen.append(messages)
        r = next(it, replies[-1])
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(llm, "_complete", fake)
    return seen


@pytest.fixture
def client(monkeypatch):
    """App with fake, empty memory so no network is needed."""
    async def ok(*a, **k):
        return True

    async def none(*a, **k):
        return []

    for name, fn in [("ensure_bank", ok), ("retain", ok), ("recall_many", none), ("list_memories", none)]:
        monkeypatch.setattr(memory, name, fn)
    with TestClient(main.app, raise_server_exceptions=False) as c:
        yield c


# ------------------------------------------------ edge cases (offline)

def test_empty_message_is_rejected(client):
    assert client.post("/chat", json={"message": "   "}).status_code == 400


def test_chat_with_empty_memory_works(client, monkeypatch):
    seen = patch_llm(monkeypatch, ["Sure, here is advice."])
    r = client.post("/chat", json={"message": "What should we write next?"})
    assert r.status_code == 200 and r.json()["text"] and r.json()["used"] == []
    assert "Nothing relevant was found" in seen[0][1]["content"]


def test_memory_off_never_touches_memory(client, monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("memory was used while OFF")

    monkeypatch.setattr(memory, "recall_many", boom)
    patch_llm(monkeypatch, ["ok"])
    assert client.post("/chat", json={"message": "hi", "memory": False}).status_code == 200


def test_groq_failure_gives_clean_502(client, monkeypatch):
    patch_llm(monkeypatch, [RuntimeError("groq down")])
    assert client.post("/chat", json={"message": "hi"}).status_code == 502
    assert client.post("/plan", json={}).status_code == 502


def test_plan_retries_bad_json_then_succeeds(client, monkeypatch):
    patch_llm(monkeypatch, ["not json", "still {broken", f"```json\n{GOOD_PLAN}\n```"])
    r = client.post("/plan", json={})
    assert r.status_code == 200 and len(r.json()) == 5


def test_plan_gives_up_after_three_bad_replies(client, monkeypatch):
    seen = patch_llm(monkeypatch, ["garbage"])
    assert client.post("/plan", json={}).status_code == 502
    assert len(seen) == 3


def test_plan_with_too_few_items_is_rejected(client, monkeypatch):
    short = json.dumps(json.loads(GOOD_PLAN)[:3])
    patch_llm(monkeypatch, [short])
    assert client.post("/plan", json={}).status_code == 502


def test_very_long_message_does_not_crash(client, monkeypatch):
    patch_llm(monkeypatch, ["ok"])
    r = client.post("/chat", json={"message": "a" * 60000})
    assert r.status_code in (200, 400, 413, 422)  # anything but a 500 crash
    patch_llm(monkeypatch, [RuntimeError("too long for model")])
    assert client.post("/chat", json={"message": "a" * 60000}).status_code in (400, 413, 422, 502)


def test_memory_service_down_does_not_break_app(monkeypatch):
    def broken():
        raise RuntimeError("hindsight down")

    monkeypatch.setattr(memory, "client", broken)  # real recall/retain, dead client
    patch_llm(monkeypatch, ["ok", GOOD_PLAN])
    with TestClient(main.app, raise_server_exceptions=False) as c:
        assert c.get("/memories").json() == []
        assert c.post("/chat", json={"message": "hi"}).status_code == 200
        assert c.post("/plan", json={}).status_code == 200
        assert c.post("/feedback", json={"title": "x", "action": "rejected"}).json() == {"ok": False}


# ------------------------------------------ feedback loop wiring (offline)

def test_rejected_idea_from_this_session_reaches_the_prompt(client, monkeypatch):
    seen = patch_llm(monkeypatch, [GOOD_PLAN])
    client.post("/plan", json={"avoid": ["Webinar recap: our year in review"]})
    assert "Webinar recap: our year in review" in seen[0][1]["content"]


def test_remembered_rejection_reaches_the_prompt(client, monkeypatch):
    fact = ("The marketer rejected the content idea: 'Webinar recap'. "
            "Do not suggest similar ideas again.")

    async def recalled(queries, per_query=6):
        return [fact]

    monkeypatch.setattr(memory, "recall_many", recalled)
    seen = patch_llm(monkeypatch, [GOOD_PLAN])
    client.post("/plan", json={})
    assert fact in seen[0][1]["content"]


# ------------------------------------------------ live (real services)


def wait_for(c, check, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        mems = c.get("/memories").json()
        if check(mems):
            return mems
        time.sleep(3)
    pytest.fail("memory did not show the expected content in time")


@pytest.fixture(scope="module")
def live():
    with TestClient(main.app) as c:
        r = c.post("/load")
        assert r.status_code == 200 and r.json()["retained"] > 0
        wait_for(c, lambda m: len(m) >= 5)
        yield c


@pytest.mark.live
@pytest.mark.skipif(not (os.getenv("GROQ_API_KEY") and os.getenv("HINDSIGHT_API_KEY")), reason="needs API keys")
def test_live_brand_new_empty_bank(monkeypatch):
    monkeypatch.setattr(memory, "BANK_ID", f"brightlane-empty-{int(time.time())}")
    with TestClient(main.app) as c:
        r = c.post("/chat", json={"message": "What should we write next?"})
        assert r.status_code == 200 and r.json()["text"]
        r = c.post("/plan", json={})
        assert r.status_code == 200 and len(r.json()) == 5


@pytest.mark.live
@pytest.mark.skipif(not (os.getenv("GROQ_API_KEY") and os.getenv("HINDSIGHT_API_KEY")), reason="needs API keys")
def test_live_rejected_idea_does_not_come_back(live):
    r = live.post("/feedback", json={"title": "Webinar recap: our year in review",
                                     "action": "rejected", "reason": "too salesy"})
    assert r.json() == {"ok": True}
    wait_for(live, lambda m: any("Webinar recap" in x["x"] for x in m))
    for _ in range(3):
        plan = live.post("/plan", json={}).json()
        assert not any("webinar" in item["t"].lower() for item in plan), plan


@pytest.mark.live
@pytest.mark.skipif(not (os.getenv("GROQ_API_KEY") and os.getenv("HINDSIGHT_API_KEY")), reason="needs API keys")
def test_live_demo_flow_three_times_in_a_row(live):
    for i in range(3):
        t0 = time.time()
        off = live.post("/chat", json={"message": "What should we write next?", "memory": False}).json()
        on = live.post("/chat", json={"message": "What should we write next?", "memory": True}).json()
        plan = live.post("/plan", json={}).json()
        assert off["text"] and off["used"] == []
        assert on["text"] and on["used"], "memory ON should use memories"
        assert len(plan) == 5 and all(p["t"] for p in plan)
        print(f"demo run {i + 1}: {time.time() - t0:.1f}s")