import asyncio, os, time
os.environ.setdefault("BANK_ID", f"brightlane-test2-{int(time.time())}")
import pytest
from fastapi.testclient import TestClient
import llm, main, memory

async def fake_stream(messages, temperature=0.4):
    for piece in ["Write ", "about ", "security."]:
        yield piece

@pytest.fixture
def client(monkeypatch):
    async def ok(*a, **k): return True
    async def none(*a, **k): return []
    for name, fn in [("ensure_bank", ok), ("retain", ok), ("recall_many", none)]:
        monkeypatch.setattr(memory, name, fn)
    monkeypatch.setattr(llm, "_complete_stream", fake_stream)
    with TestClient(main.app, raise_server_exceptions=False) as c:
        yield c

def test_chat_stream_endpoint(client):
    r = client.post("/chat/stream", json={"message": "hi", "memory": False})
    assert r.status_code == 200
    assert r.text == "Write about security."
    assert r.headers["X-Used-Memories"] == "[]"

def test_plan_returns_items_and_used(client, monkeypatch):
    good = __import__("json").dumps([{"d": d, "f": "How-to", "c": "Blog", "t": f"T{d}", "w": "w"} for d in llm.DAYS])
    async def fake(messages, temperature=0.4): return good
    monkeypatch.setattr(llm, "_complete", fake)
    r = client.post("/plan", json={})
    body = r.json()
    assert set(body) == {"items", "used"}
    assert len(body["items"]) == 5

def test_plan_instruction_reaches_prompt(client, monkeypatch):
    good = __import__("json").dumps([{"d": d, "f": "How-to", "c": "Blog", "t": f"T{d}", "w": "w"} for d in llm.DAYS])
    seen = []
    async def fake(messages, temperature=0.4):
        seen.append(messages); return good
    monkeypatch.setattr(llm, "_complete", fake)
    client.post("/plan", json={"instruction": "swap Wednesday for a case study"})
    assert "swap Wednesday for a case study" in seen[0][1]["content"]