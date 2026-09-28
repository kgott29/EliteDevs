"""
memory.py: every call to Hindsight lives in this file.

The rest of the app never talks to Hindsight directly. It calls the
small functions below, so if Hindsight's API changes we fix it in one place.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from hindsight_client import Hindsight

log = logging.getLogger("memory")

BANK_ID = os.getenv("BANK_ID", "brightlane")  # the "memory folder" for this company
SEED_FILE = Path(__file__).parent / "seed_data.json"

_client = None


def client() -> Hindsight:
    """Return the current Hindsight client."""
    global _client

    if _client is None:
        _client = Hindsight(
            base_url=os.getenv("HINDSIGHT_URL", "https://api.hindsight.vectorize.io"),
            api_key=os.getenv("HINDSIGHT_API_KEY"),
            timeout=60.0,
        )

    return _client


async def close_client() -> None:
    """Close the Hindsight client and reset it."""
    global _client

    if _client is not None:
        await _client.aclose()
        _client = None


def _texts(result) -> list[str]:
    """Pull the memory sentences out of a recall() result.

    Written defensively: the result may be an object with a `.results`
    list, or a plain list, and each item may be an object or a dict.
    """
    items = getattr(result, "results", result) or []
    out = []
    for item in items:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text")
        if text:
            out.append(text.strip())
    return out


# ---------------------------------------------------------------- basics

async def ensure_bank() -> None:
    """Create the memory bank with a mission (ignored if it already exists)."""
    try:
        await client().acreate_bank(
            bank_id=BANK_ID,
            name="Brightlane content memory",
            mission=(
                "I am the content strategist for Brightlane, a workflow automation "
                "company for finance teams. I remember every piece of content "
                "published, how it performed, the brand voice, and which ideas "
                "the marketing team approved or rejected."
            ),
        )
        log.info("created bank %s", BANK_ID)
    except Exception as e:
        log.info("bank not created (probably exists already): %s", e)


async def retain(content: str, context: str = "", timestamp: datetime | None = None) -> bool:
    """Store one piece of information. Returns True if it worked."""
    kwargs = {"bank_id": BANK_ID, "content": content}
    if context:
        kwargs["context"] = context
    if timestamp:
        kwargs["timestamp"] = timestamp
    try:
        await client().aretain(**kwargs)
        return True
    except Exception as e:  # never crash the app because memory failed
        log.warning("retain failed: %s", e)
        return False


async def recall(query: str) -> list[str]:
    """Search memory. Returns a list of memory sentences (empty on failure)."""
    try:
        return _texts(await client().arecall(bank_id=BANK_ID, query=query))
    except Exception as e:
        log.warning("recall failed for %r: %s", query, e)
        return []


async def recall_many(queries: list[str], per_query: int = 6) -> list[str]:
    """Run several searches at once, merge the answers, drop duplicates."""
    batches = await asyncio.gather(*(recall(q) for q in queries))
    seen, merged = set(), []
    for batch in batches:
        for text in batch[:per_query]:
            if text not in seen:
                seen.add(text)
                merged.append(text)
    return merged


# ------------------------------------------------- what the UI panel shows

# The UI groups memories into types. We get them by asking one question per type.
PANEL_QUERIES = {
    "voice": "brand voice, tone and target audience",
    "perf": "content performance: views and conversions by format",
    "gap": "topics that have not been covered recently, content gaps",
    "feedback": "ideas the marketer approved or rejected, and feedback they gave",
}


async def list_memories(per_type: int = 5) -> list[dict]:
    """Return [{'t': 'perf', 'x': 'sentence'}, ...] for the memory panel."""
    types = list(PANEL_QUERIES)
    batches = await asyncio.gather(*(recall(PANEL_QUERIES[t]) for t in types))
    seen, out = set(), []
    for t, batch in zip(types, batches):
        for text in batch[:per_type]:
            if text not in seen:
                seen.add(text)
                out.append({"t": t, "x": text})
    return out


# ------------------------------------------------------ seed data (history)

def load_seed() -> dict:
    return json.loads(SEED_FILE.read_text(encoding="utf-8"))


def post_to_sentence(p: dict) -> str:
    """Turn one row of data into a clear sentence. Good input = good memory."""
    date = datetime.strptime(p["published"], "%Y-%m-%d")
    return (
        f"Content published: '{p['title']}' ({p['format']}) on {p['channel']}, "
        f"topic: {p['topic']}. Published on {date:%B %d, %Y}. "
        f"It got {p['views']:,} views and {p['conversions']} conversions."
    )


async def load_history_into_memory() -> dict:
    """Retain the brand voice notes and every past post (run once before the demo)."""
    seed = load_seed()
    sem = asyncio.Semaphore(3)  # at most 3 requests at the same time

    async def one(content, context, timestamp=None):
        async with sem:
            return await retain(content, context, timestamp)

    jobs = [one(note, "brand voice") for note in seed["voice_notes"]]
    for p in seed["posts"]:
        when = datetime.fromisoformat(f"{p['published']}T09:00:00+00:00")
        jobs.append(one(post_to_sentence(p), "content performance", when))

    results = await asyncio.gather(*jobs)
    return {"retained": sum(results), "failed": len(results) - sum(results)}


def content_rows() -> list[list]:
    """The content log in the same row format the UI table uses."""
    rows = []
    for p in load_seed()["posts"]:
        date = datetime.strptime(p["published"], "%Y-%m-%d")
        rows.append([p["title"], p["format"], p["channel"],
                     f"{date:%b} {date.day}", p["views"], p["conversions"]])
    return rows