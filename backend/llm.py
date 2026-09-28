"""
llm.py: every call to the language model (Groq) lives in this file.

Two jobs:
  chat_answer()  - answer a question, optionally using recalled memories
  weekly_plan()  - produce a 5-day content plan as structured data
"""

import json
import logging
import os
import re
from datetime import date

from groq import AsyncGroq

log = logging.getLogger("llm")

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM = (
    "You are the content strategist for Brightlane, a workflow automation "
    "company for finance teams. You help the marketing team decide what to "
    "publish. Be specific and concise. Never invent numbers or past results: "
    "only use facts you were given."
)

_client = None


def client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


async def _complete(messages: list[dict], temperature: float = 0.4) -> str:
    resp = await client().chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature
    )
    return (resp.choices[0].message.content or "").strip()


def _memory_block(memories: list[str] | None) -> str:
    """Turn the recalled memories into text for the prompt."""
    if memories is None:
        return "You have no stored knowledge about this company's past content."
    if not memories:
        return "Nothing relevant was found in memory yet."
    return "What you remember about this company:\n" + "\n".join(f"- {m}" for m in memories)


# ----------------------------------------------------------------- chat

async def chat_answer(message: str, memories: list[str] | None) -> str:
    """memories=None means 'memory is OFF'; a list means 'memory is ON'."""
    prompt = (
        f"Today's date is {date.today():%B %d, %Y}.\n\n"
        f"{_memory_block(memories)}\n\n"
        f"The marketer says: {message}\n\n"
        "Reply in 2-4 sentences. If the memories contain evidence (results, "
        "gaps, feedback), cite it briefly and follow the brand voice and any "
        "feedback. If you have no memories, give sensible general advice and "
        "do not pretend to know the company's history."
    )
    return await _complete(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    )


# ----------------------------------------------------------------- plan

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
KEYS = ["d", "f", "c", "t", "w"]  # day, format, channel, title, why


def _parse_plan(raw: str) -> list[dict]:
    """Find the JSON list inside the model's reply and check its shape."""
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        raise ValueError("no JSON list in reply")
    items = json.loads(match.group(0))
    if not isinstance(items, list) or len(items) < 5:
        raise ValueError("expected 5 items")
    plan = []
    for i, item in enumerate(items[:5]):
        row = {k: str(item.get(k, "")).strip() for k in KEYS}
        row["d"] = row["d"] or DAYS[i]
        if not row["t"]:
            raise ValueError("item without a title")
        plan.append(row)
    return plan


async def weekly_plan(memories: list[str] | None) -> list[dict]:
    prompt = (
        f"Today's date is {date.today():%B %d, %Y}.\n\n"
        f"{_memory_block(memories)}\n\n"
        "Plan next week's content: exactly 5 items, Monday to Friday.\n"
        "Reply with ONLY a JSON list, no other text. Each item must be an "
        'object with these keys: "d" (day: Mon, Tue, Wed, Thu or Fri), '
        '"f" (format, e.g. How-to, Case study, Video, Newsletter), '
        '"c" (channel, e.g. Blog, LinkedIn, Email), '
        '"t" (a specific title), '
        '"w" (one sentence: why this, citing the evidence you were given).\n'
        "If you have no memories, make sensible generic suggestions and say "
        'in "w" that there is no history to draw on. Do not invent statistics.'
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]

    last_error = None
    for attempt in range(3):  # LLMs sometimes return broken JSON, so retry
        try:
            return _parse_plan(await _complete(messages, temperature=0.3))
        except Exception as e:
            last_error = e
            log.warning("plan attempt %d failed: %s", attempt + 1, e)
    raise RuntimeError(f"could not get a valid plan: {last_error}")