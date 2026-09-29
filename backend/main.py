"""
main.py: the web server. It only defines endpoints; the real work is in
memory.py (Hindsight) and llm.py (Groq).

Run with:  uvicorn main:app --reload
Then open: http://127.0.0.1:8000
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read the .env file BEFORE importing our own modules

from fastapi import BackgroundTasks, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import llm  # noqa: E402
import memory  # noqa: E402

logging.basicConfig(level=logging.INFO)

from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory.ensure_bank()  # runs once when the server starts

    # Load the company's history into memory automatically, so the agent
    # already remembers everything the moment the server is up — nobody
    # has to click a "Load history" button first for Memory ON to work.
    # We only do this if the bank looks empty, so restarting the server
    # doesn't re-retain the same facts over and over.
    existing = await memory.list_memories()
    if not existing:
        logging.info("Memory bank is empty — loading seed history now...")
        result = await memory.load_history_into_memory()
        logging.info("Seed history loaded: %s", result)
    else:
        logging.info("Memory already has %d items — skipping seed load.", len(existing))

    yield


app = FastAPI(title="Brightlane Content Strategist", lifespan=lifespan)
STATIC = Path(__file__).resolve().parent.parent / "static"

html_file_path = STATIC / "UI.html"


# ------------------------------------------------ request shapes (JSON in)

class ChatIn(BaseModel):
    message: str
    memory: bool = True  # the UI's Memory ON/OFF switch


class PlanIn(BaseModel):
    memory: bool = True
    avoid: list[str] = []  # ideas rejected in this session (shown right away)
    instruction: str = ""  # a follow-up like "swap Wednesday for a case study"


class FeedbackIn(BaseModel):
    title: str
    action: str  # "approved" or "rejected"
    reason: str = ""
    format: str = ""   # only used when action == "approved"
    channel: str = ""  # only used when action == "approved"


class PerformanceIn(BaseModel):
    title: str
    views: int
    conversions: int


# ---------------------------------------------------------------- endpoints

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
async def chat(body: ChatIn, tasks: BackgroundTasks):
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "message is empty")

    used: list[str] = []
    if body.memory:
        # Ask memory several different questions, then merge the answers.
        used = await memory.recall_many([
            message,
            "brand voice and target audience",
            "which content formats and topics perform best",
            "ideas the marketer approved or rejected, and feedback they gave",
        ])

    try:
        text = await llm.chat_answer(message, used if body.memory else None)
    except Exception as e:
        raise HTTPException(502, f"LLM error: {e}")

    # Remember what the marketer said (statements like "that was too salesy"),
    # but not plain questions, or memory fills up with "What should we write?".
    if body.memory and not message.endswith("?"):
        tasks.add_task(
            memory.retain,
            f'The marketer told the strategist: "{message}"',
            "feedback from the marketer",
        )

    return {"text": text, "used": used[:8]}


@app.post("/chat/stream")
async def chat_stream(body: ChatIn, tasks: BackgroundTasks):
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "message is empty")

    used: list[str] = []
    if body.memory:
        used = await memory.recall_many([
            message,
            "brand voice and target audience",
            "which content formats and topics perform best",
            "ideas the marketer approved or rejected, and feedback they gave",
        ])

    if body.memory and not message.endswith("?"):
        tasks.add_task(
            memory.retain,
            f'The marketer told the strategist: "{message}"',
            "feedback from the marketer",
        )

    async def gen():
        try:
            async for piece in llm.chat_answer_stream(message, used if body.memory else None):
                yield piece
        except Exception as e:
            yield f"\n[error: {e}]"

    return StreamingResponse(
        gen(), media_type="text/plain",
        headers={"X-Used-Memories": json.dumps(used[:8])},
        background=tasks,
    )


@app.get("/memories")
async def memories():
    return await memory.list_memories()


@app.post("/load")
async def load_history():
    """Store the brand voice notes and all past posts in Hindsight."""
    return await memory.load_history_into_memory()


@app.get("/content")
def content():
    """Rows for the Content log table."""
    return memory.content_rows()


@app.post("/plan")
async def plan(body: PlanIn):
    used = None
    if body.memory:
        used = await memory.recall_many([
            "which content formats and channels perform best",
            "which topics were covered recently and which are missing",
            "brand voice and target audience",
            "ideas the marketer approved or rejected, and why",
        ], per_query=8)
    try:
        items = await llm.weekly_plan(used, body.avoid, body.instruction)
    except Exception as e:
        raise HTTPException(502, f"LLM error: {e}")
    return {"items": items, "used": used or []}


@app.post("/feedback")
async def feedback(body: FeedbackIn):
    verb = "approved" if body.action == "approved" else "rejected"
    sentence = f"The marketer {verb} the content idea: '{body.title}'."
    if body.reason:
        sentence += f" Reason: {body.reason}."
    if verb == "rejected":
        sentence += " Do not suggest similar ideas again."
    ok = await memory.retain(sentence, "feedback on a suggested idea")

    if verb == "approved":
        # An approved idea is now real, published content: it belongs in
        # the Content Log immediately (as "Pending", since it has no
        # results yet), not just a one-line note buried in memory.
        row = memory.add_published_idea(body.title, body.format, body.channel)
        await memory.retain(memory.pending_sentence(row), "content performance")

    return {"ok": ok}


@app.post("/performance")
async def performance(body: PerformanceIn):
    """Record real results for something that was approved earlier. This
    is what lets the agent's own past suggestions become evidence for
    future ones."""
    ok = await memory.record_performance(body.title, body.views, body.conversions)
    if not ok:
        raise HTTPException(404, "no pending (unmeasured) idea with that title")
    return {"ok": True}


# The web page itself. Serving it from here means the UI's fetch('/chat')
# calls the same server, so there are no CORS problems.
@app.get("/")
def index():
    return FileResponse(STATIC / "UI.html")
@app.get("/learning-curve")
def learning_curve():
    f = Path(__file__).parent / "learning_curve.json"
    if not f.exists():
        raise HTTPException(404, "run eval_learning_curve.py first")
    return json.loads(f.read_text())