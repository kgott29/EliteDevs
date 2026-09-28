"""
main.py: the web server. It only defines endpoints; the real work is in
memory.py (Hindsight) and llm.py (Groq).

Run with:  uvicorn main:app --reload
Then open: http://localhost:8000
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read the .env file BEFORE importing our own modules

from fastapi import BackgroundTasks, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import llm  # noqa: E402
import memory  # noqa: E402

logging.basicConfig(level=logging.INFO)

from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory.ensure_bank()  # runs once when the server starts
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


class FeedbackIn(BaseModel):
    title: str
    action: str  # "approved" or "rejected"
    reason: str = ""


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
        return await llm.weekly_plan(used)
    except Exception as e:
        raise HTTPException(502, f"LLM error: {e}")


@app.post("/feedback")
async def feedback(body: FeedbackIn):
    verb = "approved" if body.action == "approved" else "rejected"
    sentence = f"The marketer {verb} the content idea: '{body.title}'."
    if body.reason:
        sentence += f" Reason: {body.reason}."
    if verb == "rejected":
        sentence += " Do not suggest similar ideas again."
    ok = await memory.retain(sentence, "feedback on a suggested idea")
    return {"ok": ok}


# The web page itself. Serving it from here means the UI's fetch('/chat')
# calls the same server, so there are no CORS problems.
@app.get("/")
def index():
    return FileResponse(STATIC / "UI.html")