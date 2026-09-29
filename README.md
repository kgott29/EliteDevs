# Brightlane Content Strategist

An AI content strategy agent for a B2B marketing team, built on [Hindsight](https://hindsight.vectorize.io/), a persistent memory layer for AI agents. It remembers what a company has published, how each piece performed, and what the team approved or rejected, then uses that memory to plan next week's content, answer questions, and revise ideas in conversation.

The core idea: **without memory, the agent gives generic advice any company could get. With memory, it gives advice that's specific to this company's real history.**

## Why this exists

Marketing teams reinvent the wheel every week because nobody remembers what actually worked last month. This agent is a demonstration of an agent that does remember, and that shows its work — every recommendation is traceable back to the exact memory it came from.

## How Hindsight memory is used

This is the part of the project the memory layer is judged on, so it's worth being explicit about what actually happens, not just what the pitch says.

| Hindsight call | Where | What it stores or returns |
|---|---|---|
| `retain()` | `memory.py: retain()` | One sentence per event: a published post's title/format/channel/date/views/conversions, a brand voice note, an approved or rejected idea, feedback the marketer typed in chat, or a revision instruction |
| `recall()` | `memory.py: recall()`, `recall_many()` | Searches the memory bank for sentences relevant to a question (e.g. "which content formats perform best") |
| `acreate_bank()` | `memory.py: ensure_bank()` | Creates one memory bank per company (`BANK_ID`, default `brightlane`) with a mission statement describing the agent's job, so Hindsight knows what to prioritize |

Every user-facing feature is built on top of these three calls:

- **Chat** (`POST /chat`) recalls memories relevant to the question, streams the LLM's answer, and shows exactly which memories were used.
- **Weekly planner** (`POST /plan`) recalls performance data, content gaps, brand voice, and past feedback, then asks the LLM for five ideas — each with a `src` field that must be an exact, verbatim memory the idea is based on. **If the model returns a citation that isn't a real memory, it's discarded server-side** (see `llm.py: _parse_plan`). This is the mechanism that keeps "based on memory" from being a decoration — a citation is either provably real or it isn't shown at all.
- **Proactive nudge** (`GET /nudge`) recalls content gaps and volunteers one, unprompted, after history is loaded.
- **Revise-in-chat** (`POST /revise`) lets the marketer type a plain-English instruction ("make it shorter") to change one plan item, and stores the instruction back into memory so future plans reflect it.
- **Feedback loop** (`POST /feedback`) retains every approval and rejection, so rejected angles are excluded from future plans both immediately (via a client-side `avoid` list) and permanently (via Hindsight memory).

### Before/after, concretely

With memory **off**, `/chat` and `/plan` never call `recall()` — the LLM gets no context and produces generic, interchangeable suggestions. With memory **on**, the same questions return answers grounded in this company's actual numbers, gaps, and past decisions. The UI's Memory toggle and "Compare on vs off" button exist specifically to make this difference visible in under a minute.

## Architecture

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/615b7366-45c4-41eb-8d81-af49c4f96ec3" />


- **`main.py`** — the only file with HTTP endpoints. Owns request/response shapes and orchestrates calls to `memory.py` and `llm.py`. Serves the frontend directly (avoids CORS).
- **`memory.py`** — every Hindsight call lives here. Nothing else in the codebase talks to Hindsight directly.
- **`llm.py`** — every Groq call lives here, plus the parsing/validation of the model's JSON output (with retries on malformed responses) and the citation-integrity check described above.
- **`static/index.html`** — a single-page vanilla JS frontend: chat, weekly planner, content log, and a learning-curve view. No build step.
- **`seed_data.json`** — synthetic history for a fictional company ("Brightlane"): brand voice notes and past published posts with real performance numbers, used to seed memory via `POST /load`.

## Endpoints

| Method & path | Purpose |
|---|---|
| `POST /chat` | Ask the agent a question. Streams the answer as Server-Sent Events: a `meta` event (memories used, confidence, a follow-up question) first, then `delta` text chunks, then `done` |
| `GET /memories` | List what's currently in memory, for the UI's memory panel |
| `POST /load` | Retain the seed data (brand voice + all past posts) into Hindsight — run once before a demo |
| `GET /content` | The content log table, read from `seed_data.json` |
| `POST /plan` | Generate five content ideas (Mon–Fri), each with a validated citation |
| `GET /nudge` | A proactive tip the agent volunteers about a content gap, without being asked |
| `POST /revise` | Revise one plan item based on a plain-English instruction |
| `POST /feedback` | Record an approval or rejection, retained into memory |
| `GET /health` | Liveness check |

## Setup

1. **Get a Hindsight API key.** Sign up at [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io), and create an API key.
2. **Get a Groq API key** at [console.groq.com](https://console.groq.com) (free tier is sufficient).
3. **Install dependencies:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Configure secrets:** copy `.env.example` to `.env` and fill in both API keys.
5. **Run the server:**
   ```bash
   uvicorn main:app --reload
   ```
6. Open `http://127.0.0.1:8000`.

## Trying it out

1. Click **Load history into memory** on the Content Log tab (retains the seed data — takes a little while, since Hindsight extracts facts from each item).
2. Go to the **Chat** tab, toggle **Memory** off, and ask "What should we write next?" — note the generic answer.
3. Toggle **Memory** on and ask the same question — the answer should now cite real performance numbers and gaps.
4. Try **Compare on vs off** to see both answers side by side.
5. Go to **Planner** and click **Generate this week's plan.** Expand "Why this?" on any idea to see its cited memory. Reject an idea and confirm the replacement avoids it (with a few seconds to **Undo**).
6. Click **Demo mode** for a scripted end-to-end walkthrough of all of the above.

## Design decisions worth knowing

- **Citations are verified, not trusted.** The LLM is asked to copy a memory verbatim into the `src` field; if what comes back isn't a real, exact memory string, it's dropped. This prevents the agent from inventing evidence for its own recommendations.
- **Confidence is computed, not generated.** A `"solid"` / `"limited"` / `"none"` label is derived from how many memories were actually recalled — the model is never asked to self-report how sure it is.
- **Only statements are retained from chat, not questions** ("What should we write next?" doesn't get stored; "that idea was too salesy" does) — otherwise the memory bank fills with the same recurring question.
- **Failures degrade gracefully.** If Hindsight or Groq is unreachable, endpoints return an empty result or a clear error rather than crashing, and the frontend falls back to local sample data so a demo can continue.

## Known limitations

- The seed dataset is 12 synthetic posts for one fictional company — enough to demonstrate the mechanism, not a production-scale dataset.
- The "learning curve" chart is illustrative; it does not yet plot a measured score across real interaction counts.
- Single-tenant: one Hindsight bank per deployment (`BANK_ID`), rather than dynamic per-client banks.

## Tech stack

Vanilla HTML/CSS/JS frontend, Python (FastAPI) backend, [Hindsight](https://github.com/vectorize-io/hindsight) for memory, Groq (`openai/gpt-oss-120b`) for generation.
