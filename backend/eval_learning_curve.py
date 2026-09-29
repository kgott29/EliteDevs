import asyncio
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import llm  # noqa: E402
import memory  # noqa: E402

RUNS = int(os.getenv("EVAL_RUNS", "3"))
SETTLE = float(os.getenv("EVAL_SETTLE", "20"))
CHECKPOINTS = [1, 5, 20]
HERE = Path(__file__).parent
OUT = HERE / "learning_curve.json"
ANSWERS = HERE / "eval_answers.json"

# Same searches main.py runs for /plan. Keep in sync if you change them there.
QUERIES = [
    "which content formats and channels perform best",
    "which topics were covered recently and which are missing",
    "brand voice and target audience",
    "ideas the marketer approved or rejected, and why",
]

FEEDBACK = [  # scripted marketer feedback, same sentence shape as main.py /feedback
    ("Webinar recap: our year in review", "rejected", "too salesy"),
    ("How to reconcile invoices automatically", "approved", ""),
    ("Ten reasons to switch to automation", "rejected", "too salesy"),
    ("Quick tip: rename your workflows", "approved", ""),
    ("Industry trends roundup", "rejected", "generic"),
]


def feedback_sentence(title, action, reason):
    s = f"The marketer {action} the content idea: '{title}'."
    if reason:
        s += f" Reason: {reason}."
    if action == "rejected":
        s += " Do not suggest similar ideas again."
    return s


def build_events():
    seed = memory.load_seed()
    ev = [("brand voice", n, None) for n in seed["voice_notes"]]
    for p in sorted(seed["posts"], key=lambda p: p["published"]):
        when = datetime.fromisoformat(f"{p['published']}T09:00:00+00:00")
        ev.append(("content performance", memory.post_to_sentence(p), when))
    for t, a, r in FEEDBACK:
        ev.append(("feedback on a suggested idea", feedback_sentence(t, a, r), None))
    return ev  # 19 events


# ------------------------------------------------------------------ scoring

def build_facts(posts):
    """Every number the agent may legitimately cite, raw or derived."""
    facts = []
    by = {}
    for p in posts:
        facts += [p["views"], p["conversions"], p["conversions"] / p["views"] * 100]
        facts += [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", p["title"])]
        by.setdefault(p["format"], []).append(p)
    avg = {}
    for f, rows in by.items():
        v = statistics.mean(r["views"] for r in rows)
        c = statistics.mean(r["conversions"] for r in rows)
        rate = statistics.mean(r["conversions"] / r["views"] * 100 for r in rows)
        avg[f] = (v, c)
        facts += [v, c, rate, sum(r["views"] for r in rows), sum(r["conversions"] for r in rows), len(rows)]
    for a, b in permutations(avg, 2):
        facts += [avg[a][0] / avg[b][0], avg[a][1] / avg[b][1]]
    return facts


MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
DATE = re.compile(MONTHS + r"\s+\d{1,2}(?:,?\s+\d{4})?|\d{4}-\d{2}-\d{2}", re.I)
NUM = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)([kK%x×](?![A-Za-z]))?"
    r"(\s+(?:weeks?|days?|hours?|minutes?|months?|posts?|pieces?|items?|ideas?))?"
)


def matches(v, facts):
    for x in facts:
        tol = 0.06 if x < 20 else max(0.6, 0.006 * x)
        if abs(v - x) <= tol:
            return True
    return False


def grams(s, n=3):
    w = re.findall(r"[a-z0-9']+", s.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def score_item(why, facts, title_grams):
    """Returns (supported, fabricated) for one plan item."""
    text = DATE.sub(" ", why)
    good = bad = 0
    for m in NUM.finditer(text):
        raw, suffix, unit = m.groups()
        if unit:  # "9 weeks", "5 items": not a claim we can check
            continue
        v = float(raw.replace(",", ""))
        if v == 2026 or (v <= 5 and suffix not in ("x", "×", "%")):
            continue
        if suffix in ("k", "K"):
            v *= 1000
        if matches(v, facts):
            good += 1
        else:
            bad += 1
    named = bool(grams(why) & title_grams)
    return (good > 0 or named) and bad == 0, bad > 0


def score_plan(plan, posts):
    facts = build_facts(posts)
    tg = set().union(*(grams(p["title"]) for p in posts))
    res = [score_item(it["w"], facts, tg) for it in plan]
    return sum(r[0] for r in res) / len(plan), sum(r[1] for r in res)


# ---------------------------------------------------------------------- run

async def retain_all(events):
    sem = asyncio.Semaphore(3)

    async def one(e):
        async with sem:
            return await memory.retain(*e)

    return await asyncio.gather(*(one(e) for e in events))


async def ask():
    used = await memory.recall_many(QUERIES, per_query=8)
    return used, await llm.weekly_plan(used, None)


async def run_once(i, stamp, posts):
    memory.BANK_ID = f"brightlane-eval-{stamp}-{i}"
    await memory.ensure_bank()
    events, sent, out = build_events(), 0, []
    for cp in CHECKPOINTS:
        need = cp - 1
        if need > sent:
            ok = await retain_all(events[sent:need])
            if not all(ok):
                print(f"  warning: {ok.count(False)} retain calls failed", flush=True)
            sent = need
            await asyncio.sleep(SETTLE)  # give Hindsight time to index
        used, plan = await ask()
        score, fab = score_plan(plan, posts)
        print(f"  run {i + 1}, interaction {cp}: score {score:.0%}, "
              f"made-up numbers {fab}, memories used {len(used)}", flush=True)
        out.append({"interaction": cp, "score": score, "fabricated": fab, "plan": plan})
    fn = getattr(memory.client(), "adelete_bank", None)  # best-effort cleanup
    if fn:
        try:
            await fn(bank_id=memory.BANK_ID)
        except Exception:
            pass
    return out


async def main():
    posts = memory.load_seed()["posts"]
    stamp, runs = int(time.time()), []
    for i in range(RUNS):
        print(f"Run {i + 1}/{RUNS}", flush=True)
        try:
            runs.append(await run_once(i, stamp, posts))
        except Exception as e:
            print(f"  run failed: {e}")
    if not runs:
        sys.exit("All runs failed. Check your API keys and network.")
    points = []
    for k, cp in enumerate(CHECKPOINTS):
        s = [r[k]["score"] for r in runs]
        points.append({
            "label": f"Interaction {cp}", "score": round(statistics.mean(s), 3),
            "min": min(s), "max": max(s),
            "fabricated": round(statistics.mean(r[k]["fabricated"] for r in runs), 2),
        })
    OUT.write_text(json.dumps({
        "question": "Plan next week's content (5 ideas)",
        "metric": "share of the 5 ideas that cite real evidence and no made-up numbers",
        "runs": len(runs), "generated": datetime.now(timezone.utc).isoformat(),
        "points": points,
    }, indent=2), encoding="utf-8")
    
    ANSWERS.write_text(json.dumps(
        {f"interaction_{r['interaction']}": r["plan"] for r in runs[0]}, indent=2), encoding="utf-8")
    print("\nSaved", OUT.name, "and", ANSWERS.name)
    for p in points:
        print(f"  {p['label']}: {p['score']:.0%} (range {p['min']:.0%}-{p['max']:.0%})")
        close = getattr(memory.client(), "aclose", None)
    if close:
        await close()


if __name__ == "__main__":
    asyncio.run(main())