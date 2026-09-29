# How I Stopped My Hindsight Agent From Inventing Citations

The first time my content strategy agent told a marketer "your how-to posts get 3x the engagement, based on this memory," I checked the memory it pointed to. It didn't exist. The model had written a citation that sounded exactly like the ones it had seen in its own training data — confident, specific, plausible — and completely made up. That was the moment I realized giving an agent memory is the easy part. Making it tell the truth about what it remembers is the hard part.

## What the agent does

The system is a content strategist for a B2B SaaS marketing team. It watches what a company publishes, how each piece performs, and what the team approves or rejects, and it uses that history to plan the next week of content. The pitch is simple: most marketing teams reinvent the wheel every week because nobody remembers what actually worked. An agent with persistent memory doesn't have that problem.

Under the hood it's three pieces talking to each other. A FastAPI backend sits in the middle. On one side it calls [Hindsight](https://hindsight.vectorize.io/), the memory layer from [Vectorize](https://vectorize.io/what-is-agent-memory) that stores and retrieves what the agent has learned. On the other side it calls an LLM (I used Groq) to turn those memories into an actual plan. The frontend is deliberately plain — a chat panel, a weekly planner, a memory sidebar — because the interesting engineering isn't in the UI, it's in what happens between "the model recalled five memories" and "the model wrote five sentences."

Every time something happens — a post goes live, an idea gets rejected, a marketer gives feedback — the backend calls `retain()` on a Hindsight memory bank scoped to that company. Every time the agent needs to answer something, it calls `recall()` first and feeds whatever comes back into the prompt. That part of the architecture is almost boringly simple. The part that took real iteration was making sure the agent's claims about its own memory were actually true.

## The core problem: language models love to sound certain

Here's the failure mode, concretely. I'd ask the planner for next week's content ideas, and it would return something like:

> "Write a how-to on auditing automations for access risks. Your how-tos average 9.4k views against 3.0k for opinion pieces."

That's a great answer — specific, actionable, grounded in real numbers. Except sometimes those numbers were real, and sometimes the model had pattern-matched to "this is the kind of thing a data-backed recommendation sounds like" and quietly invented them. From the outside, a true recommendation and a hallucinated one are indistinguishable. That's a serious problem for a product whose entire value proposition is "trust me, I remember."

The fix isn't "tell the model not to hallucinate" — I tried that, and it helps a little and isn't remotely reliable. The fix is to stop trusting the model's claims about its own sources and verify them in code.

## Making citations provable, not just plausible

The change I'm most proud of is small in terms of line count and large in terms of what it does to the product's honesty. Every idea the planner generates now includes a `src` field, and the model is instructed to copy that field *verbatim* from the memories it was actually given:

```python
'"src" (copy one sentence EXACTLY, word for word, from the memories '
'above that this idea is based on; copy nothing and leave it as an '
'empty string if you have no memories or this idea is not based on '
"one)."
```

That instruction alone doesn't solve anything, because a model that's willing to invent a statistic is equally willing to invent a citation for it. The actual fix happens after the model responds, in code the model never sees:

```python
known = set(memories or [])
for i, item in enumerate(items[:5]):
    row = {k: str(item.get(k, "")).strip() for k in KEYS}
    row["d"] = row["d"] or DAYS[i]
    if not row["t"]:
        raise ValueError("item without a title")
    # Only keep the citation if it is a real memory we actually gave the
    # model. A model-invented "src" would be a fake citation.
    if row["src"] not in known:
        row["src"] = ""
    plan.append(row)
```

`known` is the exact set of memory strings that were recalled from Hindsight and placed in the prompt for this request. If the string the model returns in `src` isn't a member of that set — character for character — it gets wiped out server-side before the plan ever reaches the UI. There's no partial credit, no fuzzy matching, no "close enough." A citation is either a memory that genuinely came out of `recall()`, or it's nothing.

The UI change that follows from this is just as important as the backend one. Instead of a vague "based on your memory" badge, each idea in the planner shows the actual sentence it's citing, quoted, right under the idea. If there's no `src`, there's no quote — the idea is shown as a plain suggestion with no implied evidence behind it. The agent went from asserting expertise to demonstrating it, and the difference is that one of those can be checked and the other can't.

## Streaming made the honesty problem harder, not easier

I wanted the chat responses to stream in — the difference between an agent that appears instantly and one that visibly "thinks" while it writes back matters a lot for how trustworthy it feels. But streaming text and verifying claims are in tension: you want to start showing tokens before you know what the model is going to claim.

The way I resolved that is by sending the trust information *before* any text, as a separate event in the stream:

```python
async def gen():
    yield _sse("meta", {
        "used": used[:8],
        "confidence": llm.confidence(memories),
        "follow_up": llm.follow_up(memories),
    })
    full = ""
    try:
        async for piece in llm.stream_chat(message, memories):
            full += piece
            yield _sse("delta", {"text": piece})
    except Exception as e:
        yield _sse("delta", {"text": f"\n\n(The language model failed: {e})"})
    yield _sse("done", {})
```

The `meta` event carries exactly which memories were recalled for this answer and a `confidence` label — `"solid"` if four or more relevant memories came back, `"limited"` otherwise, `"none"` if memory was off. That label isn't generated by the LLM either; it's a plain function of how much real evidence exists:

```python
def confidence(memories: list[str] | None) -> str:
    if not memories:
        return "none"
    return "solid" if len(memories) >= 4 else "limited"
```

The UI renders that meta event first, so by the time the answer finishes streaming in, the person reading it already knows how much to trust it. An agent that says "here's what I think, and here's exactly how much I actually know" is a different product than one that just sounds confident.

## The bug that taught me not to trust my own heuristics either

Citations weren't the only place I'd quietly let the system trust something it shouldn't have. I built a feature where the agent proactively volunteers a content gap — "you haven't covered security in nine weeks" — without being asked, which is a genuinely nice memory-powered touch. My first version picked a memory to surface using a keyword check: if the recalled text contained words like `"gap"`, `"no "`, or `"weeks"`, it was a gap worth mentioning.

It found a gap, all right. It surfaced this to a marketer as a content gap: *"Brand voice: we sound casual, with short sentences and no jargon."* The word "no" in "no jargon" was enough to trip a heuristic that was never checking meaning, only substring presence. It's a small bug, but it's the exact same category of mistake as the citation problem: I'd built something that looked like it understood its own memory, when really it was pattern-matching on surface features and getting lucky most of the time.

The fix was to require a stronger, more specific signal — the actual word "week" combined with a real negation phrase, not just any string containing "no":

```python
negatives = ("no ", "not been", "haven't", "have not", "last covered")
for text in candidates:
    low = text.lower()
    if "week" in low and any(w in low for w in negatives):
        return text
```

It's still a heuristic, not a semantic understanding of the sentence, and I'm not pretending otherwise. But it's a heuristic I can reason about and that fails in obvious, catchable ways rather than confidently misclassifying brand guidelines as a content strategy problem.

## What I'd tell someone starting the same project

**Verification has to live outside the model, not inside the prompt.** Every guardrail I trusted to a system-prompt instruction eventually got ignored under the right conditions. Every guardrail I implemented as a code-level check — set membership, a length threshold, a keyword match — held.

**Memory is only valuable if you can point at it.** Retrieving relevant context and building a good answer from it is necessary, but it's not the differentiator. The differentiator is being able to say "here is the exact fact that justifies this answer," in a way a skeptical user can click on and read for themselves.

**Confidence should be computed, not generated.** Asking the model to self-report how sure it is produces a plausible-sounding number that has no relationship to how much evidence actually exists. Counting the evidence and deriving a label from the count is boring, and it's honest.

**Streaming changes your architecture, not just your UI.** Once responses arrive incrementally, you have to decide what the client can trust before the text has even finished — which forces you to formalize your trust logic instead of leaving it implicit in the final response shape.

**Test your own heuristics as adversarially as you'd test the model.** The nudge bug didn't come from the LLM at all — it came from code I wrote and trusted without trying to break it. The parts of the system I was most confident about turned out to be the parts I'd checked the least.

None of this makes the agent smarter in the way a bigger model or a longer context window would. It makes the agent honest about the difference between what it knows and what it's guessing, which turns out to matter a lot more once a real person is deciding whether to act on what it says. If you're building something on [Hindsight](https://github.com/vectorize-io/hindsight), that's the bar I'd aim for: not an agent that remembers more, but one you can actually trust when it tells you what it remembers.