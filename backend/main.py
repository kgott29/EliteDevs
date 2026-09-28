"""
Step 1: "Hello memory" test.

Goal: prove that we can (1) connect to Hindsight, (2) store one fact,
(3) get that fact back. Run it with:  python main.py
"""

import os
import time

from dotenv import load_dotenv
from hindsight_client import Hindsight

# 1. Load secrets from the .env file into environment variables.
load_dotenv()

HINDSIGHT_URL = os.getenv("HINDSIGHT_URL", "https://api.hindsight.vectorize.io")
HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY")
BANK_ID = "acme-saas-test"  # the "memory folder" for this test

if not HINDSIGHT_API_KEY:
    raise SystemExit("HINDSIGHT_API_KEY is missing. Add it to your .env file.")

# 2. Connect to Hindsight.
client = Hindsight(base_url=HINDSIGHT_URL, api_key=HINDSIGHT_API_KEY)

# 3. Create the memory bank (ignore the error if it already exists).
try:
    client.create_bank(bank_id=BANK_ID, name="Acme SaaS Test Bank")
    print("Created bank:", BANK_ID)
except Exception as e:
    print("Bank not created (it probably already exists):", e)

# 4. RETAIN: store one fact.
fact = (
    "Blog post 'How to cut cloud costs' was published on LinkedIn. "
    "It got 4,200 views and 90 shares. Format: how-to guide."
)
client.retain(bank_id=BANK_ID, content=fact, context="content performance")
print("Retained:", fact)

# 5. RECALL: search for it.
# Hindsight processes what you store (an LLM extracts facts from the text),
# so it may take a few seconds before the memory is searchable.
query = "Which content performed well?"
for attempt in range(1, 7):
    result = client.recall(bank_id=BANK_ID, query=query)
    if result.results:
        print(f"\nRecall worked (attempt {attempt}). Memories found:")
        for memory in result.results:
            print(" -", memory.text)
        break
    print(f"Attempt {attempt}: nothing yet, waiting 5 seconds...")
    time.sleep(5)
else:
    print("\nNo memories came back. Check the Hindsight UI to see if the fact was stored.")

client.close()