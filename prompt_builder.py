"""
Wires the hashtag/entity KB into the best_86_61 prompt.

Design decision: inject only the KB entries that actually match the current
tweet's hashtags/entities, not the full 23+6 entry table on every call.
Reasons:
  1. Keeps prompt length reasonable (most tweets only carry 1-2 hashtags).
  2. Avoids diluting the model's attention with irrelevant reference entries -
     same failure mode as the earlier bundled-rule regressions (CI v2, doubt-framing):
     more unrelated content in the prompt tends to hurt, not help.
  3. Matches the "flat, per-lookup" design agreed on, not a dump of the whole KB.

This is the ONLY new variable versus best_86_61. No other prompt text is
touched (background paragraph, mental-rewrite instruction, event-favor clause,
guidelines, output format - all identical to best_86_61).
"""

import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(_HERE, "hashtag_entity_kb.json")

with open(KB_PATH, encoding="utf-8") as f:
    KB = json.load(f)

# ── best_86_61, verbatim, unchanged ──────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are an expert annotator for Arabic stance detection.

### Background Context
The target concerns the 2017–2018 Saudi policy change allowing women to drive. Before June 2018, women were prohibited from driving in Saudi Arabia. Tweets from this period often discuss the royal decree, implementation, licensing, religion, tradition, safety, gender roles, media coverage, and public reaction. They frequently use Saudi dialect, sarcasm, humor, rhetorical questions, and indirect expressions.

{kb_section}### Task
Given an Arabic tweet and a target topic, classify the writer's stance toward the target as exactly one of:

• Favor
• Against
• None

Before assigning a stance, mentally rewrite the tweet into its intended literal meaning while preserving the writer's opinion, sarcasm, dialect, rhetorical intent, and emojis.

Then determine the stance toward the target itself, not toward other people, quoted opinions, related entities, or hashtags.

Guidelines:

• Favor: supports, defends, promotes, or welcomes the target.
• Against: opposes, criticizes, rejects, or mocks the target.
• None: no clear stance toward the target.

Important:

• Determine where praise or criticism is directed. Negative language toward opponents of the target is usually Favor, not Against.
• Hashtags may be ironic or hijacked. Never infer stance from hashtags alone.
• Rhetorical questions, sarcasm, and emojis often convey the writer's true stance. Interpret the intended meaning rather than the literal wording.
• Distinguish reporting from endorsement. Mentioning an event or policy does not by itself express a stance, unless it is framed positively (e.g. promoting, celebrating, or inviting participation), in which case it leans Favor.
• If the stance toward the target cannot reasonably be inferred, output None.

Respond with ONLY one word:

Favor
Against
None

Do not provide any explanation, punctuation, or additional text."""


def find_matches(tweet_text: str):
    """Return list of (type, canonical, gloss) for KB entries present in the tweet."""
    matches = []
    for h in KB["hashtags"]:
        if any(v in tweet_text for v in h["variants"]):
            matches.append(("hashtag", h["canonical"], h["gloss"]))
    for e in KB["entities"]:
        if any(v in tweet_text for v in e["variants"]):
            matches.append(("entity", e["canonical"], e["gloss"]))
    return matches


def build_kb_section(tweet_text: str) -> str:
    """Build the injected reference block for this specific tweet, or empty string if no match."""
    matches = find_matches(tweet_text)
    if not matches:
        return ""

    lines = ["### Reference (factual only — do not infer stance from these alone)\n"]
    for kind, canonical, gloss in matches:
        label = "Hashtag" if kind == "hashtag" else "Entity"
        lines.append(f"- {label} — {canonical}: {gloss}")
    lines.append("")  # trailing blank line before ### Task
    return "\n".join(lines) + "\n"


def build_prompt_for_tweet(tweet_text: str) -> str:
    """Returns the full system prompt for this specific tweet (KB section injected only if relevant)."""
    kb_section = build_kb_section(tweet_text)
    return BASE_SYSTEM_PROMPT.format(kb_section=kb_section)


if __name__ == "__main__":
    # Quick self-check on a few real tweets
    import csv
    _DATA = os.path.join(_HERE, "data", "ground_truth.csv")
    with open(_DATA, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sample = rows[3]  # tweet with #المراه_السعوديه_تسوق mentioning عادل الجبير context
    prompt = build_prompt_for_tweet(sample["tweet_text"])
    print("=== SAMPLE TWEET ===")
    print(sample["tweet_text"])
    print("\n=== INJECTED KB SECTION ===")
    print(build_kb_section(sample["tweet_text"]) or "(no match)")

    # Coverage/length stats across all 352
    lengths = []
    no_match_count = 0
    for r in rows:
        matches = find_matches(r["tweet_text"])
        if not matches:
            no_match_count += 1
        lengths.append(len(matches))

    print(f"\nTweets with 0 KB matches: {no_match_count}/352")
    print(f"Avg matches per tweet: {sum(lengths)/len(lengths):.2f}")
    print(f"Max matches on one tweet: {max(lengths)}")
