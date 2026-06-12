"""Debug the two-turn extraction — shows raw output from both turns."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sarvam = OpenAI(api_key=os.environ["SARVAM_API_KEY"], base_url="https://api.sarvam.ai/v1")

TEXT = (
    "GraphRAG combines graph-based knowledge representation with retrieval-augmented generation. "
    "Microsoft Research published the original GraphRAG paper in 2024. "
    "Darren Edge and Ha Trinh led the research team at Microsoft. "
    "The technique significantly improves multi-hop question answering compared to naive RAG."
)

extraction_prompt = (
    "You are an expert knowledge extraction system. Carefully read the text below and identify "
    "every named entity and every meaningful relationship between them. Think through the text thoroughly.\n\n"
    "Text:\n" + TEXT
)

FORMAT_PROMPT = (
    "Based on your analysis above, output ONLY a valid JSON object with no explanation, "
    "no markdown fences, no extra text. The JSON must follow this exact schema:\n\n"
    '{"entities": [{"name": "...", "type": "PERSON or ORG or CONCEPT or LOCATION or EVENT", "description": "..."}], '
    '"relations": [{"source": "...", "target": "...", "relationship": "..."}]}\n\n'
    "Fill in real values from your analysis. Use only these types: PERSON, ORG, CONCEPT, LOCATION, EVENT."
)

print("=== TURN 1 ===")
resp1 = sarvam.chat.completions.create(
    model="sarvam-30b",
    messages=[{"role": "user", "content": extraction_prompt}],
    max_tokens=4096,
    extra_body={"reasoning_effort": "low"},
)
msg1 = resp1.choices[0].message
thinking = msg1.content or getattr(msg1, "reasoning_content", None) or ""
print(f"content populated : {msg1.content is not None}")
print(f"thinking length   : {len(thinking)} chars")
print(f"last 300 chars    :\n{thinking[-300:]}\n")

print("=== TURN 2 ===")
resp2 = sarvam.chat.completions.create(
    model="sarvam-30b",
    messages=[
        {"role": "user",      "content": extraction_prompt},
        {"role": "assistant", "content": thinking},
        {"role": "user",      "content": FORMAT_PROMPT},
    ],
    max_tokens=2048,
    extra_body={"reasoning_effort": "low"},
)
msg2 = resp2.choices[0].message
raw2 = msg2.content or getattr(msg2, "reasoning_content", None) or ""
print(f"content populated : {msg2.content is not None}")
print(f"raw length        : {len(raw2)} chars")
print(f"\nFULL RAW OUTPUT:\n{raw2}")
