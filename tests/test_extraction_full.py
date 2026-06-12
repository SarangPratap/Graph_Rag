"""Count fallbacks across all chunks in the sample PDF."""

from __future__ import annotations

import logging
import os
import re
import json

from dotenv import load_dotenv
from openai import OpenAI

from graphrag.ingestion.chunker import chunk_documents
from graphrag.ingestion.pdf_loader import load_pdfs_from_folder
from graphrag.extraction.entity_extractor import _build_prompt, _extract_json_object

load_dotenv()

sarvam = OpenAI(
    api_key=os.environ["SARVAM_API_KEY"],
    base_url="https://api.sarvam.ai/v1",
)


def probe_chunk(chunk, i, total):
    """Extract from one chunk and report which fallback path was taken."""
    prompt = _build_prompt(chunk.text)
    try:
        response = sarvam.chat.completions.create(
            model="sarvam-30b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
        )
        msg = response.choices[0].message
        content_none = msg.content is None
        raw = msg.content or getattr(msg, "reasoning_content", None) or ""

        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        extracted = _extract_json_object(cleaned)

        try:
            parsed = json.loads(extracted)
            n_ent = len(parsed.get("entities", []))
            n_rel = len(parsed.get("relations", []))
            fallback = "L1:reasoning_content" if content_none else "ok"
            print(f"[{i:3}/{total}] {fallback:22s}  entities={n_ent:2d}  relations={n_rel:2d}")
            return fallback, n_ent, n_rel
        except Exception as e:
            fallback = "L2:json_parse_fail"
            print(f"[{i:3}/{total}] {fallback:22s}  ({type(e).__name__}: {str(e)[:60]})")
            return fallback, 0, 0

    except Exception as e:
        fallback = "L3:api_or_crash"
        print(f"[{i:3}/{total}] {fallback:22s}  ({type(e).__name__}: {str(e)[:60]})")
        return fallback, 0, 0


def main():
    docs = load_pdfs_from_folder("./sample_pdfs")
    chunks = chunk_documents(docs)
    total = len(chunks)
    print(f"\nLoaded {len(docs)} doc(s), {total} chunks. Running extraction...\n")

    counts = {"ok": 0, "L1:reasoning_content": 0, "L2:json_parse_fail": 0, "L3:api_or_crash": 0}
    total_entities = 0
    total_relations = 0

    for i, chunk in enumerate(chunks, 1):
        label, n_ent, n_rel = probe_chunk(chunk, i, total)
        key = label if label in counts else "L2:json_parse_fail"
        counts[key] += 1
        total_entities += n_ent
        total_relations += n_rel

    print(f"\n{'='*55}")
    print(f"SUMMARY  ({total} chunks)")
    print(f"{'='*55}")
    print(f"  Success (content populated)  : {counts['ok']:3d}")
    print(f"  L1 fallback (reasoning_content used) : {counts['L1:reasoning_content']:3d}")
    print(f"  L2 fallback (JSON parse failed)      : {counts['L2:json_parse_fail']:3d}")
    print(f"  L3 fallback (API/crash)              : {counts['L3:api_or_crash']:3d}")
    print(f"  Total entities extracted     : {total_entities}")
    print(f"  Total relations extracted    : {total_relations}")


if __name__ == "__main__":
    main()
