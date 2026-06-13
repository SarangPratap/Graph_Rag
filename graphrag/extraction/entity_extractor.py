"""Extract entities and relations from text chunks using Sarvam AI (two-turn strategy)."""

from __future__ import annotations

import json
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from graphrag.models import Chunk, Entity, ExtractionResult, Relation

load_dotenv()

logger = logging.getLogger(__name__)

sarvam = OpenAI(
    api_key=os.environ["SARVAM_API_KEY"],
    base_url="https://api.sarvam.ai/v1",
)

_VALID_TYPES = {"PERSON", "ORG", "CONCEPT", "LOCATION", "EVENT"}

# Turn 1 — free reasoning, no format pressure
_EXTRACTION_PROMPT = """You are an expert knowledge extraction system. Carefully read the text below and identify:
- Every named entity (people, organizations, concepts, locations, events)
- Every meaningful relationship between those entities

Think through the text thoroughly. There is no output format required — just reason carefully.

Text:
{text_chunk}"""

# Turn 2 — model sees its own reasoning, just formats the final answer
_FORMAT_PROMPT = """Based on your analysis above, output ONLY the following JSON object with no explanation, no markdown, no extra text:

{
  "entities": [
    {"name": "Entity Name", "type": "PERSON|ORG|CONCEPT|LOCATION|EVENT", "description": "brief description"}
  ],
  "relations": [
    {"source": "Entity A", "target": "Entity B", "relationship": "how they connect"}
  ]
}

Use only these entity types: PERSON, ORG, CONCEPT, LOCATION, EVENT."""


def extract(chunk: Chunk) -> ExtractionResult:
    """Extract entities and relations from a single Chunk using a two-turn Sarvam conversation.

    Turn 1 lets the model reason freely. Turn 2 asks it to format its findings as
    clean JSON — short response, no token pressure, reliable output.
    Returns an ExtractionResult with empty lists on any failure.
    """
    try:
        extraction_prompt = _EXTRACTION_PROMPT.replace("{text_chunk}", chunk.text)

        # Turn 1: reason freely about the text
        resp1 = sarvam.chat.completions.create(
            model="sarvam-30b",
            messages=[{"role": "user", "content": extraction_prompt}],
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
        )
        msg1 = resp1.choices[0].message
        thinking = msg1.content or getattr(msg1, "reasoning_content", None) or ""

        # Turn 2: convert reasoning into final JSON
        resp2 = sarvam.chat.completions.create(
            model="sarvam-30b",
            messages=[
                {"role": "user",      "content": extraction_prompt},
                {"role": "assistant", "content": thinking},
                {"role": "user",      "content": _FORMAT_PROMPT},
            ],
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"},
        )
        msg2 = resp2.choices[0].message
        raw = msg2.content or getattr(msg2, "reasoning_content", None) or ""

        parsed = _parse_json(raw)

        # If parsing yielded nothing, retry turn 2 once — the model occasionally
        # buries the JSON inside another reasoning block on the first attempt.
        if not parsed.get("entities") and not parsed.get("relations"):
            logger.warning("Empty parse for chunk %s — retrying turn 2", chunk.id)
            resp2b = sarvam.chat.completions.create(
                model="sarvam-30b",
                messages=[
                    {"role": "user",      "content": extraction_prompt},
                    {"role": "assistant", "content": thinking},
                    {"role": "user",      "content": _FORMAT_PROMPT},
                ],
                max_tokens=4096,
                extra_body={"reasoning_effort": "low"},
            )
            msg2b = resp2b.choices[0].message
            raw = msg2b.content or getattr(msg2b, "reasoning_content", None) or ""
            parsed = _parse_json(raw)

        return _build_result(chunk, parsed)

    except Exception as exc:
        logger.error("Extraction failed for chunk %s: %s", chunk.id, exc)
        return ExtractionResult(chunk_id=chunk.id)


def extract_batch(chunks: list[Chunk]) -> list[ExtractionResult]:
    """Extract entities and relations from every Chunk in the list (sequential)."""
    total = len(chunks)
    results: list[ExtractionResult] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Extracting chunk {i}/{total}...")
        results.append(extract(chunk))
    return results


def extract_batch_parallel(
    chunks: list[Chunk], max_workers: int = 5
) -> list[ExtractionResult]:
    """Extract entities from chunks in parallel using a thread pool.

    Order of results matches the order of input chunks.
    Failed chunks return an empty ExtractionResult so the pipeline never aborts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(chunks)
    results: list[ExtractionResult | None] = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(extract, chunk): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed += 1
            print(f"Extracting chunks: {completed}/{total}", flush=True)
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Chunk %s failed: %s", chunks[idx].id, exc)
                results[idx] = ExtractionResult(chunk_id=chunks[idx].id)

    return [r for r in results if r is not None]


def _clean_response(raw: str) -> str:
    """Strip thinking tags and all code-fence variants from a model response."""
    # Remove <think>...</think> blocks (may be unclosed — strip greedily)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Fallback: if tag is unclosed, drop everything after <think>
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    # Strip code fences: ```json, ```JSON, ``` etc.
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")
    return text.strip()


def _parse_json(raw: str) -> dict:
    """Parse the first balanced JSON object from a model response.

    Three-layer strategy:
      1. Direct json.loads on the cleaned text (fastest, works for clean responses)
      2. Balanced-brace walker starting at the {"entities" key
      3. Regex salvage — extract whatever complete entity/relation objects exist
         in a truncated or malformed response rather than discarding the whole chunk
    """
    text = _clean_response(raw)

    # Layer 1: try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Layer 2: find and walk the outermost { ... } block
    start = text.find('{"entities"')
    if start == -1:
        start = text.find('{')
    if start != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError as exc:
                        logger.warning("JSON parse failed after brace walk: %s", exc)
                        break

    # Layer 3: regex salvage for truncated responses
    logger.warning("Falling back to regex salvage on %d-char response", len(text))
    return _salvage_json(text)


def _salvage_json(text: str) -> dict:
    """Recover individual entity and relation objects from a malformed/truncated response."""
    entities = []
    relations = []

    # Match complete entity objects regardless of key order
    for m in re.finditer(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]*)"\s*\}',
        text,
    ):
        entities.append({"name": m.group(1), "type": m.group(2), "description": m.group(3)})

    # Match complete relation objects
    for m in re.finditer(
        r'\{\s*"source"\s*:\s*"([^"]+)"\s*,\s*"target"\s*:\s*"([^"]+)"\s*,\s*"relationship"\s*:\s*"([^"]*)"\s*\}',
        text,
    ):
        relations.append({"source": m.group(1), "target": m.group(2), "relationship": m.group(3)})

    if entities or relations:
        logger.info("Salvaged %d entities and %d relations", len(entities), len(relations))
    else:
        logger.warning("Salvage found nothing — chunk will have no entities")

    return {"entities": entities, "relations": relations}


def _build_result(chunk: Chunk, parsed: dict) -> ExtractionResult:
    """Convert a parsed JSON dict into an ExtractionResult."""
    entities: list[Entity] = []
    name_to_id: dict[str, str] = {}

    for raw_ent in parsed.get("entities", []):
        try:
            ent_type = str(raw_ent.get("type", "CONCEPT")).upper()
            if ent_type not in _VALID_TYPES:
                ent_type = "CONCEPT"
            entity = Entity(
                name=raw_ent["name"],
                type=ent_type,
                description=raw_ent.get("description", ""),
                chunk_id=chunk.id,
            )
            entities.append(entity)
            name_to_id[raw_ent["name"]] = entity.id
        except Exception as exc:
            logger.warning("Skipping malformed entity %s: %s", raw_ent, exc)

    relations: list[Relation] = []
    for raw_rel in parsed.get("relations", []):
        try:
            src_name = raw_rel.get("source") or raw_rel.get("source_name", "")
            tgt_name = raw_rel.get("target") or raw_rel.get("target_name", "")
            src_id = name_to_id.get(src_name)
            tgt_id = name_to_id.get(tgt_name)
            if not src_id or not tgt_id:
                continue
            rel_type = raw_rel.get("relationship") or raw_rel.get("type", "RELATES_TO")
            weight = max(0.0, min(1.0, float(raw_rel.get("weight", 0.5))))
            relations.append(Relation(
                source_id=src_id,
                target_id=tgt_id,
                type=rel_type,
                weight=weight,
            ))
        except Exception as exc:
            logger.warning("Skipping malformed relation %s: %s", raw_rel, exc)

    return ExtractionResult(chunk_id=chunk.id, entities=entities, relations=relations)
