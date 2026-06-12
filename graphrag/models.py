"""Pydantic v2 data models shared across the entire GraphRAG pipeline."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _md5(value: str) -> str:
    """Return the MD5 hex digest of a UTF-8 string."""
    return hashlib.md5(value.encode()).hexdigest()


class Document(BaseModel):
    """A source document loaded from a PDF or web page."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="MD5 hash of document text — auto-computed if omitted")
    title: str = Field(description="Human-readable document title")
    source: str = Field(description="Absolute file path or URL of the source")
    text: str = Field(description="Full extracted plain text of the document")
    pages: int = Field(description="Number of pages (PDFs) or 0 for web content")
    checksum: str = Field(description="MD5 hash of document text — identical to id")
    ingested_at: datetime = Field(
        default_factory=datetime.now,
        description="UTC timestamp when the document was ingested",
    )

    @model_validator(mode="before")
    @classmethod
    def compute_ids(cls, data: Any) -> Any:
        """Auto-compute id and checksum from text when not supplied."""
        if isinstance(data, dict):
            text = data.get("text", "")
            digest = _md5(text)
            if not data.get("id"):
                data["id"] = digest
            if not data.get("checksum"):
                data["checksum"] = digest
        return data


class Chunk(BaseModel):
    """A paragraph-boundary chunk derived from a Document."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="MD5 hash of chunk text — auto-computed if omitted")
    text: str = Field(description="Plain text content of the chunk")
    source: str = Field(description="Absolute file path or URL inherited from the parent document")
    doc_title: str = Field(description="Title of the parent document")
    doc_id: str = Field(description="ID of the parent Document this chunk belongs to")
    position: int = Field(description="Zero-based index of this chunk within its document")

    @model_validator(mode="before")
    @classmethod
    def compute_id(cls, data: Any) -> Any:
        """Auto-compute id from chunk text when not supplied."""
        if isinstance(data, dict) and not data.get("id"):
            data["id"] = _md5(data.get("text", ""))
        return data


class Entity(BaseModel):
    """A named entity extracted from a Chunk by Sarvam AI."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="MD5 hash of lowercased entity name — auto-computed if omitted")
    name: str = Field(description="Canonical surface form of the entity")
    type: Literal["PERSON", "ORG", "CONCEPT", "LOCATION", "EVENT"] = Field(
        description="Entity category — one of PERSON, ORG, CONCEPT, LOCATION, EVENT"
    )
    description: str = Field(description="One-sentence description of the entity")
    chunk_id: str = Field(description="ID of the Chunk this entity was extracted from")

    @model_validator(mode="before")
    @classmethod
    def compute_id(cls, data: Any) -> Any:
        """Auto-compute id from lowercased entity name when not supplied."""
        if isinstance(data, dict) and not data.get("id"):
            data["id"] = _md5(data.get("name", "").lower())
        return data


class Relation(BaseModel):
    """A directed relationship between two entities extracted by Sarvam AI."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="ID of the source Entity")
    target_id: str = Field(description="ID of the target Entity")
    type: str = Field(description="Relation label, e.g. WORKS_AT, PART_OF, FOUNDED_BY")
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this relation, in the range [0.0, 1.0]",
    )


class ExtractionResult(BaseModel):
    """All entities and relations extracted from a single Chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="ID of the Chunk that was processed")
    entities: list[Entity] = Field(
        default_factory=list,
        description="Entities extracted from the chunk",
    )
    relations: list[Relation] = Field(
        default_factory=list,
        description="Relations between entities extracted from the chunk",
    )
