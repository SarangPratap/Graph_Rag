"""Crawl web URLs using Crawl4AI async crawler and return clean text document dicts."""

from __future__ import annotations

import logging
from typing import Any

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


async def crawl_url(url: str) -> dict[str, Any]:
    """Crawl a single URL and return a document dict with clean text and metadata."""
    pass


async def crawl_urls(urls: list[str]) -> list[dict[str, Any]]:
    """Crawl multiple URLs concurrently and return a list of document dicts."""
    pass
