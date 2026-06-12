"""Crawl web URLs using Crawl4AI async crawler and return Document instances."""

from __future__ import annotations

import asyncio
import logging

from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv

from graphrag.models import Document

load_dotenv()

logger = logging.getLogger(__name__)


async def crawl_url(url: str) -> Document | None:
    """Crawl a single URL and return a Document instance with clean markdown text.

    Returns None if the page could not be fetched or produced no content.
    """
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

        if not result.success:
            logger.error("Failed to crawl %s: %s", url, result.error_message)
            return None

        text = result.markdown.strip() if result.markdown else ""
        if not text:
            logger.warning("No content extracted from %s", url)
            return None

        title = result.metadata.get("title") or url
        return Document(title=title, source=url, text=text, pages=0)

    except Exception as exc:
        logger.error("Crawler error for %s: %s", url, exc)
        return None


async def crawl_urls(urls: list[str]) -> list[Document]:
    """Crawl multiple URLs concurrently and return a list of Document instances.

    URLs that fail are skipped — errors are logged but do not abort the batch.
    """
    tasks = [crawl_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    docs = [doc for doc in results if doc is not None]
    logger.info("Crawled %d/%d URLs successfully", len(docs), len(urls))
    return docs


def crawl_urls_sync(urls: list[str]) -> list[Document]:
    """Synchronous wrapper around crawl_urls for use outside async contexts."""
    return asyncio.run(crawl_urls(urls))
