"""Test the web crawler against a real URL."""

from __future__ import annotations

from graphrag.ingestion.web_crawler import crawl_urls_sync


def main() -> None:
    urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://this-url-does-not-exist-xyz.com",  # failure case
    ]

    print(f"Crawling {len(urls)} URLs...\n")
    docs = crawl_urls_sync(urls)

    print(f"Successfully crawled: {len(docs)}/{len(urls)}\n")
    for doc in docs:
        print(f"  Title  : {doc.title}")
        print(f"  Source : {doc.source}")
        print(f"  Length : {len(doc.text)} chars")
        print(f"  ID     : {doc.id[:8]}...")
        print(f"  Preview: {doc.text[:120].strip()}")
        print()


if __name__ == "__main__":
    main()
