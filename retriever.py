"""
retriever.py

Lightweight keyword-based retrieval for SHL catalog.
Optimized for low-memory deployment environments like Render free tier.
"""

import json
from pathlib import Path
from typing import List

CATALOG_PATH = Path(__file__).parent / "catalog.json"


def _build_rich_text(item: dict) -> str:
    """
    Combine important catalog fields into searchable text.
    """
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("competencies", [])),
        " ".join(item.get("roles", [])),
        " ".join(item.get("job_levels", [])),
        item.get("test_type", ""),
    ]

    return " ".join(parts).lower()


class SHLRetriever:
    def __init__(self):
        self.catalog: List[dict] = []
        self.search_texts: List[str] = []
        self._load()

    def _load(self):
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.search_texts = [
            _build_rich_text(item)
            for item in self.catalog
        ]

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Lightweight keyword-based retrieval.
        """
        query = query.lower()
        query_words = query.split()

        scored_results = []

        for item, searchable_text in zip(self.catalog, self.search_texts):
            score = 0

            for word in query_words:
                if word in searchable_text:
                    score += 1

            scored_results.append((score, item))

        scored_results.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            item
            for score, item in scored_results[:top_k]
        ]

    def get_by_names(self, names: List[str]) -> List[dict]:
        name_lower = {n.lower() for n in names}

        return [
            item
            for item in self.catalog
            if item["name"].lower() in name_lower
        ]

    def get_all(self) -> List[dict]:
        return list(self.catalog)


retriever = None


def get_retriever():
    global retriever

    if retriever is None:
        retriever = SHLRetriever()

    return retriever
