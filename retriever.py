"""
retriever.py

Handles embedding generation and FAISS-based semantic search over the SHL catalog.
At startup, embeds all catalog items. At query time, returns top-k most relevant assessments.
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CATALOG_PATH = Path(__file__).parent / "catalog.json"
INDEX_PATH = Path(__file__).parent / "faiss_index.pkl"

# We use a lightweight but high-quality model that runs fast without GPU
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _build_rich_text(item: dict) -> str:
    """
    Concatenate all semantically meaningful fields into one string for embedding.
    More detail = better retrieval quality.
    """
    parts = [
        item["name"],
        item.get("description", ""),
        "Test type: " + _expand_test_type(item.get("test_type", "")),
        "Competencies: " + ", ".join(item.get("competencies", [])),
        "Suitable for roles: " + ", ".join(item.get("roles", [])),
        "Job levels: " + ", ".join(item.get("job_levels", [])),
    ]
    return ". ".join(p for p in parts if p.strip())


def _expand_test_type(code: str) -> str:
    mapping = {
        "A": "Ability and Aptitude (cognitive reasoning test)",
        "P": "Personality and Behaviour (personality questionnaire)",
        "K": "Knowledge and Skills (knowledge test)",
        "S": "Simulation (coding simulation or job simulation)",
        "B": "Behavioural (situational judgement or behavioural assessment)",
    }
    return mapping.get(code, code)


class SHLRetriever:
    def __init__(self):
        self.catalog: List[dict] = []
        self.model: SentenceTransformer = None
        self.index: faiss.Index = None
        self._load()

    def _load(self):
        # Load catalog
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        # Load or build FAISS index
        if INDEX_PATH.exists():
            with open(INDEX_PATH, "rb") as f:
                saved = pickle.load(f)
            self.model = SentenceTransformer(EMBED_MODEL_NAME)
            self.index = saved["index"]
            # Validate index matches catalog size
            if self.index.ntotal != len(self.catalog):
                self._build_index()
        else:
            self._build_index()

    def _build_index(self):
        """Build embeddings and FAISS index from scratch."""
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        texts = [_build_rich_text(item) for item in self.catalog]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim on normalized vecs
        self.index.add(embeddings.astype(np.float32))

        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"index": self.index}, f)

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Semantic search: returns top_k most relevant catalog items for the query.
        query: natural language description of what assessments are needed
        """
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = dict(self.catalog[idx])
                item["_score"] = float(score)
                results.append(item)
        return results

    def get_by_names(self, names: List[str]) -> List[dict]:
        """Retrieve specific items by name (for comparison queries)."""
        name_lower = {n.lower() for n in names}
        return [item for item in self.catalog if item["name"].lower() in name_lower]

    def get_all(self) -> List[dict]:
        return list(self.catalog)
# Singleton - initialized once at module import time
retriever = None
def get_retriever():
    global retriever
    if retriever is None:
        retriever = SHLRetriever()
    return retriever
