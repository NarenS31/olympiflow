"""City knowledge base + retrieval for the LLM advisory layer (Phase 4).

WHY this exists: the mathematical explanation tells us WHICH sensors/regions
drive a predicted slowdown, but not what a planner can DO about it. The city
knowledge base holds that operational context (corridors, bottlenecks, signal
capability, transit alternatives). We retrieve only the chunks relevant to the
current explanation and inject them as the prompt's "CITY CONTEXT" block.

This productionizes the keyword-retrieval pattern from the OlympiFlow RAG
advisor (backend/app/services/rag_advisor.py): score each chunk by how many
query terms it contains, take the top-k. Simple, offline, deterministic — no
embeddings, no external services (CLAUDE.md: local only, reproducible).

Python 3.9 compatible (typing.Dict/List/Optional, no `X | Y`).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from ...utils.io_utils import PKG_ROOT

_KB_DIR = os.path.join(PKG_ROOT, "models", "advisor", "kb")

# Words too common to carry retrieval signal — dropped from the query so a chunk
# isn't ranked highly just for containing "the" or "sensor".
_STOPWORDS = {
    "the", "and", "for", "sensor", "node", "near", "from", "with", "into",
    "los", "angeles", "mph", "min", "minutes", "lat", "lon",
}


def _tokens(text: str) -> List[str]:
    """Lowercase word tokens of length >= 3 (matches the OlympiFlow scorer)."""
    return [t for t in re.findall(r"[a-z0-9\-]+", text.lower())
            if len(t) >= 3 and t not in _STOPWORDS]


class KnowledgeBase:
    """Loads one city's KB JSON and retrieves the chunks relevant to an
    explanation. One instance per city; cheap to construct."""

    def __init__(self, city: str, kb_filename: str):
        self.city = city
        path = os.path.join(_KB_DIR, kb_filename)
        with open(path, "r") as f:
            data = json.load(f)
        self.display_name: str = data.get("display_name", city)
        self.chunks: List[Dict[str, Any]] = data["chunks"]

    # --- building the retrieval query from an explanation -------------------
    @staticmethod
    def query_from_explanation(exp: Dict[str, Any]) -> str:
        """Concatenate the human-readable names driving this prediction into one
        query string. We use the TARGET node name plus every top-node name,
        because those region labels (e.g. "Glendale / Burbank") are exactly the
        region_tags the KB chunks are keyed on."""
        parts: List[str] = [exp["prediction"]["node_name"]]
        for n in exp.get("top_nodes", []):
            parts.append(n["node_name"])
        return " ".join(parts)

    # --- scoring -----------------------------------------------------------
    def _score(self, query_terms: List[str], chunk: Dict[str, Any]) -> float:
        """Keyword relevance of a chunk to the query.

        region_tags are weighted higher than body text: a tag is a curated,
        deliberate hook ("this chunk is about Glendale/Burbank"), so a tag hit is
        stronger evidence than the same word appearing in prose. We also give a
        multi-word tag credit when the whole tag phrase appears in the query
        (e.g. "glendale / burbank" -> both "glendale" and "burbank" present)."""
        body = " ".join(_tokens(chunk["text"]))
        tag_terms = set()
        for tag in chunk.get("region_tags", []):
            tag_terms.update(_tokens(tag))

        score = 0.0
        seen = set()
        for t in query_terms:
            if t in seen:
                continue  # count each distinct query term once (avoid double-weighting repeated regions)
            seen.add(t)
            if t in tag_terms:
                score += 2.0            # tag hit: strong, curated signal
            elif t in body:
                score += 1.0            # body hit: weaker
        return score

    def retrieve(self, exp: Dict[str, Any], top_k: int = 4,
                 pad_to_k: bool = False) -> List[Dict[str, Any]]:
        """Return the top_k most relevant chunks for this explanation, highest
        first. Ties break by original KB order (stable sort) for determinism.

        pad_to_k (Phase 15b, for the C_RICH condition): when True, after the
        signal-bearing chunks are taken we KEEP GOING down the ranked list and
        include zero-scoring chunks too, until we reach top_k. That turns a
        normal (relevant-only) retrieval into a deliberately FULLER context block
        — the most relevant corridors first, then the rest of the KB (capacities,
        signal-timing, transit, incident history). The default pad_to_k=False path
        is byte-identical to the Phase-4 behaviour, so conditions A/B/C are
        untouched."""
        query_terms = _tokens(self.query_from_explanation(exp))
        scored = [(self._score(query_terms, c), i, c)
                  for i, c in enumerate(self.chunks)]
        # Sort by score desc, then original index asc (stable, reproducible).
        scored.sort(key=lambda x: (-x[0], x[1]))
        # Default: keep only chunks with any signal. Rich mode: include zero-
        # scoring chunks too so the context is genuinely fuller, not just re-ranked.
        relevant: List[Dict[str, Any]] = []
        for s, i, c in scored:
            if s > 0 or pad_to_k:
                relevant.append(c)
            if len(relevant) >= top_k:
                break
        # If everything scored 0 (unlikely), fall back to the first top_k so the
        # advisor still gets some context.
        if not relevant:
            relevant = self.chunks[:top_k]
        return relevant


def load_kb_for_city(city: str, advisor_cfg: Dict[str, Any]) -> KnowledgeBase:
    """Resolve a city name to its KB file via configs/advisor.yaml `cities:`."""
    cities: Dict[str, str] = advisor_cfg.get("cities", {})
    if city not in cities:
        raise KeyError(
            "No knowledge base registered for city '{}'. Add it under `cities:` "
            "in configs/advisor.yaml and drop a {}.json in models/advisor/kb/."
            .format(city, city))
    return KnowledgeBase(city, cities[city])


def render_kb_block(chunks: List[Dict[str, Any]], max_chars: int) -> str:
    """Render retrieved chunks as the prompt's CITY CONTEXT block, truncated to
    a character budget so the prompt stays small."""
    parts: List[str] = []
    budget = max_chars
    for c in chunks:
        snippet = c["text"][:max(0, budget)]
        parts.append("[{}]\n{}".format(c["title"], snippet))
        budget -= len(snippet)
        if budget <= 0:
            break
    return "\n\n---\n\n".join(parts)
