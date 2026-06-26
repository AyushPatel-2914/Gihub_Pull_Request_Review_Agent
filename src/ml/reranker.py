"""
src/ml/reranker.py
─────────────────────────────────────────────────────────────
Cross-Encoder Re-ranker — Second-Stage Relevance Scoring

THE PROBLEM IT SOLVES:
  Vector (bi-encoder) search is fast but approximate. It embeds
  the query and documents SEPARATELY, then compares. This misses
  subtle relevance signals.

  A cross-encoder takes the (query, document) PAIR together and
  scores their relevance directly. It's slower, so we don't use
  it for the initial search — but it's excellent for RE-RANKING
  the top candidates the vector search already found.

THE TWO-STAGE PIPELINE:
  Stage 1 (Phase 2): Bi-encoder retrieves top 15 candidates    (fast, wide net)
  Stage 2 (Phase 3): Cross-encoder re-scores those 15          (slow, precise)
                     → keeps only the genuinely relevant ones

  This saves LLM tokens (fewer, better chunks) AND improves
  review quality (the chunks that survive are truly relevant).

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~80MB, much smaller than the embedding model
  - Purpose-built for relevance re-ranking
  - Runs on CPU acceptably for ~15 candidates
─────────────────────────────────────────────────────────────
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Model & Threshold Config ───────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Chunks scoring below this after re-ranking are discarded.
# Cross-encoder scores are logits; ~0 is the rough relevance boundary.
RELEVANCE_THRESHOLD = -2.0
# Keep at most this many chunks after re-ranking (token budget control)
MAX_CHUNKS_AFTER_RERANK = 8


@dataclass
class RerankedChunk:
    """A chunk with its new cross-encoder relevance score."""
    chunk: dict          # original chunk dict (id, metadata, etc.)
    rerank_score: float  # cross-encoder relevance score
    original_score: float  # original bi-encoder/BM25 score


class CrossEncoderReranker:
    """
    Wraps a cross-encoder model for re-ranking RAG chunks.
    The model loads lazily on first use to avoid the cost when
    re-ranking is disabled or there are no chunks.
    """

    def __init__(self):
        self._model = None  # lazy-loaded

    def _ensure_model(self):
        """Loads the cross-encoder model on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder model: %s", RERANKER_MODEL)
            self._model = CrossEncoder(RERANKER_MODEL)
            logger.info("Cross-encoder model loaded")
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        threshold: float = RELEVANCE_THRESHOLD,
        max_keep: int = MAX_CHUNKS_AFTER_RERANK,
    ) -> list[dict]:
        """
        Re-ranks chunks by true relevance to the query.

        Args:
          query     : the query text (typically the PR diff summary)
          chunks    : list of chunk dicts from the RAG retriever.
                      Each must have chunk["metadata"]["content"].
          threshold : discard chunks scoring below this
          max_keep  : keep at most this many top chunks

        Returns:
          list of chunk dicts, re-ordered by relevance, low-scoring
          ones removed. Each surviving chunk gets a "rerank_score"
          field added to its metadata.
        """
        if not chunks:
            return []

        # Build (query, document) pairs for the cross-encoder
        pairs = []
        for chunk in chunks:
            content = chunk.get("metadata", {}).get("content", "")
            pairs.append([query, content])

        model = self._ensure_model()

        logger.info("Re-ranking %d chunks with cross-encoder...", len(chunks))
        scores = model.predict(pairs)

        # Pair chunks with their new scores
        scored = []
        for chunk, score in zip(chunks, scores):
            scored.append(RerankedChunk(
                chunk=chunk,
                rerank_score=float(score),
                original_score=chunk.get("score", 0.0),
            ))

        # Sort by rerank score descending
        scored.sort(key=lambda x: x.rerank_score, reverse=True)

        # Filter by threshold and cap count
        survivors = [
            s for s in scored if s.rerank_score >= threshold
        ][:max_keep]

        # If threshold killed everything, keep the top few anyway —
        # better to give the LLM something than nothing
        if not survivors and scored:
            logger.warning(
                "All chunks below rerank threshold — keeping top %d anyway",
                min(3, len(scored)),
            )
            survivors = scored[:3]

        # Annotate surviving chunks with their rerank score and return
        result = []
        for s in survivors:
            chunk = s.chunk
            chunk.setdefault("metadata", {})["rerank_score"] = round(s.rerank_score, 3)
            # Update the visible score to the rerank score for display
            chunk["rerank_score"] = round(s.rerank_score, 3)
            result.append(chunk)

        logger.info(
            "Re-ranking complete | input=%d | kept=%d | discarded=%d",
            len(chunks), len(result), len(chunks) - len(result),
        )

        return result
