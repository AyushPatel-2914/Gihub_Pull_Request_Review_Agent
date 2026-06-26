import logging
from dataclasses import dataclass, field
from ingestion.embedder import Embedder
from retrieval.pinecone_client import PineconeClient
from retrieval.bm25_fallback import BM25Index, extract_symbols_from_diff

logger = logging.getLogger(__name__)
MAX_HEALING_ROUNDS = 3
TOP_K_INITIAL = 15
TOP_K_HEALING = 8
MIN_SIMILARITY_SCORE = 0.3
MAX_CONTEXT_CHUNKS = 20

@dataclass
class RetrievalResult:
    chunks: list = field(default_factory=list)
    healing_rounds: int = 0
    missing_symbols: list = field(default_factory=list)
    context_string: str = ""
    was_healed: bool = False

class SelfHealingRetriever:
    def __init__(self, embedder, pinecone_client):
        self.embedder = embedder
        self.pinecone = pinecone_client
        self.bm25 = BM25Index()

    def retrieve(self, diff, pr_metadata):
        result = RetrievalResult()
        referenced_symbols = extract_symbols_from_diff(diff)
        logger.info("Symbols extracted: %d", len(referenced_symbols))
        query = pr_metadata.get("title", "") + "\n\n" + "\n".join([l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")][:50])
        query_vector = self.embedder.embed_query(query)
        initial_matches = self.pinecone.query(vector=query_vector, top_k=TOP_K_INITIAL)
        retrieved_chunks = [m for m in initial_matches if m.get("score", 0) >= MIN_SIMILARITY_SCORE]
        seen_ids = {c["id"] for c in retrieved_chunks}
        healing_round = 0
        for round_num in range(1, MAX_HEALING_ROUNDS + 1):
            missing = [s for s in referenced_symbols if len(s) >= 3 and s not in "\n".join(c.get("metadata", {}).get("content", "") for c in retrieved_chunks)]
            if not missing:
                break
            logger.info("Healing round %d | missing: %s", round_num, missing[:5])
            healing_round = round_num
            result.was_healed = True
            healing_query = "definition of " + " ".join(missing[:10])
            healing_vector = self.embedder.embed_query(healing_query)
            for match in self.pinecone.query(vector=healing_vector, top_k=TOP_K_HEALING):
                if match["id"] not in seen_ids and match.get("score", 0) >= MIN_SIMILARITY_SCORE:
                    retrieved_chunks.append(match)
                    seen_ids.add(match["id"])
            self.bm25.build([c.get("metadata", {}) for c in retrieved_chunks])
            for symbol in missing:
                for bm25_hit in self.bm25.search(symbol, top_k=3):
                    synthetic_id = f"bm25::{bm25_hit.filepath}::{symbol}"
                    if synthetic_id not in seen_ids:
                        retrieved_chunks.append({"id": synthetic_id, "score": min(bm25_hit.score / 10, 1.0), "metadata": {"content": bm25_hit.content, "filepath": bm25_hit.filepath, "symbol_name": bm25_hit.symbol_name, "start_line": bm25_hit.start_line, "end_line": bm25_hit.end_line, "source": "bm25_fallback"}})
                        seen_ids.add(synthetic_id)
        result.missing_symbols = [s for s in referenced_symbols if len(s) >= 3 and s not in "\n".join(c.get("metadata", {}).get("content", "") for c in retrieved_chunks)]
        retrieved_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        retrieved_chunks = retrieved_chunks[:MAX_CONTEXT_CHUNKS]
        result.chunks = retrieved_chunks
        result.healing_rounds = healing_round
        result.context_string = self._format_context(retrieved_chunks)
        logger.info("RAG complete | chunks=%d | healing_rounds=%d", len(retrieved_chunks), healing_round)
        return result

    def _format_context(self, chunks):
        if not chunks:
            return ""
        parts = ["## Relevant Codebase Context (Retrieved via RAG)\n"]
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            filepath = meta.get("filepath", "unknown")
            symbol = meta.get("symbol_name", "")
            content = meta.get("content", "")
            score = chunk.get("score", 0)
            header = f"### [{i}] `{filepath}`"
            if symbol:
                header += f" — `{symbol}`"
            header += f" (score={score:.2f})"
            parts.append(header)
            parts.append(f"```\n{content}\n```\n")
        return "\n".join(parts)
