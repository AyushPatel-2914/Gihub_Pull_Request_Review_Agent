import logging
from dataclasses import dataclass, field
from ingestion.embedder import Embedder
from retrieval.pinecone_client import PineconeClient
from retrieval.bm25_fallback import BM25Index, extract_symbols_from_diff

logger = logging.getLogger(__name__)

MAX_HEALING_ROUNDS = 3
TOP_K_INITIAL = 15
TOP_K_HEALING = 8
MIN_SIMILARITY_SCORE = 0.15
MAX_CONTEXT_CHUNKS = 20
MAX_CRITIQUE_ROUNDS = 1


@dataclass
class RetrievalResult:
    chunks: list = field(default_factory=list)
    healing_rounds: int = 0
    missing_symbols: list = field(default_factory=list)
    context_string: str = ""
    was_healed: bool = False
    critique_ran: bool = False
    critique_sufficient: bool = True
    critique_reasoning: str = ""


class SelfHealingRetriever:
    def __init__(self, embedder, pinecone_client, critic=None):
        self.embedder = embedder
        self.pinecone = pinecone_client
        self.bm25 = BM25Index()
        self.critic = critic

    def retrieve(self, diff, pr_metadata):
        result = RetrievalResult()
        referenced_symbols = extract_symbols_from_diff(diff)
        logger.info("Symbols extracted: %d", len(referenced_symbols))

        query = pr_metadata.get("title", "") + "\n\n" + "\n".join(
            l[1:] for l in diff.splitlines()
            if l.startswith("+") and not l.startswith("+++")
        )[:2000]
        query_vector = self.embedder.embed_query(query)
        initial_matches = self.pinecone.query(vector=query_vector, top_k=TOP_K_INITIAL)
        retrieved_chunks = [m for m in initial_matches if m.get("score", 0) >= MIN_SIMILARITY_SCORE]
        seen_ids = {c["id"] for c in retrieved_chunks}
        logger.info("Initial retrieval | candidates=%d | after_filter=%d",
                    len(initial_matches), len(retrieved_chunks))

        healing_round = 0
        for round_num in range(1, MAX_HEALING_ROUNDS + 1):
            missing = self._find_missing_symbols(referenced_symbols, retrieved_chunks)
            if not missing:
                break
            logger.info("Healing round %d | missing: %s", round_num, missing[:5])
            healing_round = round_num
            result.was_healed = True
            self._heal(missing, diff, retrieved_chunks, seen_ids)

        if self.critic is not None and retrieved_chunks:
            logger.info("Running LLM completeness critique (Phase 4)...")
            critique = self.critic.critique(diff, retrieved_chunks)
            result.critique_ran = True
            result.critique_sufficient = critique["sufficient"]
            result.critique_reasoning = critique["reasoning"]
            if not critique["sufficient"] and critique["missing"]:
                logger.info("Critic flagged INSUFFICIENT | missing: %s", critique["missing"][:5])
                result.was_healed = True
                healing_round += 1
                self._heal(critique["missing"], diff, retrieved_chunks, seen_ids)
            else:
                logger.info("Critic judged context SUFFICIENT")

        result.missing_symbols = self._find_missing_symbols(referenced_symbols, retrieved_chunks)
        retrieved_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        retrieved_chunks = retrieved_chunks[:MAX_CONTEXT_CHUNKS]
        result.chunks = retrieved_chunks
        result.healing_rounds = healing_round
        result.context_string = self._format_context(retrieved_chunks)
        logger.info("RAG complete | chunks=%d | healing_rounds=%d | critique_ran=%s",
                    len(retrieved_chunks), healing_round, result.critique_ran)
        return result

    def _heal(self, missing_symbols, diff, retrieved_chunks, seen_ids):
        healing_query = "definition of " + " ".join(missing_symbols[:10]) + "\n\n" + diff[:500]
        healing_vector = self.embedder.embed_query(healing_query)
        healing_matches = self.pinecone.query(vector=healing_vector, top_k=TOP_K_HEALING)
        for match in healing_matches:
            if match["id"] not in seen_ids and match.get("score", 0) >= MIN_SIMILARITY_SCORE:
                retrieved_chunks.append(match)
                seen_ids.add(match["id"])
        self.bm25.build([c.get("metadata", {}) for c in retrieved_chunks])
        for symbol in missing_symbols:
            for bm25_hit in self.bm25.search(symbol, top_k=3):
                synthetic_id = f"bm25::{bm25_hit.filepath}::{symbol}"
                if synthetic_id not in seen_ids:
                    retrieved_chunks.append({
                        "id": synthetic_id,
                        "score": min(bm25_hit.score / 10, 1.0),
                        "metadata": {
                            "content": bm25_hit.content,
                            "filepath": bm25_hit.filepath,
                            "symbol_name": bm25_hit.symbol_name,
                            "start_line": bm25_hit.start_line,
                            "end_line": bm25_hit.end_line,
                            "source": "bm25_fallback",
                        }
                    })
                    seen_ids.add(synthetic_id)

    def _find_missing_symbols(self, referenced_symbols, retrieved_chunks):
        if not referenced_symbols:
            return []
        all_content = "\n".join(
            chunk.get("metadata", {}).get("content", "") for chunk in retrieved_chunks
        )
        return [s for s in referenced_symbols if len(s) >= 3 and s not in all_content]

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
            source = meta.get("source", "semantic")
            header = f"### [{i}] `{filepath}`"
            if symbol:
                header += f" — `{symbol}`"
            header += f" (score={score:.2f}"
            if source == "bm25_fallback":
                header += ", via BM25"
            header += ")"
            parts.append(header)
            parts.append(f"```\n{content}\n```\n")
        return "\n".join(parts)
