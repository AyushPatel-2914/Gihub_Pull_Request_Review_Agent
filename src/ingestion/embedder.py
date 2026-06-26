import logging, time
from sentence_transformers import SentenceTransformer
from ingestion.file_parser import CodeChunk

logger = logging.getLogger(__name__)
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIMENSIONS = 768
UPSERT_BATCH_SIZE = 100
BATCH_DELAY_SECONDS = 0.5
SEARCH_DOCUMENT_PREFIX = "search_document: "
SEARCH_QUERY_PREFIX = "search_query: "

class Embedder:
    def __init__(self):
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        logger.info("Embedding model loaded | dimensions=%d", EMBEDDING_DIMENSIONS)

    def embed_chunks(self, chunks):
        if not chunks:
            return []
        texts = [SEARCH_DOCUMENT_PREFIX + chunk.content for chunk in chunks]
        logger.info("Embedding %d chunks...", len(chunks))
        start = time.time()
        vectors = self.model.encode(texts, batch_size=32, show_progress_bar=len(chunks) > 50, normalize_embeddings=True, convert_to_numpy=True)
        logger.info("Embedding complete | chunks=%d | elapsed=%.2fs", len(chunks), time.time() - start)
        return [(chunk.chunk_id, vector.tolist(), chunk.to_pinecone_metadata()) for chunk, vector in zip(chunks, vectors)]

    def embed_query(self, query):
        vector = self.model.encode(SEARCH_QUERY_PREFIX + query, normalize_embeddings=True, convert_to_numpy=True)
        return vector.tolist()

    def embed_texts(self, texts):
        prefixed = [SEARCH_DOCUMENT_PREFIX + t for t in texts]
        vectors = self.model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
