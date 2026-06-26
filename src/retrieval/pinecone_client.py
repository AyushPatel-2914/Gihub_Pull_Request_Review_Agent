import logging, os, time
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)
INDEX_NAME = "pr-review-agent"
UPSERT_BATCH_SIZE = 100
BATCH_DELAY = 0.5
TOP_K_DEFAULT = 10

def repo_to_namespace(repo_full_name):
    return repo_full_name.replace("/", "-").replace("_", "-").lower()

class PineconeClient:
    def __init__(self):
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY is not set.")
        self.pc = Pinecone(api_key=api_key)
        self.index = self._get_or_create_index()
        repo_full_name = os.environ.get("REPO_FULL_NAME", "unknown-repo")
        self.namespace = repo_to_namespace(repo_full_name)
        logger.info("PineconeClient ready | index=%s | namespace=%s", INDEX_NAME, self.namespace)

    def _get_or_create_index(self):
        existing = [idx.name for idx in self.pc.list_indexes()]
        if INDEX_NAME not in existing:
            logger.info("Creating index '%s'...", INDEX_NAME)
            self.pc.create_index(name=INDEX_NAME, dimension=768, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            time.sleep(5)
        return self.pc.Index(INDEX_NAME)

    def upsert_vectors(self, vectors, namespace=None):
        ns = namespace or self.namespace
        if not vectors:
            return 0
        total = 0
        batches = [vectors[i:i+UPSERT_BATCH_SIZE] for i in range(0, len(vectors), UPSERT_BATCH_SIZE)]
        for batch_num, batch in enumerate(batches, start=1):
            self.index.upsert(vectors=batch, namespace=ns)
            total += len(batch)
            if batch_num < len(batches):
                time.sleep(BATCH_DELAY)
        logger.info("Upsert complete | total_vectors=%d", total)
        return total

    def query(self, vector, top_k=TOP_K_DEFAULT, namespace=None, filter=None):
        ns = namespace or self.namespace
        kwargs = {"vector": vector, "top_k": top_k, "namespace": ns, "include_metadata": True}
        if filter:
            kwargs["filter"] = filter
        try:
            response = self.index.query(**kwargs)
            return response.get("matches", [])
        except Exception as e:
            logger.error("Pinecone query failed: %s", e)
            return []

    def delete_by_filepath(self, filepath, namespace=None):
        ns = namespace or self.namespace
        try:
            self.index.delete(filter={"filepath": {"$eq": filepath}}, namespace=ns)
        except Exception as e:
            logger.error("Failed to delete vectors for '%s': %s", filepath, e)

    def delete_namespace(self, namespace=None):
        ns = namespace or self.namespace
        try:
            self.index.delete(delete_all=True, namespace=ns)
        except Exception as e:
            logger.error("Failed to delete namespace '%s': %s", ns, e)

    def get_namespace_stats(self, namespace=None):
        ns = namespace or self.namespace
        try:
            stats = self.index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(ns, {})
            return {"namespace": ns, "vector_count": ns_stats.get("vector_count", 0)}
        except Exception as e:
            logger.error("Failed to get index stats: %s", e)
            return {"namespace": ns, "vector_count": -1}
