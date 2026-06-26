import os

files = {}

files['src/ingestion/file_parser.py'] = '''
import ast, json, logging, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
MAX_CHUNK_CHARS = 6000
OVERLAP_CHARS = 200

SKIP_DIRS = {".git", ".github", "__pycache__", "node_modules", ".venv", "venv", "env", "dist", "build", ".next", ".nuxt", ".pytest_cache", ".mypy_cache", "migrations", ".ruff_cache"}
SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip", ".tar", ".gz", ".exe", ".bin", ".pyc", ".pyo", ".so", ".dll", ".class", ".lock"}

@dataclass
class CodeChunk:
    chunk_id: str
    content: str
    filepath: str
    file_type: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_pinecone_metadata(self):
        return {"filepath": self.filepath, "file_type": self.file_type, "start_line": self.start_line, "end_line": self.end_line, "symbol_name": self.symbol_name or "", "content": self.content[:2000], **self.metadata}

def detect_file_type(filepath):
    suffix = Path(filepath).suffix.lower()
    mapping = {".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby", ".cpp": "cpp", ".c": "c", ".h": "c", ".cs": "csharp", ".php": "php", ".swift": "swift", ".kt": "kotlin", ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".html": "html", ".htm": "html", ".css": "css", ".scss": "css", ".sql": "sql", ".ipynb": "notebook", ".xml": "xml"}
    if Path(filepath).name.lower() == "dockerfile":
        return "dockerfile"
    return mapping.get(suffix, "unknown")

def _chunk_sliding_window(content, filepath, file_type, namespace):
    chunks = []
    lines = content.splitlines()
    total_chars = 0
    chunk_start_line = 1
    current_lines = []
    chunk_idx = 0
    for line_num, line in enumerate(lines, start=1):
        current_lines.append(line)
        total_chars += len(line) + 1
        if total_chars >= MAX_CHUNK_CHARS:
            chunk_content = "\\n".join(current_lines)
            chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{chunk_idx}", content=chunk_content, filepath=filepath, file_type=file_type, start_line=chunk_start_line, end_line=line_num, metadata={"namespace": namespace}))
            chunk_idx += 1
            overlap_lines = []
            overlap_chars = 0
            for ol in reversed(current_lines):
                if overlap_chars + len(ol) > OVERLAP_CHARS: break
                overlap_lines.insert(0, ol)
                overlap_chars += len(ol) + 1
            current_lines = overlap_lines
            chunk_start_line = line_num - len(overlap_lines) + 1
            total_chars = overlap_chars
    if current_lines:
        chunk_content = "\\n".join(current_lines)
        if chunk_content.strip():
            chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{chunk_idx}", content=chunk_content, filepath=filepath, file_type=file_type, start_line=chunk_start_line, end_line=len(lines), metadata={"namespace": namespace}))
    return chunks

def _chunk_python(content, filepath, namespace):
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_sliding_window(content, filepath, "python", namespace)
    lines = content.splitlines()
    chunks = []
    top_level_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(n, "col_offset", 0) == 0]
    top_level_nodes.sort(key=lambda n: n.lineno)
    if not top_level_nodes:
        return _chunk_sliding_window(content, filepath, "python", namespace)
    for idx, node in enumerate(top_level_nodes):
        start = node.lineno - 1
        end = node.end_lineno
        chunk_content = "\\n".join(lines[start:end])
        chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{idx}", content=chunk_content, filepath=filepath, file_type="python", start_line=start+1, end_line=end, symbol_name=node.name, metadata={"namespace": namespace}))
    return chunks

def _chunk_markdown(content, filepath, namespace):
    lines = content.splitlines()
    chunks = []
    current_section_lines = []
    current_start = 1
    section_idx = 0
    for i, line in enumerate(lines, start=1):
        if (line.startswith("## ") or line.startswith("# ")) and current_section_lines:
            section_content = "\\n".join(current_section_lines)
            if section_content.strip():
                chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{section_idx}", content=section_content, filepath=filepath, file_type="markdown", start_line=current_start, end_line=i-1, symbol_name=current_section_lines[0].lstrip("#").strip() if current_section_lines else None, metadata={"namespace": namespace}))
                section_idx += 1
            current_section_lines = [line]
            current_start = i
        else:
            current_section_lines.append(line)
    if current_section_lines:
        section_content = "\\n".join(current_section_lines)
        if section_content.strip():
            chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{section_idx}", content=section_content, filepath=filepath, file_type="markdown", start_line=current_start, end_line=len(lines), symbol_name=current_section_lines[0].lstrip("#").strip() if current_section_lines else None, metadata={"namespace": namespace}))
    return chunks if chunks else _chunk_sliding_window(content, filepath, "markdown", namespace)

def _chunk_json(content, filepath, namespace):
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _chunk_sliding_window(content, filepath, "json", namespace)
    if not isinstance(parsed, dict):
        return _chunk_sliding_window(content, filepath, "json", namespace)
    chunks = []
    for idx, (key, value) in enumerate(parsed.items()):
        chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{idx}", content=json.dumps({key: value}, indent=2), filepath=filepath, file_type="json", start_line=1, end_line=1, symbol_name=key, metadata={"namespace": namespace}))
    return chunks if chunks else _chunk_sliding_window(content, filepath, "json", namespace)

def parse_file(filepath, content, namespace):
    if not content or not content.strip():
        return []
    file_type = detect_file_type(filepath)
    if Path(filepath).suffix.lower() in SKIP_EXTENSIONS:
        return []
    if file_type == "python":
        return _chunk_python(content, filepath, namespace)
    elif file_type in ("markdown", "rst", "text"):
        return _chunk_markdown(content, filepath, namespace)
    elif file_type == "json":
        return _chunk_json(content, filepath, namespace)
    else:
        return _chunk_sliding_window(content, filepath, file_type, namespace)
'''

files['src/ingestion/embedder.py'] = '''
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
'''

files['src/retrieval/pinecone_client.py'] = '''
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
'''

files['src/retrieval/bm25_fallback.py'] = '''
import logging, re
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

@dataclass
class BM25Result:
    content: str
    filepath: str
    score: float
    symbol_name: str = ""
    start_line: int = 0
    end_line: int = 0

class BM25Index:
    def __init__(self):
        self.corpus = []
        self.tokenized = []
        self.bm25 = None
        self._built = False

    def build(self, chunks):
        if not chunks:
            return
        self.corpus = chunks
        self.tokenized = [self._tokenize(chunk.get("content", "")) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        self._built = True

    def search(self, query, top_k=5):
        if not self._built:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.corpus[idx]
            results.append(BM25Result(content=chunk.get("content", ""), filepath=chunk.get("filepath", ""), score=float(scores[idx]), symbol_name=chunk.get("symbol_name", ""), start_line=chunk.get("start_line", 0), end_line=chunk.get("end_line", 0)))
        return results

    def _tokenize(self, text):
        if not text:
            return []
        text = re.sub(r"([a-z])([A-Z])", r"\\1 \\2", text)
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if len(t) > 1]

def extract_symbols_from_diff(diff):
    symbols = set()
    added_lines = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    added_content = "\\n".join(added_lines)
    symbols.update(re.findall(r"(?:def|class)\\s+([a-zA-Z_][a-zA-Z0-9_]*)", added_content))
    symbols.update(re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\(", added_content))
    symbols.update(re.findall(r"(?:import|from)\\s+([a-zA-Z_][a-zA-Z0-9_.]*)", added_content))
    symbols.update(re.findall(r"\\b([A-Z][A-Z0-9_]{2,})\\b", added_content))
    stop_words = {"if","else","elif","for","while","try","except","finally","with","as","return","yield","lambda","and","or","not","in","is","True","False","None","self","cls","def","class","import","from","pass","break","continue","raise","del","global","nonlocal","assert","async","await","print","len","range","str","int","float","list","dict","set","tuple","type"}
    return [s for s in symbols if s.lower() not in stop_words]
'''

files['src/rag/retriever.py'] = '''
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
        query = pr_metadata.get("title", "") + "\\n\\n" + "\\n".join([l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")][:50])
        query_vector = self.embedder.embed_query(query)
        initial_matches = self.pinecone.query(vector=query_vector, top_k=TOP_K_INITIAL)
        retrieved_chunks = [m for m in initial_matches if m.get("score", 0) >= MIN_SIMILARITY_SCORE]
        seen_ids = {c["id"] for c in retrieved_chunks}
        healing_round = 0
        for round_num in range(1, MAX_HEALING_ROUNDS + 1):
            missing = [s for s in referenced_symbols if len(s) >= 3 and s not in "\\n".join(c.get("metadata", {}).get("content", "") for c in retrieved_chunks)]
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
        result.missing_symbols = [s for s in referenced_symbols if len(s) >= 3 and s not in "\\n".join(c.get("metadata", {}).get("content", "") for c in retrieved_chunks)]
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
        parts = ["## Relevant Codebase Context (Retrieved via RAG)\\n"]
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
            parts.append(f"```\\n{content}\\n```\\n")
        return "\\n".join(parts)
'''

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.lstrip('\n'))
    print(f"Written: {filepath}")

print("\nAll files written successfully!")