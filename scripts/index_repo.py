import sys
sys.path.insert(0, r'D:\Github_pr_review_agent_\Gihub_Pull_Request_Review_Agent\src')
import argparse, logging, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv()
from ingestion.file_parser import parse_file, detect_file_type, SKIP_DIRS
from ingestion.embedder import Embedder
from retrieval.pinecone_client import PineconeClient, repo_to_namespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("index_repo")

INDEXABLE_EXTENSIONS = {".py",".js",".ts",".jsx",".tsx",".java",".go",".rs",".cpp",".c",".h",".rb",".php",".cs",".swift",".kt",".md",".txt",".rst",".yaml",".yml",".json",".toml",".ini",".cfg",".sh",".bash",".zsh",".html",".css",".scss",".sql",".xml"}
SKIP_EXTENSIONS = {".png",".jpg",".jpeg",".gif",".svg",".ico",".pdf",".zip",".tar",".gz",".exe",".bin",".pyc",".pyo",".so",".dll",".class",".woff",".woff2",".ttf",".eot",".mp4",".mp3",".wav",".lock"}
MAX_FILE_SIZE = 500000

def discover_files(repo_path):
    root = Path(repo_path).resolve()
    discovered = []
    for path in root.rglob("*"):
        if path.is_dir(): continue
        if any(skip in path.parts for skip in SKIP_DIRS): continue
        if path.suffix.lower() in SKIP_EXTENSIONS: continue
        if path.suffix.lower() not in INDEXABLE_EXTENSIONS:
            if path.name.lower() not in {"dockerfile","makefile","rakefile"}: continue
        try:
            if path.stat().st_size > MAX_FILE_SIZE: continue
        except OSError: continue
        discovered.append(path)
    logger.info("Discovery complete | found=%d", len(discovered))
    return discovered

def index_repository(repo_path, repo_name, dry_run=False, clear_first=False):
    namespace = repo_to_namespace(repo_name)
    logger.info("="*60)
    logger.info("Repo: %s | Namespace: %s | Mode: %s", repo_name, namespace, "DRY RUN" if dry_run else "LIVE")
    logger.info("="*60)
    files = discover_files(repo_path)
    if not files:
        logger.warning("No indexable files found")
        return
    if dry_run:
        for f in sorted(files):
            logger.info("  %s", f.relative_to(Path(repo_path).resolve()))
        logger.info("Would index %d files. Run without --dry-run to upload.", len(files))
        return
    pinecone_client = PineconeClient()
    if clear_first:
        pinecone_client.delete_namespace(namespace)
        time.sleep(2)
    embedder = Embedder()
    all_chunks = []
    root = Path(repo_path).resolve()
    for file_path in files:
        try:
            rel_path = str(file_path.relative_to(root))
            content = file_path.read_text(encoding="utf-8", errors="replace")
            chunks = parse_file(filepath=rel_path, content=content, namespace=namespace)
            if chunks:
                all_chunks.extend(chunks)
        except Exception as e:
            logger.error("Failed: %s - %s", file_path, e)
    logger.info("Chunks created: %d", len(all_chunks))
    if not all_chunks:
        return
    vectors = embedder.embed_chunks(all_chunks)
    pinecone_client.upsert_vectors(vectors, namespace=namespace)
    time.sleep(3)
    stats = pinecone_client.get_namespace_stats(namespace)
    logger.info("Done! Vectors in Pinecone: %d", stats.get("vector_count", "?"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", default=".")
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not os.environ.get("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY not set")
        sys.exit(1)
    index_repository(args.repo_path, args.repo_name, args.dry_run, args.clear)

if __name__ == "__main__":
    main()

