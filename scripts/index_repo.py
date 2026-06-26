"""
scripts/index_repo.py
─────────────────────────────────────────────────────────────
One-Time Repository Indexer — Pinecone Seeding Script

PURPOSE:
  This script is run ONCE manually to seed the Pinecone
  vector database with the entire repository codebase.

  After this initial seed, the agent's Phase 2 logic handles
  incremental updates: deleting old vectors and upserting new
  ones for only the files changed in each PR.

USAGE:
  # From the repo root:
  python scripts/index_repo.py --repo-path . --namespace my-repo-name

PHASE STATUS:
  This script is a STUB for Phase 1.
  The actual embedding and Pinecone upsert logic is
  implemented in Phase 2 (src/ingestion/ and src/retrieval/).

  For now, running this script will:
    1. Walk the repo and discover all files
    2. Print a manifest of what would be indexed
    3. Exit without uploading anything

  Phase 2 will fill in the real implementation.
─────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Load .env for local development
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("index_repo")


# ──────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ──────────────────────────────────────────────────────────────

# File extensions we will index
# Binary files, images, compiled artifacts are skipped
INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".cpp", ".c", ".h",
    ".rb", ".php", ".cs", ".swift", ".kt",
    ".md", ".txt", ".rst",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh",
    ".html", ".css", ".scss",
    ".sql",
    ".ipynb",
}

# Directories to always skip
SKIP_DIRS = {
    ".git", ".github", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".env",
    "dist", "build", ".next", ".nuxt",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "migrations",
}


def discover_files(repo_path: str) -> list[Path]:
    """
    Walks the repository and returns a list of all
    indexable file paths, respecting skip rules.

    Args:
      repo_path: root directory of the repository

    Returns:
      list of Path objects for indexable files
    """
    root = Path(repo_path).resolve()
    discovered = []

    for path in root.rglob("*"):
        # Skip directories themselves
        if path.is_dir():
            continue

        # Skip files inside blacklisted directories
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue

        # Skip files whose extension is not in our indexable set
        if path.suffix.lower() not in INDEXABLE_EXTENSIONS:
            continue

        # Skip very large files (> 500KB) — likely generated/minified
        try:
            if path.stat().st_size > 500_000:
                logger.debug("Skipping large file: %s", path)
                continue
        except OSError:
            continue

        discovered.append(path)

    return discovered


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seed Pinecone with the full repository codebase."
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to the repository root (default: current directory)",
    )
    parser.add_argument(
        "--namespace",
        required=True,
        help="Pinecone namespace for this repo (e.g. 'my-org-backend-api')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="[Phase 1 default] Print manifest without uploading to Pinecone",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Repository Indexer — Phase 1 Stub")
    logger.info("=" * 60)
    logger.info("Repo path  : %s", args.repo_path)
    logger.info("Namespace  : %s", args.namespace)
    logger.info("Mode       : %s", "DRY RUN" if args.dry_run else "LIVE UPLOAD")
    logger.info("-" * 60)

    # ── Discover files ─────────────────────────────────────────
    logger.info("Discovering indexable files...")
    files = discover_files(args.repo_path)
    logger.info("Found %d indexable files", len(files))

    # ── Print manifest ─────────────────────────────────────────
    if args.dry_run:
        logger.info("\nFile manifest (dry run — nothing uploaded):")
        for f in sorted(files):
            rel = f.relative_to(Path(args.repo_path).resolve())
            size = f.stat().st_size
            logger.info("  [%6d bytes] %s", size, rel)

        logger.info("-" * 60)
        logger.info(
            "DRY RUN COMPLETE: %d files would be indexed into "
            "Pinecone namespace '%s'",
            len(files),
            args.namespace,
        )
        logger.info(
            "Phase 2 will implement: chunking → embedding → Pinecone upsert"
        )

    else:
        # This block will be filled in Phase 2
        logger.error(
            "Live upload not yet implemented. "
            "This will be built in Phase 2. "
            "Run with --dry-run for now."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()