"""
src/agent.py
─────────────────────────────────────────────────────────────
PR Review Agent — GitHub Actions Entry Point

This is called by GitHub Actions: `python src/agent.py`

Phase 5B refactor:
  The actual review pipeline now lives in agent_core.run_review()
  so it can be shared with the FastAPI webhook server. This file
  just reads the GitHub Actions environment variables and calls it.
─────────────────────────────────────────────────────────────
"""

import os
import sys
import logging

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_core import run_review

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent")


def validate_environment():
    required = {
        "GITHUB_TOKEN": "Auto-injected by GitHub Actions",
        "GROQ_API_KEY": "Get free key at https://console.groq.com",
        "PINECONE_API_KEY": "Get free key at https://app.pinecone.io",
        "PR_NUMBER": "Auto-injected by GitHub Actions",
        "REPO_FULL_NAME": "Auto-injected by GitHub Actions",
    }
    missing = [f"  x {var}: {hint}" for var, hint in required.items()
               if not os.environ.get(var)]
    if missing:
        logger.error("Missing environment variables:\n%s", "\n".join(missing))
        sys.exit(1)
    logger.info("Environment validation passed")


if __name__ == "__main__":
    try:
        validate_environment()
        repo = os.environ["REPO_FULL_NAME"]
        pr_number = int(os.environ["PR_NUMBER"])
        result = run_review(repo, pr_number)
        logger.info("=" * 60)
        logger.info("Agent complete | %s", result)
        logger.info("=" * 60)
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        sys.exit(1)
