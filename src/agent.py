"""
src/agent.py
─────────────────────────────────────────────────────────────
PR Review Agent — Main Orchestrator
This is the entry point called by GitHub Actions: `python src/agent.py`

Phase 1 execution flow:
  1. Load environment variables
  2. Initialize GitHub client → fetch PR metadata, diff, changed files
  3. Fetch file contents for each changed file
  4. Run static analysis across all changed files
  5. Send diff + static findings to Llama 3 via Groq
  6. Post the markdown review back to the GitHub PR thread

Each phase will extend this file by adding steps between
the existing ones — the flow grows downward naturally.
─────────────────────────────────────────────────────────────
"""

import os
import sys
import logging

# Load .env for local development — no-op in GitHub Actions
# (environment variables are already set by the workflow)
from dotenv import load_dotenv
load_dotenv()

# ── Local module imports ───────────────────────────────────────
from github.gh_client import GitHubClient
from analysis.static_analyzer import analyze_changed_files
from llm.groq_client import GroqLLMClient


# ──────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────
# Format shows timestamp, level, module name, and message.
# In GitHub Actions, this output appears in the workflow run logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agent")


# ──────────────────────────────────────────────────────────────
# HELPER: VALIDATE REQUIRED ENVIRONMENT VARIABLES
# ──────────────────────────────────────────────────────────────

def validate_environment() -> None:
    """
    Checks that all required environment variables are present
    before the agent does any work.

    Raises:
      SystemExit: if any required variable is missing,
                  with a clear message about which one is absent
                  and how to fix it.
    """
    required = {
        "GITHUB_TOKEN": "Auto-injected by GitHub Actions. For local dev, add to .env",
        "GROQ_API_KEY": "Get a free key at https://console.groq.com — add to GitHub Secrets",
        "PINECONE_API_KEY": "Get a free key at https://app.pinecone.io — add to GitHub Secrets",
        "PR_NUMBER": "Auto-injected by GitHub Actions workflow",
        "REPO_FULL_NAME": "Auto-injected by GitHub Actions workflow",
    }

    missing = []
    for var, hint in required.items():
        if not os.environ.get(var):
            missing.append(f"  ✗ {var}: {hint}")

    if missing:
        logger.error(
            "Missing required environment variables:\n%s", "\n".join(missing)
        )
        sys.exit(1)

    logger.info("Environment validation passed ✓")


# ──────────────────────────────────────────────────────────────
# PHASE 1: MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def run_phase1() -> None:
    """
    Executes the complete Phase 1 review pipeline:
      Step 1 → Validate environment
      Step 2 → Fetch PR data from GitHub
      Step 3 → Fetch content of changed files
      Step 4 → Run static analysis
      Step 5 → Generate LLM review via Groq
      Step 6 → Post review comment to GitHub PR
    """

    # ── Step 1: Environment Check ──────────────────────────────
    logger.info("=" * 60)
    logger.info("PR Review Agent — Phase 1 Starting")
    logger.info("=" * 60)
    validate_environment()

    # ── Step 2: Fetch PR Data ──────────────────────────────────
    logger.info("Step 2: Initialising GitHub client...")
    gh = GitHubClient()

    pr_metadata = gh.get_pr_metadata()
    logger.info(
        "PR metadata fetched | PR=#%d | title='%s' | author=%s | files=%d",
        pr_metadata["number"],
        pr_metadata["title"],
        pr_metadata["author"],
        pr_metadata["changed_files"],
    )

    diff = gh.get_pr_diff()
    if not diff:
        logger.warning("Empty diff received — PR may have no text changes.")

    changed_files = gh.get_changed_files()
    logger.info("Changed files retrieved: %d file(s)", len(changed_files))

    # ── Step 3: Fetch File Contents ────────────────────────────
    # We need the actual file content (not just the diff patch)
    # for static analysis tools like flake8 and yamllint.
    # Deleted files are skipped — they have no content to fetch.
    logger.info("Step 3: Fetching file contents for static analysis...")

    for file_info in changed_files:
        if file_info["status"] == "removed":
            file_info["content"] = ""
            logger.debug("Skipping content fetch for removed file: %s", file_info["filename"])
            continue

        content = gh.get_file_content(file_info["filename"])
        file_info["content"] = content
        logger.debug(
            "Fetched content | file=%s | chars=%d",
            file_info["filename"],
            len(content),
        )

    # ── Step 4: Static Analysis ────────────────────────────────
    logger.info("Step 4: Running static analysis...")
    static_report = analyze_changed_files(changed_files)

    if static_report:
        logger.info("Static analysis found issues — injecting into LLM context")
    else:
        logger.info("Static analysis: no issues found across changed files")

    # ── Step 5: LLM Review Generation ─────────────────────────
    logger.info("Step 5: Generating AI review via Groq (Llama 3)...")
    llm = GroqLLMClient()

    review_markdown = llm.generate_review(
        pr_metadata=pr_metadata,
        diff=diff,
        static_analysis_report=static_report,
    )
    logger.info("LLM review generated | chars=%d", len(review_markdown))

    # ── Step 6: Post to GitHub ─────────────────────────────────
    logger.info("Step 6: Posting review comment to GitHub PR #%d...", pr_metadata["number"])

    # Prepend a metadata header to the GitHub comment
    # This helps engineering managers immediately see key stats
    header = _build_comment_header(pr_metadata)
    full_comment = header + "\n\n" + review_markdown

    gh.post_pr_comment(full_comment)

    logger.info("=" * 60)
    logger.info("Phase 1 Complete — Review posted to PR #%d", pr_metadata["number"])
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# HELPER: COMMENT HEADER
# ──────────────────────────────────────────────────────────────

def _build_comment_header(pr_metadata: dict) -> str:
    """
    Builds a compact metadata table that sits at the top of
    every GitHub comment posted by the agent.

    This gives engineering managers an instant snapshot of
    PR scope before reading the detailed review.
    """
    return (
        f"<!-- pr-review-agent -->\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| **PR** | #{pr_metadata['number']}: {pr_metadata['title']} |\n"
        f"| **Author** | `{pr_metadata['author']}` |\n"
        f"| **Branch** | `{pr_metadata['head_branch']}` → `{pr_metadata['base_branch']}` |\n"
        f"| **Scope** | "
        f"+{pr_metadata['additions']} / -{pr_metadata['deletions']} lines, "
        f"{pr_metadata['changed_files']} file(s) |\n"
        f"| **Review Engine** | Llama 3.3 70B via Groq |\n"
    )


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Called directly by GitHub Actions:
      run: python src/agent.py

    As phases are added, this block will route to the correct
    phase runner. For now, Phase 1 is always executed.
    """
    try:
        run_phase1()
    except Exception as e:
        logger.exception("Unhandled exception in agent: %s", e)
        sys.exit(1)