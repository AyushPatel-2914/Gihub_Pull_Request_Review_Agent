"""
src/ml/intent_router.py
─────────────────────────────────────────────────────────────
Intent Router — Classifies PR Type and Swaps Review Rules

WHY THIS MATTERS:
  A bug fix and a new feature need different review lenses.
  - BUG_FIX     → focus on root cause, regression risk, test coverage
  - NEW_FEATURE → focus on design, API surface, edge cases, docs
  - REFACTOR    → focus on behavior preservation, no logic change
  - DOCS        → focus on clarity, accuracy, completeness
  - CONFIG      → focus on security, correctness, breaking changes

  Instead of one generic prompt, the router detects intent and
  swaps in a specialized review instruction set.

PLUGGABLE DESIGN:
  This module exposes a single classify_intent() function backed
  by a strategy. Phase 3 ships the LLM strategy. A local
  zero-shot model strategy can be dropped in later by setting
  INTENT_STRATEGY = "local" without touching the agent.
─────────────────────────────────────────────────────────────
"""

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Which strategy to use: "llm" (Phase 3) or "local" (future Option B)
INTENT_STRATEGY = os.environ.get("INTENT_STRATEGY", "llm")


class PRIntent(str, Enum):
    """The set of PR intents the router can classify."""
    BUG_FIX = "BUG_FIX"
    NEW_FEATURE = "NEW_FEATURE"
    REFACTOR = "REFACTOR"
    DOCS = "DOCS"
    CONFIG = "CONFIG"
    UNKNOWN = "UNKNOWN"


@dataclass
class IntentResult:
    """
    Result of intent classification.

    Fields:
      intent      : the classified PRIntent
      confidence  : 0.0–1.0 confidence score
      reasoning   : short explanation of why this intent was chosen
      review_focus: specialized review instructions for this intent
    """
    intent: PRIntent
    confidence: float
    reasoning: str
    review_focus: str


# ──────────────────────────────────────────────────────────────
# REVIEW FOCUS RULES PER INTENT
# ──────────────────────────────────────────────────────────────
# These get injected into the LLM review prompt so the review
# is tailored to what actually matters for each kind of change.

REVIEW_FOCUS = {
    PRIntent.BUG_FIX: (
        "This is a BUG FIX. Focus your review on:\n"
        "- Does the fix address the ROOT CAUSE, not just the symptom?\n"
        "- Could this fix introduce a regression elsewhere?\n"
        "- Is there a test that would have caught this bug and prevents recurrence?\n"
        "- Are there other places in the codebase with the same bug pattern?\n"
        "- Is the fix minimal and surgical, or does it change unrelated code?"
    ),
    PRIntent.NEW_FEATURE: (
        "This is a NEW FEATURE. Focus your review on:\n"
        "- Is the API/interface design clean and consistent with the codebase?\n"
        "- Are edge cases and error conditions handled?\n"
        "- Is the feature adequately tested?\n"
        "- Is there documentation for how to use it?\n"
        "- Does it introduce security, performance, or scalability concerns?\n"
        "- Does it follow existing patterns in the codebase?"
    ),
    PRIntent.REFACTOR: (
        "This is a REFACTOR. Focus your review on:\n"
        "- Is BEHAVIOR PRESERVED? Refactors must not change functionality.\n"
        "- Are existing tests still passing and sufficient to prove equivalence?\n"
        "- Is the new structure genuinely cleaner/more maintainable?\n"
        "- Was anything accidentally changed beyond the refactor scope?\n"
        "- Are there hidden behavioral differences (ordering, exceptions, types)?"
    ),
    PRIntent.DOCS: (
        "This is a DOCUMENTATION change. Focus your review on:\n"
        "- Is the information ACCURATE and up to date with the code?\n"
        "- Is it clear, well-structured, and easy to follow?\n"
        "- Are there broken links, typos, or formatting issues?\n"
        "- Is anything important missing or under-explained?\n"
        "Keep code-style critiques minimal — this is prose, not logic."
    ),
    PRIntent.CONFIG: (
        "This is a CONFIGURATION change. Focus your review on:\n"
        "- Are there SECURITY implications (exposed secrets, weak settings)?\n"
        "- Could this break existing deployments or environments?\n"
        "- Are values correct, valid, and within sane ranges?\n"
        "- Is the change documented for other developers?\n"
        "- Are there defaults that should be overridable?"
    ),
    PRIntent.UNKNOWN: (
        "Review this PR holistically, covering correctness, design, "
        "testing, security, and maintainability."
    ),
}


# ──────────────────────────────────────────────────────────────
# LLM-BASED CLASSIFIER (Phase 3 — Option A)
# ──────────────────────────────────────────────────────────────

INTENT_CLASSIFY_PROMPT = """You are a PR classifier. Given a pull request's title, \
description, and file changes, classify its primary intent into EXACTLY ONE category:

- BUG_FIX: Fixes a defect or incorrect behavior
- NEW_FEATURE: Adds new functionality or capabilities
- REFACTOR: Restructures code without changing behavior
- DOCS: Changes documentation, comments, or markdown only
- CONFIG: Changes configuration, CI, dependencies, or build files

Respond with ONLY a JSON object, no other text:
{"intent": "CATEGORY", "confidence": 0.0-1.0, "reasoning": "one short sentence"}"""


def _classify_with_llm(pr_metadata: dict, diff: str, changed_files: list) -> IntentResult:
    """
    Uses the Groq LLM to classify PR intent.
    Makes one fast, cheap classification call.

    Args:
      pr_metadata   : dict with title, body
      diff          : the unified diff (truncated for the classifier)
      changed_files : list of changed file dicts

    Returns:
      IntentResult
    """
    # Lazy import to avoid circular dependency with the LLM client
    from llm.groq_client import GroqLLMClient

    # Build a compact summary for the classifier
    file_list = "\n".join(
        f"  - {f['filename']} ({f['status']}, +{f['additions']}/-{f['deletions']})"
        for f in changed_files[:20]
    )

    user_message = (
        f"PR Title: {pr_metadata.get('title', '')}\n\n"
        f"PR Description: {pr_metadata.get('body', '') or '(none)'}\n\n"
        f"Changed Files:\n{file_list}\n\n"
        f"Diff (truncated):\n```\n{diff[:3000]}\n```"
    )

    try:
        client = GroqLLMClient()
        # Use the raw chat method with a low token budget — this is a
        # classification, not a full review
        response = client.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": INTENT_CLASSIFY_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=150,
            temperature=0.0,  # deterministic classification
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        intent_str = parsed.get("intent", "UNKNOWN").upper()
        try:
            intent = PRIntent(intent_str)
        except ValueError:
            logger.warning("LLM returned unknown intent '%s'", intent_str)
            intent = PRIntent.UNKNOWN

        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "")

        logger.info(
            "Intent classified (LLM) | intent=%s | confidence=%.2f",
            intent.value, confidence,
        )

        return IntentResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            review_focus=REVIEW_FOCUS[intent],
        )

    except Exception as e:
        logger.error("LLM intent classification failed: %s — using heuristic", e)
        return _classify_with_heuristic(pr_metadata, changed_files)


# ──────────────────────────────────────────────────────────────
# HEURISTIC FALLBACK
# ──────────────────────────────────────────────────────────────

def _classify_with_heuristic(pr_metadata: dict, changed_files: list) -> IntentResult:
    """
    A keyword + file-extension heuristic used if the LLM call fails.
    Never crashes — always returns a usable result.
    """
    title = (pr_metadata.get("title", "") or "").lower()
    body = (pr_metadata.get("body", "") or "").lower()
    text = title + " " + body

    filenames = [f["filename"].lower() for f in changed_files]

    # Known config file names (checked before docs so requirements.txt
    # isn't mistaken for documentation just because it ends in .txt)
    config_names = ("dockerfile", "requirements.txt", "makefile", "package.json")
    config_exts = (".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".lock")

    def _is_config(f):
        return f.endswith(config_exts) or any(c in f for c in config_names)

    # All files are config?
    if filenames and all(_is_config(f) for f in filenames):
        return IntentResult(PRIntent.CONFIG, 0.7, "All changed files are configuration.", REVIEW_FOCUS[PRIntent.CONFIG])

    # All files are docs? (exclude anything already caught as config)
    if filenames and all(
        f.endswith((".md", ".rst", ".txt")) and not _is_config(f)
        for f in filenames
    ):
        return IntentResult(PRIntent.DOCS, 0.7, "All changed files are documentation.", REVIEW_FOCUS[PRIntent.DOCS])

    # Keyword matching
    if any(kw in text for kw in ["fix", "bug", "patch", "resolve", "issue", "crash", "error"]):
        return IntentResult(PRIntent.BUG_FIX, 0.6, "Title/description suggests a bug fix.", REVIEW_FOCUS[PRIntent.BUG_FIX])
    if any(kw in text for kw in ["refactor", "cleanup", "restructure", "rename", "reorganize"]):
        return IntentResult(PRIntent.REFACTOR, 0.6, "Title/description suggests a refactor.", REVIEW_FOCUS[PRIntent.REFACTOR])
    if any(kw in text for kw in ["add", "feature", "implement", "new", "support", "introduce"]):
        return IntentResult(PRIntent.NEW_FEATURE, 0.6, "Title/description suggests a new feature.", REVIEW_FOCUS[PRIntent.NEW_FEATURE])

    return IntentResult(PRIntent.UNKNOWN, 0.4, "Could not confidently classify intent.", REVIEW_FOCUS[PRIntent.UNKNOWN])


# ──────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ──────────────────────────────────────────────────────────────

def classify_intent(pr_metadata: dict, diff: str, changed_files: list) -> IntentResult:
    """
    Main entry point. Classifies PR intent using the configured strategy.

    Strategy is chosen via the INTENT_STRATEGY env var:
      "llm"   → Groq LLM classifier (default, Phase 3 Option A)
      "local" → local zero-shot model (future Option B — not yet built)

    Args:
      pr_metadata   : dict from GitHubClient.get_pr_metadata()
      diff          : raw unified diff string
      changed_files : list of changed file dicts

    Returns:
      IntentResult
    """
    if INTENT_STRATEGY == "local":
        # Placeholder for Option B — falls back to LLM for now
        logger.info("Local intent strategy not yet implemented — using LLM")
        return _classify_with_llm(pr_metadata, diff, changed_files)

    return _classify_with_llm(pr_metadata, diff, changed_files)
