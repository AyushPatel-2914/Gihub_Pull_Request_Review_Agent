"""
src/llm/groq_client.py
─────────────────────────────────────────────────────────────
Groq LLM Client — Llama 3 via Groq API
Phase 1 responsibility:
  - Authenticate with Groq using GROQ_API_KEY
  - Build a structured system prompt for code review
  - Send the diff + static analysis findings to Llama 3
  - Return the raw markdown review text

The model used is llama-3.3-70b-versatile:
  - Free on Groq's developer tier
  - 128k context window (enough for large diffs)
  - Fast inference (~300 tokens/sec on Groq)

In Phase 3 this client will be extended to request
structured JSON output with file paths and line numbers.
─────────────────────────────────────────────────────────────
"""

import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# ── Model Configuration ────────────────────────────────────────
# llama-3.3-70b-versatile: best free model on Groq for reasoning
# Change this if Groq adds newer Llama models to their free tier
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Token Safety Limits ────────────────────────────────────────
# Groq free tier rate limits — stay conservative
MAX_INPUT_CHARS = 80_000    # ~20k tokens; leaves room for output
MAX_OUTPUT_TOKENS = 4_096   # sufficient for a thorough PR review


# ──────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ──────────────────────────────────────────────────────────────

PHASE1_SYSTEM_PROMPT = """You are an expert Senior Software Engineer and Code Reviewer.
Your task is to review a GitHub Pull Request diff and provide actionable, constructive feedback.

You will receive:
1. PR metadata (title, description, author, branch info)
2. Static analysis findings (objective lint/syntax issues already detected)
3. The raw unified diff of all changes

Your review must:
- Be written in clear, professional markdown
- Identify bugs, security issues, performance problems, and design concerns
- Acknowledge what the PR does well — not just problems
- Reference specific file names and approximate line areas when possible
- Distinguish between CRITICAL issues (must fix), SUGGESTIONS (should consider), and NITPICKS (minor style)
- NOT repeat findings already covered by the static analysis section
- End with an overall assessment: APPROVE / REQUEST CHANGES / NEEDS DISCUSSION

Format your response EXACTLY as follows:

## 🤖 Automated PR Review

### 📋 Summary
[2-3 sentence overview of what this PR does and your overall impression]

### ✅ What's Good
[Bullet points of positive aspects]

### 🚨 Critical Issues
[Bullet points of must-fix problems. Write NONE if nothing critical.]

### 💡 Suggestions
[Bullet points of improvements worth considering]

### 🔍 Nitpicks
[Minor style/naming comments. Write NONE if nothing to add.]

### 📊 Overall Verdict
**[APPROVE / REQUEST CHANGES / NEEDS DISCUSSION]** — [one sentence reason]

---
*This review was generated automatically. Please use human judgment before merging.*
"""


# ──────────────────────────────────────────────────────────────
# GROQ CLIENT CLASS
# ──────────────────────────────────────────────────────────────

class GroqLLMClient:
    """
    Thin wrapper around the Groq Python SDK.
    Manages authentication, prompt building, and API calls.
    """

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it "
                "to GitHub Secrets or your local .env file."
            )
        self.client = Groq(api_key=api_key)
        logger.info("GroqLLMClient initialised | model=%s", GROQ_MODEL)

    def _build_user_message(
        self,
        pr_metadata: dict,
        diff: str,
        static_analysis_report: str,
    ) -> str:
        """
        Assembles the user-turn message that gets sent to the LLM.
        Keeps each section clearly labelled so the model can parse
        what it's looking at.

        Args:
          pr_metadata          : dict from GitHubClient.get_pr_metadata()
          diff                 : raw unified diff string
          static_analysis_report: pre-formatted string from static_analyzer

        Returns:
          str: the complete user message
        """
        # Truncate diff if it's enormous — safety valve for Phase 1
        # Phase 5 will add proper token counting and chunking
        truncated_diff = diff
        if len(diff) > MAX_INPUT_CHARS:
            truncated_diff = diff[:MAX_INPUT_CHARS]
            truncated_diff += (
                "\n\n[DIFF TRUNCATED: This PR is very large. "
                "Only the first portion is shown. "
                "Phase 5 will add full token-aware chunking.]\n"
            )
            logger.warning(
                "Diff truncated from %d to %d chars for Phase 1",
                len(diff),
                MAX_INPUT_CHARS,
            )

        # Build the message
        parts = [
            "# Pull Request Information",
            f"**PR #{pr_metadata['number']}**: {pr_metadata['title']}",
            f"**Author**: {pr_metadata['author']}",
            f"**Branch**: `{pr_metadata['head_branch']}` → `{pr_metadata['base_branch']}`",
            f"**Scope**: +{pr_metadata['additions']} / -{pr_metadata['deletions']} lines "
            f"across {pr_metadata['changed_files']} file(s)",
            "",
            "## PR Description",
            pr_metadata["body"] or "*No description provided.*",
            "",
        ]

        # Inject static analysis if there are findings
        if static_analysis_report.strip():
            parts.append(static_analysis_report)
            parts.append("")

        parts += [
            "## Raw Unified Diff",
            "```diff",
            truncated_diff,
            "```",
        ]

        return "\n".join(parts)

    def generate_review(
        self,
        pr_metadata: dict,
        diff: str,
        static_analysis_report: str = "",
        system_prompt: str = PHASE1_SYSTEM_PROMPT,
    ) -> str:
        """
        Sends the PR context to Llama 3 via Groq and returns
        a markdown-formatted code review.

        Args:
          pr_metadata           : dict from GitHubClient.get_pr_metadata()
          diff                  : raw unified diff string
          static_analysis_report: optional pre-analysis string
          system_prompt         : system prompt (swappable in later phases)

        Returns:
          str: markdown review text ready to post to GitHub
        """
        user_message = self._build_user_message(
            pr_metadata, diff, static_analysis_report
        )

        logger.info(
            "Sending to Groq | model=%s | user_msg_chars=%d",
            GROQ_MODEL,
            len(user_message),
        )

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.2,    # Low temperature = more deterministic reviews
                top_p=0.9,
            )

            review_text = response.choices[0].message.content
            usage = response.usage

            logger.info(
                "Groq response received | "
                "prompt_tokens=%d | completion_tokens=%d | total=%d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

            return review_text

        except Exception as e:
            logger.error("Groq API call failed: %s", e)
            # Return a graceful fallback comment rather than crashing
            return (
                "## 🤖 Automated PR Review\n\n"
                "⚠️ **The AI review agent encountered an error and could not "
                f"complete the review.**\n\n"
                f"Error details: `{type(e).__name__}: {e}`\n\n"
                "Please review this PR manually."
            )