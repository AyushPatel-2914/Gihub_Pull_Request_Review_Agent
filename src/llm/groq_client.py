"""
src/llm/groq_client.py
─────────────────────────────────────────────────────────────
Groq LLM Client — Llama 3 via Groq API (Phase 1 + 3)

Phase 3 additions:
  - generate_structured_review(): returns a JSON object with
    per-line findings (file path, line number, severity, comment)
    so the agent can post INLINE comments on specific diff lines.
  - Intent-aware prompting: the review focus from the intent
    router is injected into the system prompt.
─────────────────────────────────────────────────────────────
"""

import os
import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_INPUT_CHARS = 80_000
MAX_OUTPUT_TOKENS = 4_096


# ──────────────────────────────────────────────────────────────
# PHASE 1 PROMPT (markdown summary review) — still used as fallback
# ──────────────────────────────────────────────────────────────

PHASE1_SYSTEM_PROMPT = """You are an expert Senior Software Engineer and Code Reviewer.
Your task is to review a GitHub Pull Request diff and provide actionable, constructive feedback.

Your review must:
- Be written in clear, professional markdown
- Identify bugs, security issues, performance problems, and design concerns
- Acknowledge what the PR does well
- Distinguish between CRITICAL issues, SUGGESTIONS, and NITPICKS
- End with an overall assessment: APPROVE / REQUEST CHANGES / NEEDS DISCUSSION

Format:

## 🤖 Automated PR Review

### 📋 Summary
[overview]

### ✅ What's Good
[bullets]

### 🚨 Critical Issues
[bullets or NONE]

### 💡 Suggestions
[bullets]

### 🔍 Nitpicks
[bullets or NONE]

### 📊 Overall Verdict
**[APPROVE / REQUEST CHANGES / NEEDS DISCUSSION]** — [reason]

---
*This review was generated automatically. Please use human judgment before merging.*
"""


# ──────────────────────────────────────────────────────────────
# PHASE 3 PROMPT (structured JSON for inline comments)
# ──────────────────────────────────────────────────────────────

STRUCTURED_REVIEW_PROMPT = """You are an expert Senior Software Engineer reviewing a GitHub Pull Request.

{intent_focus}

You will receive the PR diff, relevant codebase context, and static analysis findings.

Produce a review as a STRICT JSON object with this exact schema:
{{
  "summary": "2-3 sentence overview of the PR and your impression",
  "verdict": "APPROVE | REQUEST_CHANGES | NEEDS_DISCUSSION",
  "positives": ["string", ...],
  "inline_comments": [
    {{
      "file": "exact/path/from/diff.py",
      "line": 42,
      "severity": "CRITICAL | SUGGESTION | NITPICK",
      "comment": "specific, actionable feedback for this exact line"
    }}
  ],
  "general_comments": ["feedback not tied to a specific line", ...]
}}

CRITICAL RULES:
- Respond with ONLY the JSON object, no markdown fences, no other text.
- For "line", use the line number in the NEW version of the file (the + side of the diff).
- Only create inline_comments for lines that are actually changed (appear with + in the diff).
- If a finding isn't tied to a specific changed line, put it in general_comments.
- Be precise with file paths — copy them exactly from the diff headers.
- Limit to the most important 10 inline comments. Quality over quantity.
- If the PR is clean, inline_comments can be an empty array."""


class GroqLLMClient:
    """Wrapper around the Groq SDK for both markdown and structured reviews."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        logger.info("GroqLLMClient initialised | model=%s", GROQ_MODEL)

    # ──────────────────────────────────────────────────────────
    # SHARED MESSAGE BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_user_message(self, pr_metadata, diff, context_report):
        truncated_diff = diff
        if len(diff) > MAX_INPUT_CHARS:
            truncated_diff = diff[:MAX_INPUT_CHARS] + "\n\n[DIFF TRUNCATED]\n"
            logger.warning("Diff truncated to %d chars", MAX_INPUT_CHARS)

        parts = [
            "# Pull Request Information",
            f"**PR #{pr_metadata['number']}**: {pr_metadata['title']}",
            f"**Author**: {pr_metadata['author']}",
            f"**Branch**: `{pr_metadata['head_branch']}` → `{pr_metadata['base_branch']}`",
            f"**Scope**: +{pr_metadata['additions']} / -{pr_metadata['deletions']} lines",
            "",
            "## PR Description",
            pr_metadata["body"] or "*No description provided.*",
            "",
        ]
        if context_report and context_report.strip():
            parts.append(context_report)
            parts.append("")
        parts += ["## Raw Unified Diff", "```diff", truncated_diff, "```"]
        return "\n".join(parts)

    # ──────────────────────────────────────────────────────────
    # PHASE 1: MARKDOWN REVIEW (fallback)
    # ──────────────────────────────────────────────────────────

    def generate_review(self, pr_metadata, diff, static_analysis_report="",
                        system_prompt=PHASE1_SYSTEM_PROMPT):
        """Generates a markdown review (Phase 1 style). Used as a fallback."""
        user_message = self._build_user_message(pr_metadata, diff, static_analysis_report)
        logger.info("Sending markdown review to Groq | chars=%d", len(user_message))
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Groq markdown review failed: %s", e)
            return (
                "## 🤖 Automated PR Review\n\n"
                f"⚠️ Review failed: `{type(e).__name__}: {e}`\n\nPlease review manually."
            )

    # ──────────────────────────────────────────────────────────
    # PHASE 3: STRUCTURED JSON REVIEW
    # ──────────────────────────────────────────────────────────

    def generate_structured_review(self, pr_metadata, diff, context_report="",
                                   intent_focus=""):
        """
        Generates a structured JSON review for inline comments.

        Args:
          pr_metadata   : PR metadata dict
          diff          : unified diff string
          context_report: combined static analysis + RAG context
          intent_focus  : review focus instructions from the intent router

        Returns:
          dict with keys: summary, verdict, positives,
          inline_comments, general_comments
          Returns a safe fallback dict if parsing fails.
        """
        system_prompt = STRUCTURED_REVIEW_PROMPT.format(
            intent_focus=intent_focus or "Review this PR holistically."
        )
        user_message = self._build_user_message(pr_metadata, diff, context_report)

        logger.info("Sending structured review to Groq | chars=%d", len(user_message))

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.2,
                response_format={"type": "json_object"},  # Groq JSON mode
            )
            raw = response.choices[0].message.content.strip()
            usage = response.usage
            logger.info(
                "Structured review received | prompt=%d | completion=%d",
                usage.prompt_tokens, usage.completion_tokens,
            )

            # Strip any stray fences just in case
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            # Validate and normalize the structure
            return self._validate_structured_review(parsed)

        except json.JSONDecodeError as e:
            logger.error("Failed to parse structured review JSON: %s", e)
            return self._fallback_review("The review output could not be parsed as JSON.")
        except Exception as e:
            logger.error("Groq structured review failed: %s", e)
            return self._fallback_review(f"{type(e).__name__}: {e}")

    def _validate_structured_review(self, parsed: dict) -> dict:
        """
        Ensures the parsed review has all required keys with safe defaults.
        Filters out malformed inline comments.
        """
        valid_severities = {"CRITICAL", "SUGGESTION", "NITPICK"}
        valid_verdicts = {"APPROVE", "REQUEST_CHANGES", "NEEDS_DISCUSSION"}

        # Clean inline comments
        clean_inline = []
        for c in parsed.get("inline_comments", []):
            if not isinstance(c, dict):
                continue
            if "file" not in c or "line" not in c or "comment" not in c:
                continue
            try:
                line = int(c["line"])
            except (ValueError, TypeError):
                continue
            severity = str(c.get("severity", "SUGGESTION")).upper()
            if severity not in valid_severities:
                severity = "SUGGESTION"
            clean_inline.append({
                "file": str(c["file"]),
                "line": line,
                "severity": severity,
                "comment": str(c["comment"]),
            })

        verdict = str(parsed.get("verdict", "NEEDS_DISCUSSION")).upper()
        if verdict not in valid_verdicts:
            verdict = "NEEDS_DISCUSSION"

        return {
            "summary": str(parsed.get("summary", "")),
            "verdict": verdict,
            "positives": [str(p) for p in parsed.get("positives", [])],
            "inline_comments": clean_inline,
            "general_comments": [str(g) for g in parsed.get("general_comments", [])],
        }

    def _fallback_review(self, error_msg: str) -> dict:
        """Returns a safe review dict when generation/parsing fails."""
        return {
            "summary": f"⚠️ The structured review could not be completed: {error_msg}",
            "verdict": "NEEDS_DISCUSSION",
            "positives": [],
            "inline_comments": [],
            "general_comments": ["Please review this PR manually."],
        }
