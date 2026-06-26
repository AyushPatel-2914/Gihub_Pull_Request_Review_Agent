import os, json, logging
from groq import Groq

logger = logging.getLogger(__name__)
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_INPUT_CHARS = 80000
MAX_OUTPUT_TOKENS = 4096

PHASE1_SYSTEM_PROMPT = """You are an expert Senior Software Engineer and Code Reviewer.
Review the PR diff and provide actionable feedback in markdown.
Format:
## Automated PR Review
### Summary
### What is Good
### Critical Issues
### Suggestions
### Nitpicks
### Overall Verdict
**[APPROVE / REQUEST CHANGES / NEEDS DISCUSSION]** - reason
---
*Automated review. Use human judgment before merging.*"""

STRUCTURED_REVIEW_PROMPT = """You are an expert Senior Software Engineer reviewing a GitHub Pull Request.

{intent_focus}

Produce a review as a STRICT JSON object:
{{
  "summary": "2-3 sentence overview",
  "verdict": "APPROVE | REQUEST_CHANGES | NEEDS_DISCUSSION",
  "positives": ["string"],
  "inline_comments": [
    {{
      "file": "exact/path/file.py",
      "line": 42,
      "severity": "CRITICAL | SUGGESTION | NITPICK",
      "comment": "specific actionable feedback"
    }}
  ],
  "general_comments": ["feedback not tied to a specific line"]
}}

RULES:
- Respond with ONLY the JSON object, no markdown fences.
- Only comment on lines that appear with + in the diff.
- Copy file paths exactly from the diff headers.
- Limit to 10 most important inline comments.
- If PR is clean, inline_comments can be empty array."""


class GroqLLMClient:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=api_key)
        logger.info("GroqLLMClient initialised | model=%s", GROQ_MODEL)

    def _build_user_message(self, pr_metadata, diff, context_report):
        truncated_diff = diff
        if len(diff) > MAX_INPUT_CHARS:
            truncated_diff = diff[:MAX_INPUT_CHARS] + "\n\n[DIFF TRUNCATED]\n"
        parts = [
            "# Pull Request Information",
            f"**PR #{pr_metadata['number']}**: {pr_metadata['title']}",
            f"**Author**: {pr_metadata['author']}",
            f"**Branch**: `{pr_metadata['head_branch']}` to `{pr_metadata['base_branch']}`",
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

    def generate_review(self, pr_metadata, diff, static_analysis_report="", system_prompt=PHASE1_SYSTEM_PROMPT):
        user_message = self._build_user_message(pr_metadata, diff, static_analysis_report)
        logger.info("Sending markdown review to Groq | chars=%d", len(user_message))
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                max_tokens=MAX_OUTPUT_TOKENS, temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Groq review failed: %s", e)
            return f"## Automated PR Review\n\nReview failed: `{e}`\n\nPlease review manually."

    def generate_structured_review(self, pr_metadata, diff, context_report="", intent_focus=""):
        system_prompt = STRUCTURED_REVIEW_PROMPT.format(intent_focus=intent_focus or "Review holistically.")
        user_message = self._build_user_message(pr_metadata, diff, context_report)
        logger.info("Sending structured review to Groq | chars=%d", len(user_message))
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                max_tokens=MAX_OUTPUT_TOKENS, temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
            logger.info("Structured review received | tokens=%d", response.usage.total_tokens)
            return self._validate_structured_review(json.loads(raw))
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON: %s", e)
            return self._fallback_review("Output could not be parsed as JSON.")
        except Exception as e:
            logger.error("Groq structured review failed: %s", e)
            return self._fallback_review(f"{type(e).__name__}: {e}")

    def _validate_structured_review(self, parsed):
        valid_severities = {"CRITICAL", "SUGGESTION", "NITPICK"}
        valid_verdicts = {"APPROVE", "REQUEST_CHANGES", "NEEDS_DISCUSSION"}
        clean_inline = []
        for c in parsed.get("inline_comments", []):
            if not isinstance(c, dict): continue
            if "file" not in c or "line" not in c or "comment" not in c: continue
            try:
                line = int(c["line"])
            except (ValueError, TypeError):
                continue
            severity = str(c.get("severity", "SUGGESTION")).upper()
            if severity not in valid_severities:
                severity = "SUGGESTION"
            clean_inline.append({"file": str(c["file"]), "line": line, "severity": severity, "comment": str(c["comment"])})
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

    def _fallback_review(self, error_msg):
        return {"summary": f"Review could not be completed: {error_msg}", "verdict": "NEEDS_DISCUSSION", "positives": [], "inline_comments": [], "general_comments": ["Please review manually."]}
