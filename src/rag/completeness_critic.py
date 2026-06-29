"""
src/rag/completeness_critic.py
─────────────────────────────────────────────────────────────
Completeness Critic — LLM-Based Context Sufficiency Check (Phase 4)

THE GAP THIS FILLS:
  Phase 2's self-healing checks completeness by STRING MATCHING —
  it looks for symbol names verbatim in the retrieved content.
  This produces false alarms:

    PR calls: validate_token()
    Context has: check_jwt(), verify_auth()  (the real implementation)
    String match: "validate_token NOT FOUND" → unnecessary healing

  And it misses real gaps:

    PR calls: process_payment()
    Context has: a DIFFERENT process_payment() from another module
    String match: "process_payment FOUND" → false confidence

  The Completeness Critic asks the LLM to JUDGE sufficiency
  semantically: "Given this diff and this context, can you review
  it well, or is something important missing?"

HOW IT FITS THE SELF-HEALING LOOP:
  Phase 2 loop (string-based):
    retrieve → string-check symbols → heal if names missing

  Phase 4 loop (string + semantic):
    retrieve → string-check symbols → heal if names missing
            → LLM completeness critique → heal if LLM says insufficient
            → targeted re-query for exactly what the LLM flagged

  The LLM critique runs AFTER string-based healing, as a final
  semantic gate before the review is generated.

COST CONTROL:
  This adds one LLM call per critique. To keep it cheap:
    - It runs at most ONCE per PR (not per healing round)
    - It uses a small token budget (the critique is short)
    - It only triggers if there are referenced symbols to check
─────────────────────────────────────────────────────────────
"""

import json
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# CRITIQUE PROMPT
# ──────────────────────────────────────────────────────────────

COMPLETENESS_CRITIQUE_PROMPT = """You are a code review context auditor.

Your job is NOT to review the code. Your job is to judge whether the \
provided CONTEXT is sufficient to thoroughly review the PR DIFF.

A reviewer needs to see the definitions of functions, classes, and \
constants that the changed code CALLS or DEPENDS ON, so they can judge \
correctness. If those definitions are missing from the context, the \
review would be guessing.

Analyze the diff and the context, then respond with ONLY a JSON object:
{
  "sufficient": true | false,
  "confidence": 0.0-1.0,
  "missing": ["specific function/class/symbol names whose definitions are needed but absent"],
  "reasoning": "one sentence explaining your judgment"
}

RULES:
- "missing" should list ONLY symbols genuinely needed for review that are \
absent from the context. Do not list standard library calls, builtins, or \
trivially obvious things.
- If the context is adequate, set sufficient=true and missing=[].
- Be conservative: only flag something as missing if its absence would \
genuinely harm the review quality.
- Respond with ONLY the JSON, no other text."""


class CompletenessCritic:
    """
    Uses the LLM to judge whether retrieved context is sufficient.
    Reuses the existing Groq client — no new model downloads.
    """

    def __init__(self, groq_client):
        """
        Args:
          groq_client : an initialized GroqLLMClient instance
                        (reused so we don't create multiple clients)
        """
        self.client = groq_client

    def critique(self, diff: str, context_chunks: list) -> dict:
        """
        Asks the LLM whether the context is sufficient to review the diff.

        Args:
          diff           : the unified diff string
          context_chunks : list of retrieved chunk dicts

        Returns:
          dict with keys:
            - sufficient : bool
            - confidence : float
            - missing    : list[str] of symbol names still needed
            - reasoning  : str
          Returns a safe "sufficient=true" default if the call fails,
          so a critique failure never blocks the review.
        """
        # Build a compact context summary for the critique
        context_summary = self._summarize_context(context_chunks)

        # Truncate diff for the critique (we don't need the whole thing)
        diff_excerpt = diff[:4000]

        user_message = (
            f"## PR DIFF\n```diff\n{diff_excerpt}\n```\n\n"
            f"## RETRIEVED CONTEXT\n{context_summary}\n\n"
            f"Is this context sufficient to thoroughly review the diff?"
        )

        try:
            response = self.client.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": COMPLETENESS_CRITIQUE_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=300,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            result = self._validate_critique(parsed)
            logger.info(
                "Completeness critique | sufficient=%s | confidence=%.2f | missing=%s",
                result["sufficient"], result["confidence"], result["missing"][:5],
            )
            return result

        except Exception as e:
            logger.error("Completeness critique failed: %s — assuming sufficient", e)
            # Fail open: never block a review because the critique errored
            return {
                "sufficient": True,
                "confidence": 0.0,
                "missing": [],
                "reasoning": "Critique unavailable; proceeding with retrieved context.",
            }

    def _summarize_context(self, context_chunks: list) -> str:
        """
        Builds a compact representation of the retrieved context
        for the critique prompt — filepath, symbol, and a content
        snippet for each chunk.
        """
        if not context_chunks:
            return "(no context retrieved)"

        parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            meta = chunk.get("metadata", {})
            filepath = meta.get("filepath", "unknown")
            symbol = meta.get("symbol_name", "")
            content = meta.get("content", "")[:600]  # snippet only
            label = f"[{i}] {filepath}"
            if symbol:
                label += f" ({symbol})"
            parts.append(f"{label}:\n{content}\n")
        return "\n".join(parts)

    def _validate_critique(self, parsed: dict) -> dict:
        """Normalizes the LLM critique output with safe defaults."""
        sufficient = bool(parsed.get("sufficient", True))
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.5
        missing = parsed.get("missing", [])
        if not isinstance(missing, list):
            missing = []
        missing = [str(m) for m in missing if m]
        reasoning = str(parsed.get("reasoning", ""))

        return {
            "sufficient": sufficient,
            "confidence": confidence,
            "missing": missing,
            "reasoning": reasoning,
        }
