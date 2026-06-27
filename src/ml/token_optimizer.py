"""
src/ml/token_optimizer.py
─────────────────────────────────────────────────────────────
Token Optimizer — Safety Guard Against Huge PRs

THE PROBLEM:
  A 10,000-line PR could blow past the LLM's context window,
  causing a crash or a massive bill. The agent must never crash
  on a large PR — it should gracefully trim to fit.

WHAT IT DOES:
  1. Counts tokens accurately using tiktoken
  2. Budgets the context window across sections (diff, RAG, static)
  3. Intelligently trims the LOWEST-value content first:
       - Keeps the full diff if possible (it's the core artifact)
       - Trims RAG context chunks from the bottom (least relevant)
       - Summarizes oversized diffs by keeping headers + key hunks
  4. Always leaves room for the model's response

TOKEN BUDGET (llama-3.3-70b on Groq, 128k context):
  We use a conservative working budget to stay fast and cheap.
─────────────────────────────────────────────────────────────
"""

import logging

logger = logging.getLogger(__name__)

# ── Token Budget Configuration ─────────────────────────────────
# Conservative budget — Groq free tier is rate-limited, and
# smaller prompts are faster and cheaper.
MAX_TOTAL_TOKENS = 20_000        # working budget for the whole prompt
RESERVED_FOR_RESPONSE = 4_096    # leave room for the LLM's review
RESERVED_FOR_SYSTEM = 1_000      # system prompt + metadata overhead

# Per-section budgets (of the remaining input budget)
DIFF_BUDGET_RATIO = 0.55         # diff gets the largest share
RAG_BUDGET_RATIO = 0.35          # RAG context
STATIC_BUDGET_RATIO = 0.10       # static analysis findings


# ──────────────────────────────────────────────────────────────
# TOKEN COUNTING
# ──────────────────────────────────────────────────────────────

class TokenCounter:
    """
    Wraps tiktoken for accurate token counting.
    Uses the cl100k_base encoding (close enough to Llama's tokenizer
    for budgeting purposes — both are BPE-based).
    """

    def __init__(self):
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self._available = True
        except Exception as e:
            logger.warning("tiktoken unavailable (%s) — using char estimate", e)
            self._available = False

    def count(self, text: str) -> int:
        """
        Returns the token count of a string.
        Falls back to a char-based estimate (~4 chars/token) if
        tiktoken isn't available.
        """
        if not text:
            return 0
        if self._available:
            return len(self.encoding.encode(text))
        # Fallback estimate
        return len(text) // 4


# ──────────────────────────────────────────────────────────────
# DIFF TRIMMING
# ──────────────────────────────────────────────────────────────

def trim_diff(diff: str, max_tokens: int, counter: TokenCounter) -> tuple[str, bool]:
    """
    Trims a diff to fit within a token budget, preserving the most
    important parts: file headers and the start of each file's changes.

    Strategy:
      - If it fits, return as-is.
      - If not, keep each file's header + as many hunks as fit,
        distributing the budget across files so no single huge file
        eats everything.

    Returns:
      (trimmed_diff, was_trimmed)
    """
    if counter.count(diff) <= max_tokens:
        return diff, False

    logger.info("Diff exceeds %d tokens — trimming", max_tokens)

    # Split the diff into per-file sections
    file_sections = []
    current = []
    for line in diff.splitlines():
        if line.startswith("diff --git") and current:
            file_sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        file_sections.append("\n".join(current))

    if not file_sections:
        # No file boundaries — hard truncate
        return _hard_truncate(diff, max_tokens, counter), True

    # Distribute budget evenly across files
    per_file_budget = max_tokens // len(file_sections)
    trimmed_parts = []

    for section in file_sections:
        if counter.count(section) <= per_file_budget:
            trimmed_parts.append(section)
        else:
            # Keep the header + as many lines as fit
            lines = section.splitlines()
            kept = []
            running = 0
            for line in lines:
                line_tokens = counter.count(line)
                if running + line_tokens > per_file_budget:
                    kept.append("... [file diff truncated to fit token budget] ...")
                    break
                kept.append(line)
                running += line_tokens
            trimmed_parts.append("\n".join(kept))

    trimmed = "\n".join(trimmed_parts)
    return trimmed, True


def _hard_truncate(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """Last-resort truncation by characters when no structure exists."""
    if not counter._available:
        return text[:max_tokens * 4] + "\n... [truncated] ..."
    tokens = counter.encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = counter.encoding.decode(tokens[:max_tokens])
    return truncated + "\n... [truncated] ..."


# ──────────────────────────────────────────────────────────────
# RAG CONTEXT TRIMMING
# ──────────────────────────────────────────────────────────────

def trim_rag_context(rag_chunks: list, max_tokens: int, counter: TokenCounter) -> tuple[list, bool]:
    """
    Trims RAG chunks to fit a token budget by dropping the
    lowest-ranked chunks first (the list is assumed sorted by
    relevance, highest first).

    Args:
      rag_chunks : list of chunk dicts (sorted best-first)
      max_tokens : token budget for all RAG context
      counter    : TokenCounter instance

    Returns:
      (kept_chunks, was_trimmed)
    """
    if not rag_chunks:
        return [], False

    kept = []
    running = 0
    trimmed = False

    for chunk in rag_chunks:
        content = chunk.get("metadata", {}).get("content", "")
        chunk_tokens = counter.count(content) + 50  # +50 for formatting overhead
        if running + chunk_tokens > max_tokens:
            trimmed = True
            break
        kept.append(chunk)
        running += chunk_tokens

    if trimmed:
        logger.info(
            "RAG context trimmed | kept %d/%d chunks to fit budget",
            len(kept), len(rag_chunks),
        )

    return kept, trimmed


# ──────────────────────────────────────────────────────────────
# MAIN OPTIMIZER
# ──────────────────────────────────────────────────────────────

class TokenOptimizer:
    """
    Orchestrates token budgeting across all prompt sections.
    Call optimize() before building the final LLM prompt.
    """

    def __init__(self):
        self.counter = TokenCounter()

    def get_input_budget(self) -> int:
        """Total tokens available for input (excludes response reserve)."""
        return MAX_TOTAL_TOKENS - RESERVED_FOR_RESPONSE - RESERVED_FOR_SYSTEM

    def optimize(self, diff: str, rag_chunks: list, static_report: str) -> dict:
        """
        Trims all sections to fit within the total token budget.

        Args:
          diff          : raw unified diff
          rag_chunks    : list of RAG chunk dicts (sorted best-first)
          static_report : static analysis report string

        Returns:
          dict with keys:
            - diff           : (possibly trimmed) diff
            - rag_chunks     : (possibly trimmed) chunk list
            - static_report  : (possibly trimmed) static report
            - was_trimmed    : bool, True if anything was trimmed
            - token_summary  : dict of token counts per section
        """
        input_budget = self.get_input_budget()

        diff_budget = int(input_budget * DIFF_BUDGET_RATIO)
        rag_budget = int(input_budget * RAG_BUDGET_RATIO)
        static_budget = int(input_budget * STATIC_BUDGET_RATIO)

        # Trim diff
        trimmed_diff, diff_trimmed = trim_diff(diff, diff_budget, self.counter)

        # Trim RAG context
        trimmed_chunks, rag_trimmed = trim_rag_context(
            rag_chunks, rag_budget, self.counter
        )

        # Trim static report (simple truncation — it's already concise)
        static_trimmed = False
        if self.counter.count(static_report) > static_budget:
            static_report = _hard_truncate(static_report, static_budget, self.counter)
            static_trimmed = True

        token_summary = {
            "diff_tokens": self.counter.count(trimmed_diff),
            "rag_chunks_kept": len(trimmed_chunks),
            "static_tokens": self.counter.count(static_report),
            "input_budget": input_budget,
        }

        was_trimmed = diff_trimmed or rag_trimmed or static_trimmed

        logger.info(
            "Token optimization | diff=%d tok | rag=%d chunks | static=%d tok | trimmed=%s",
            token_summary["diff_tokens"],
            token_summary["rag_chunks_kept"],
            token_summary["static_tokens"],
            was_trimmed,
        )

        return {
            "diff": trimmed_diff,
            "rag_chunks": trimmed_chunks,
            "static_report": static_report,
            "was_trimmed": was_trimmed,
            "token_summary": token_summary,
        }
