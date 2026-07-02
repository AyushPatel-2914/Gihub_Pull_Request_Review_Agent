"""
src/db/supabase_writer.py
─────────────────────────────────────────────────────────────
Supabase Writer — Persists Every Review to the Dashboard DB

After the agent generates a review, this module saves it to
Supabase so the dashboard can display it.

Called at the end of agent_core.run_review() with the full
review data — review_data, risk_result, intent_result, etc.

Required env var: SUPABASE_URL and SUPABASE_SERVICE_KEY
These are added as GitHub Actions secrets.

If Supabase is not configured, this module logs a warning
and returns silently — the review is still posted to GitHub,
the dashboard just won't have a record of it.
─────────────────────────────────────────────────────────────
"""

import logging
import os

logger = logging.getLogger(__name__)


def save_review(
    pr_metadata: dict,
    review_data: dict,
    risk_result,
    intent_result,
    rag_result_chunks: list,
    critique_ran: bool,
    critique_sufficient: bool,
    was_healed: bool,
) -> str | None:
    """
    Saves a completed review to Supabase.

    Returns the UUID of the created review row, or None on failure.
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        logger.info(
            "SUPABASE_URL or SUPABASE_SERVICE_KEY not set — "
            "skipping dashboard persistence. Review posted to GitHub only."
        )
        return None

    try:
        from supabase import create_client
        db = create_client(supabase_url, supabase_key)

        # ── Insert PR review ──────────────────────────────────
        review_row = {
            "repo_full_name": pr_metadata.get("repo_full_name",
                             os.environ.get("REPO_FULL_NAME", "")),
            "pr_number": pr_metadata["number"],
            "pr_title": pr_metadata["title"],
            "pr_author": pr_metadata["author"],
            "head_sha": pr_metadata.get("head_sha"),
            "base_branch": pr_metadata.get("base_branch"),
            "head_branch": pr_metadata.get("head_branch"),
            "additions": pr_metadata.get("additions", 0),
            "deletions": pr_metadata.get("deletions", 0),
            "changed_files": pr_metadata.get("changed_files", 0),
            "risk_score": risk_result.score,
            "risk_level": risk_result.level,
            "intent": intent_result.intent.value,
            "intent_confidence": intent_result.confidence,
            "verdict": review_data.get("verdict", "NEEDS_DISCUSSION"),
            "summary": review_data.get("summary", ""),
            "rag_active": len(rag_result_chunks) > 0,
            "critique_ran": critique_ran,
            "critique_sufficient": critique_sufficient,
            "was_healed": was_healed,
        }

        result = db.table("pr_reviews").upsert(
            review_row,
            on_conflict="repo_full_name,pr_number,head_sha",
        ).execute()

        if not result.data:
            logger.error("Supabase upsert returned no data")
            return None

        review_id = result.data[0]["id"]
        logger.info("Review saved to Supabase | id=%s", review_id)

        # ── Insert inline comments ────────────────────────────
        inline_rows = [
            {
                "review_id": review_id,
                "filepath": c["file"],
                "line_number": c.get("line"),
                "severity": c.get("severity"),
                "comment": c["comment"],
            }
            for c in review_data.get("inline_comments", [])
        ]
        if inline_rows:
            db.table("inline_comments").insert(inline_rows).execute()
            logger.info("Saved %d inline comments", len(inline_rows))

        # ── Insert general comments ───────────────────────────
        general_rows = [
            {"review_id": review_id, "comment": c}
            for c in review_data.get("general_comments", [])
        ]
        if general_rows:
            db.table("general_comments").insert(general_rows).execute()

        # ── Insert risk factors ───────────────────────────────
        factor_rows = [
            {"review_id": review_id, "factor": f}
            for f in risk_result.factors
            if not f.startswith("✓")  # exclude "tests reduce risk" positive
        ]
        if factor_rows:
            db.table("risk_factors").insert(factor_rows).execute()

        return review_id

    except Exception as e:
        logger.error("Failed to save review to Supabase: %s", e)
        return None
