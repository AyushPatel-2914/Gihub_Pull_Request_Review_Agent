"""
src/agent_core.py
─────────────────────────────────────────────────────────────
Agent Core — Shared Review Pipeline

WHY THIS EXISTS:
  Phase 1-5A's agent.py reads PR info from environment variables
  (GitHub Actions injects them). The FastAPI server (Phase 5B)
  receives PR info from a webhook payload instead.

  Rather than duplicate the whole pipeline, we extract the core
  review logic into run_review() which takes EXPLICIT parameters.
  Both entry points call it:

    GitHub Actions → agent.py → reads env → run_review(...)
    Webhook        → server   → reads payload → run_review(...)

  This keeps one source of truth for the review pipeline.
─────────────────────────────────────────────────────────────
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gh.gh_client import GitHubClient
from analysis.static_analyzer import analyze_changed_files
from llm.groq_client import GroqLLMClient
from ingestion.file_parser import parse_file
from ingestion.embedder import Embedder, SEARCH_DOCUMENT_PREFIX
from retrieval.pinecone_client import PineconeClient
from rag.retriever import SelfHealingRetriever
from rag.completeness_critic import CompletenessCritic
from ml.intent_router import classify_intent
from ml.reranker import CrossEncoderReranker
from ml.risk_scorer import score_pr_risk, format_risk_section
from ml.token_optimizer import TokenOptimizer

logger = logging.getLogger("agent_core")

# ── Shared singletons ──────────────────────────────────────────
# In webhook/server mode, loading the embedding + reranker models
# on every request is wasteful. We cache them across requests.
# In GitHub Actions mode (one-shot), these load once anyway.
_EMBEDDER = None
_RERANKER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoderReranker()
    return _RERANKER


def run_review(repo_full_name: str, pr_number: int) -> dict:
    """
    Runs the complete review pipeline for a single PR.

    This is the SINGLE SOURCE OF TRUTH for the review logic,
    callable from both GitHub Actions and the webhook server.

    Args:
      repo_full_name : "owner/reponame"
      pr_number      : the PR number to review

    Returns:
      dict with summary stats: {risk_score, intent, inline_count, verdict}

    Required env vars (must be set before calling):
      GITHUB_TOKEN, GROQ_API_KEY, PINECONE_API_KEY
    The function sets REPO_FULL_NAME and PR_NUMBER itself so the
    downstream clients pick them up.
    """
    # The existing clients read these from the environment.
    # We set them explicitly so the same code works in both modes.
    os.environ["REPO_FULL_NAME"] = repo_full_name
    os.environ["PR_NUMBER"] = str(pr_number)

    logger.info("=" * 60)
    logger.info("Reviewing PR #%d in %s", pr_number, repo_full_name)
    logger.info("=" * 60)

    # ── Fetch PR data ─────────────────────────────────────────
    gh = GitHubClient()
    pr_metadata = gh.get_pr_metadata()
    diff = gh.get_pr_diff()
    changed_files = gh.get_changed_files()

    for file_info in changed_files:
        if file_info["status"] == "removed":
            file_info["content"] = ""
        else:
            file_info["content"] = gh.get_file_content(file_info["filename"])

    # ── Static analysis ───────────────────────────────────────
    static_report = analyze_changed_files(changed_files)

    # ── Risk scoring (Phase 5A) ───────────────────────────────
    risk_result = score_pr_risk(pr_metadata, changed_files, diff)
    logger.info("Risk: %d%% (%s)", risk_result.score, risk_result.level)

    # ── Intent classification (Phase 3) ───────────────────────
    intent_result = classify_intent(pr_metadata, diff, changed_files)
    logger.info("Intent: %s (%.0f%%)", intent_result.intent.value,
                intent_result.confidence * 100)

    # ── RAG + re-ranking (Phase 2 + 3) ────────────────────────
    rag_chunks = []
    critique_ran = False
    critique_sufficient = True
    try:
        pinecone_client = PineconeClient()
        for file_info in changed_files:
            if file_info["status"] in ("modified", "removed", "renamed"):
                pinecone_client.delete_by_filepath(file_info["filename"])

        embedder = _get_embedder()
        namespace = pinecone_client.namespace
        files_to_index = [f for f in changed_files
                          if f["status"] != "removed" and f.get("content", "").strip()]

        if files_to_index:
            all_new_chunks = []
            for file_info in files_to_index:
                all_new_chunks.extend(parse_file(
                    filepath=file_info["filename"],
                    content=file_info["content"],
                    namespace=namespace,
                ))
            if all_new_chunks:
                texts = [SEARCH_DOCUMENT_PREFIX + c.content for c in all_new_chunks]
                vectors_raw = embedder.model.encode(
                    texts, batch_size=8, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True,
                )
                pinecone_vectors = [
                    (c.chunk_id, v.tolist(), c.to_pinecone_metadata())
                    for c, v in zip(all_new_chunks, vectors_raw)
                ]
                pinecone_client.upsert_vectors(pinecone_vectors, namespace=namespace)

        # Phase 4: build the LLM completeness critic and pass it
        # to the retriever so it runs a semantic sufficiency check
        # after string-based healing.
        critic = CompletenessCritic(GroqLLMClient())
        retriever = SelfHealingRetriever(
            embedder=embedder,
            pinecone_client=pinecone_client,
            critic=critic,
        )
        rag_result = retriever.retrieve(diff=diff, pr_metadata=pr_metadata)
        rag_chunks = rag_result.chunks
        # Phase 4: capture critique diagnostics for the PR comment
        critique_ran = rag_result.critique_ran
        critique_sufficient = rag_result.critique_sufficient

        if rag_chunks:
            reranker = _get_reranker()
            rerank_query = pr_metadata["title"] + "\n" + "\n".join(
                l[1:] for l in diff.splitlines()
                if l.startswith("+") and not l.startswith("+++")
            )[:1000]
            rag_chunks = reranker.rerank(rerank_query, rag_chunks)

    except Exception as e:
        logger.error("RAG pipeline failed: %s — continuing without RAG", e)
        rag_chunks = []

    # ── Token optimization (Phase 5A) ─────────────────────────
    optimizer = TokenOptimizer()
    optimized = optimizer.optimize(
        diff=diff, rag_chunks=rag_chunks, static_report=static_report or "",
    )
    diff = optimized["diff"]
    rag_chunks = optimized["rag_chunks"]
    static_report = optimized["static_report"]
    rag_context = _format_rag_context(rag_chunks)

    # ── Structured review (Phase 3) ───────────────────────────
    llm = GroqLLMClient()
    combined_context = ""
    if static_report:
        combined_context += static_report + "\n\n"
    if rag_context:
        combined_context += rag_context

    review_data = llm.generate_structured_review(
        pr_metadata=pr_metadata,
        diff=diff,
        context_report=combined_context,
        intent_focus=intent_result.review_focus,
    )

    # ── Post review ───────────────────────────────────────────
    header = _build_header(pr_metadata, intent_result, risk_result,
                           rag_context, len(rag_chunks), optimized["was_trimmed"],
                           critique_ran, critique_sufficient)
    risk_section = format_risk_section(risk_result)
    review_data["summary"] = (
        header + "\n\n" + risk_section + "\n\n" + review_data.get("summary", "")
    )

    gh.post_review_with_inline_comments(review_data, commit_sha=pr_metadata["head_sha"])

    logger.info("Review posted | PR #%d | risk=%d%%", pr_number, risk_result.score)


    # Save to dashboard DB
    try:
        from db.supabase_writer import save_review as save_to_db
        save_to_db(
            pr_metadata={**pr_metadata, 'repo_full_name': repo_full_name},
            review_data=review_data,
            risk_result=risk_result,
            intent_result=intent_result,
            rag_result_chunks=rag_chunks,
            critique_ran=critique_ran,
            critique_sufficient=critique_sufficient,
            was_healed=False,
        )
    except Exception as e:
        logger.error('Dashboard save failed: %s', e)

    return {
        "pr_number": pr_number,
        "repo": repo_full_name,
        "risk_score": risk_result.score,
        "risk_level": risk_result.level,
        "intent": intent_result.intent.value,
        "inline_comments": len(review_data.get("inline_comments", [])),
        "verdict": review_data.get("verdict"),
    }


def _format_rag_context(chunks):
    if not chunks:
        return ""
    parts = ["## Relevant Codebase Context (RAG + Re-ranked)\n"]
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        filepath = meta.get("filepath", "unknown")
        symbol = meta.get("symbol_name", "")
        content = meta.get("content", "")
        score = chunk.get("rerank_score", chunk.get("score", 0))
        header = f"### [{i}] `{filepath}`"
        if symbol:
            header += f" — `{symbol}`"
        header += f" (relevance={score:.2f})"
        parts.append(header)
        parts.append(f"```\n{content}\n```\n")
    return "\n".join(parts)


def _build_header(pr_metadata, intent_result, risk_result, rag_context,
                  chunk_count, was_trimmed, critique_ran=False, critique_sufficient=True):
    rag_status = f"✅ {chunk_count} chunks" if rag_context else "⚠️ Skipped"
    trim_note = " (trimmed)" if was_trimmed else ""

    # Phase 4: completeness critique status row
    if critique_ran:
        critique_status = "✅ Sufficient" if critique_sufficient else "🔄 Healed (was insufficient)"
    else:
        critique_status = "— Not run"

    return (
        f"<!-- pr-review-agent -->\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| **PR** | #{pr_metadata['number']}: {pr_metadata['title']} |\n"
        f"| **Author** | `{pr_metadata['author']}` |\n"
        f"| **Risk Score** | {risk_result.emoji} **{risk_result.score}% {risk_result.level}** |\n"
        f"| **Intent** | {intent_result.intent.value} ({intent_result.confidence:.0%}) |\n"
        f"| **Scope** | +{pr_metadata['additions']} / -{pr_metadata['deletions']} lines, "
        f"{pr_metadata['changed_files']} file(s) |\n"
        f"| **Review Engine** | Llama 3.3 70B via Groq |\n"
        f"| **RAG Context** | {rag_status} (re-ranked){trim_note} |\n"
        f"| **Completeness Critique** | {critique_status} |\n"
    )
