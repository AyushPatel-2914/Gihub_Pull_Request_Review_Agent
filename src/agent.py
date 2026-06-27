import os, sys, logging
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gh.gh_client import GitHubClient
from analysis.static_analyzer import analyze_changed_files
from llm.groq_client import GroqLLMClient
from ingestion.file_parser import parse_file
from ingestion.embedder import Embedder, SEARCH_DOCUMENT_PREFIX
from retrieval.pinecone_client import PineconeClient
from rag.retriever import SelfHealingRetriever
from ml.intent_router import classify_intent
from ml.reranker import CrossEncoderReranker
from ml.risk_scorer import score_pr_risk, format_risk_section
from ml.token_optimizer import TokenOptimizer

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
    missing = [f"  x {var}: {hint}" for var, hint in required.items() if not os.environ.get(var)]
    if missing:
        logger.error("Missing environment variables:\n%s", "\n".join(missing))
        sys.exit(1)
    logger.info("Environment validation passed")


def run_agent():
    logger.info("=" * 60)
    logger.info("PR Review Agent - Phase 1-5A Starting")
    logger.info("=" * 60)
    validate_environment()

    # ── Step 2: GitHub data ───────────────────────────────────
    logger.info("Step 2: Fetching PR data from GitHub...")
    gh = GitHubClient()
    pr_metadata = gh.get_pr_metadata()
    diff = gh.get_pr_diff()
    changed_files = gh.get_changed_files()
    logger.info("PR #%d | author=%s | files=%d | +%d/-%d",
                pr_metadata["number"], pr_metadata["author"],
                pr_metadata["changed_files"], pr_metadata["additions"],
                pr_metadata["deletions"])

    # ── Step 3: File contents ─────────────────────────────────
    logger.info("Step 3: Fetching file contents...")
    for file_info in changed_files:
        if file_info["status"] == "removed":
            file_info["content"] = ""
            continue
        file_info["content"] = gh.get_file_content(file_info["filename"])

    # ── Step 4: Static analysis ───────────────────────────────
    logger.info("Step 4: Running static analysis...")
    static_report = analyze_changed_files(changed_files)
    logger.info("Static analysis: %s", "issues found" if static_report else "clean")

    # ── Step 4-risk: Predictive Risk Scoring (Phase 5A) ───────
    logger.info("Step 4-risk: Computing PR risk score...")
    risk_result = score_pr_risk(pr_metadata, changed_files, diff)
    logger.info("Risk: %d%% (%s)", risk_result.score, risk_result.level)

    # ── Step 4a: Intent classification (Phase 3) ──────────────
    logger.info("Step 4a: Classifying PR intent...")
    intent_result = classify_intent(pr_metadata, diff, changed_files)
    logger.info("Intent: %s (%.0f%%)", intent_result.intent.value, intent_result.confidence * 100)

    # ── Step 4b-4e: RAG + re-ranking ──────────────────────────
    rag_context = ""
    rag_chunks = []
    try:
        logger.info("Step 4b: Updating Pinecone index for changed files...")
        pinecone_client = PineconeClient()
        for file_info in changed_files:
            if file_info["status"] in ("modified", "removed", "renamed"):
                pinecone_client.delete_by_filepath(file_info["filename"])

        embedder = Embedder()
        namespace = pinecone_client.namespace
        files_to_index = [f for f in changed_files
                          if f["status"] != "removed" and f.get("content", "").strip()]

        if files_to_index:
            logger.info("Step 4c: Re-indexing %d changed file(s)...", len(files_to_index))
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
                logger.info("Re-indexed %d chunks", len(all_new_chunks))

        logger.info("Step 4d: Running Self-Healing RAG retrieval...")
        retriever = SelfHealingRetriever(embedder=embedder, pinecone_client=pinecone_client)
        rag_result = retriever.retrieve(diff=diff, pr_metadata=pr_metadata)
        rag_chunks = rag_result.chunks
        logger.info("RAG complete | chunks=%d | healing_rounds=%d",
                    len(rag_chunks), rag_result.healing_rounds)

        if rag_chunks:
            logger.info("Step 4e: Re-ranking %d chunks...", len(rag_chunks))
            reranker = CrossEncoderReranker()
            rerank_query = pr_metadata["title"] + "\n" + "\n".join(
                l[1:] for l in diff.splitlines()
                if l.startswith("+") and not l.startswith("+++")
            )[:1000]
            rag_chunks = reranker.rerank(rerank_query, rag_chunks)
            logger.info("After re-ranking: %d chunks kept", len(rag_chunks))

    except Exception as e:
        logger.error("Phase 2/3 RAG pipeline failed: %s — continuing without RAG", e)
        rag_chunks = []

    # ── Step 4-token: Token optimization (Phase 5A) ───────────
    logger.info("Step 4-token: Optimizing token budget...")
    optimizer = TokenOptimizer()
    optimized = optimizer.optimize(
        diff=diff,
        rag_chunks=rag_chunks,
        static_report=static_report or "",
    )
    diff = optimized["diff"]
    rag_chunks = optimized["rag_chunks"]
    static_report = optimized["static_report"]
    if optimized["was_trimmed"]:
        logger.info("Content was trimmed to fit token budget")

    # Rebuild RAG context string from (possibly trimmed) chunks
    rag_context = _format_rag_context(rag_chunks)

    # ── Step 5: Structured review (Phase 3) ───────────────────
    logger.info("Step 5: Generating structured AI review via Groq...")
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
    logger.info("Structured review | inline=%d | verdict=%s",
                len(review_data.get("inline_comments", [])),
                review_data.get("verdict"))

    # ── Step 6: Post review with risk + inline comments ───────
    logger.info("Step 6: Posting review to PR #%d...", pr_metadata["number"])

    # Build the header with risk score + intent + RAG status
    header = _build_header(pr_metadata, intent_result, risk_result,
                           rag_context, len(rag_chunks),
                           optimized["was_trimmed"])

    # Inject risk section + header into the summary
    risk_section = format_risk_section(risk_result)
    review_data["summary"] = (
        header + "\n\n" + risk_section + "\n\n" + review_data.get("summary", "")
    )

    gh.post_review_with_inline_comments(review_data, commit_sha=pr_metadata["head_sha"])

    logger.info("=" * 60)
    logger.info("Agent complete | PR #%d | risk=%d%% | reviewed successfully",
                pr_metadata["number"], risk_result.score)
    logger.info("=" * 60)


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
                  chunk_count, was_trimmed):
    rag_status = f"✅ {chunk_count} chunks" if rag_context else "⚠️ Skipped"
    trim_note = " (trimmed)" if was_trimmed else ""
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
    )


if __name__ == "__main__":
    try:
        run_agent()
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        sys.exit(1)
