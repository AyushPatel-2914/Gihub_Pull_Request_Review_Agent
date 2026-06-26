import os, sys, logging
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gh.gh_client import GitHubClient
from analysis.static_analyzer import analyze_changed_files
from llm.groq_client import GroqLLMClient
from ingestion.file_parser import parse_file
from ingestion.embedder import Embedder
from retrieval.pinecone_client import PineconeClient
from rag.retriever import SelfHealingRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("agent")

def validate_environment():
    required = {"GITHUB_TOKEN": "Auto-injected by GitHub Actions", "GROQ_API_KEY": "Get free key at https://console.groq.com", "PINECONE_API_KEY": "Get free key at https://app.pinecone.io", "PR_NUMBER": "Auto-injected by GitHub Actions", "REPO_FULL_NAME": "Auto-injected by GitHub Actions"}
    missing = [f"  x {var}: {hint}" for var, hint in required.items() if not os.environ.get(var)]
    if missing:
        logger.error("Missing environment variables:\n%s", "\n".join(missing))
        sys.exit(1)
    logger.info("Environment validation passed")

def run_agent():
    logger.info("=" * 60)
    logger.info("PR Review Agent - Phase 1 + 2 Starting")
    logger.info("=" * 60)
    validate_environment()

    logger.info("Step 2: Fetching PR data from GitHub...")
    gh = GitHubClient()
    pr_metadata = gh.get_pr_metadata()
    diff = gh.get_pr_diff()
    changed_files = gh.get_changed_files()
    logger.info("PR #%d | author=%s | files=%d | +%d/-%d", pr_metadata["number"], pr_metadata["author"], pr_metadata["changed_files"], pr_metadata["additions"], pr_metadata["deletions"])

    logger.info("Step 3: Fetching file contents...")
    for file_info in changed_files:
        if file_info["status"] == "removed":
            file_info["content"] = ""
            continue
        file_info["content"] = gh.get_file_content(file_info["filename"])

    logger.info("Step 4: Running static analysis...")
    static_report = analyze_changed_files(changed_files)
    logger.info("Static analysis: %s", "issues found" if static_report else "clean")

    rag_context = ""
    try:
        logger.info("Step 4b: Updating Pinecone index for changed files...")
        pinecone_client = PineconeClient()
        for file_info in changed_files:
            if file_info["status"] in ("modified", "removed", "renamed"):
                logger.info("Deleting old vectors for: %s", file_info["filename"])
                pinecone_client.delete_by_filepath(file_info["filename"])

        embedder = Embedder()
        namespace = pinecone_client.namespace
        files_to_index = [f for f in changed_files if f["status"] != "removed" and f.get("content", "").strip()]

        if files_to_index:
            logger.info("Step 4c: Re-indexing %d changed file(s)...", len(files_to_index))
            all_new_chunks = []
            for file_info in files_to_index:
                chunks = parse_file(filepath=file_info["filename"], content=file_info["content"], namespace=namespace)
                all_new_chunks.extend(chunks)
            if all_new_chunks:
                from ingestion.embedder import SEARCH_DOCUMENT_PREFIX
                texts = [SEARCH_DOCUMENT_PREFIX + c.content for c in all_new_chunks]
                vectors_raw = embedder.model.encode(texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
                pinecone_vectors = [(c.chunk_id, v.tolist(), c.to_pinecone_metadata()) for c, v in zip(all_new_chunks, vectors_raw)]
                pinecone_client.upsert_vectors(pinecone_vectors, namespace=namespace)
                logger.info("Re-indexed %d chunks", len(all_new_chunks))

        logger.info("Step 4d: Running Self-Healing RAG retrieval...")
        retriever = SelfHealingRetriever(embedder=embedder, pinecone_client=pinecone_client)
        rag_result = retriever.retrieve(diff=diff, pr_metadata=pr_metadata)
        logger.info("RAG complete | chunks=%d | healing_rounds=%d | was_healed=%s", len(rag_result.chunks), rag_result.healing_rounds, rag_result.was_healed)
        rag_context = rag_result.context_string
        if rag_result.was_healed:
            static_report += f"\n\n> Self-Healing RAG ran {rag_result.healing_rounds} healing round(s)."

    except Exception as e:
        logger.error("Phase 2 RAG pipeline failed: %s - falling back to Phase 1", e)
        rag_context = ""

    logger.info("Step 5: Generating AI review via Groq (Llama 3)...")
    llm = GroqLLMClient()
    combined_context = ""
    if static_report:
        combined_context += static_report + "\n\n"
    if rag_context:
        combined_context += rag_context
    review_markdown = llm.generate_review(pr_metadata=pr_metadata, diff=diff, static_analysis_report=combined_context)
    logger.info("Review generated | chars=%d", len(review_markdown))

    logger.info("Step 6: Posting review to PR #%d...", pr_metadata["number"])
    rag_status = "Active" if rag_context else "Skipped (run index_repo.py first)"
    header = (
        f"<!-- pr-review-agent -->\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| **PR** | #{pr_metadata['number']}: {pr_metadata['title']} |\n"
        f"| **Author** | `{pr_metadata['author']}` |\n"
        f"| **Branch** | `{pr_metadata['head_branch']}` to `{pr_metadata['base_branch']}` |\n"
        f"| **Scope** | +{pr_metadata['additions']} / -{pr_metadata['deletions']} lines, {pr_metadata['changed_files']} file(s) |\n"
        f"| **Review Engine** | Llama 3.3 70B via Groq |\n"
        f"| **RAG Context** | {rag_status} |\n"
    )
    gh.post_pr_comment(header + "\n\n" + review_markdown)
    logger.info("=" * 60)
    logger.info("Agent complete | PR #%d reviewed successfully", pr_metadata["number"])
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        run_agent()
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        sys.exit(1)