import os

# Read current agent_core.py
with open('src/agent_core.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Patch 1: Add CompletenessCritic import after SelfHealingRetriever import
old_import = "from rag.retriever import SelfHealingRetriever"
new_import = """from rag.retriever import SelfHealingRetriever
from rag.completeness_critic import CompletenessCritic"""
content = content.replace(old_import, new_import, 1)

# Patch 2: Initialize critique vars before try block
old_init = "    rag_chunks = []\n"
new_init = """    rag_chunks = []
    critique_ran = False
    critique_sufficient = True
"""
content = content.replace(old_init, new_init, 1)

# Patch 3: Wire critic into retriever
old_retriever = """        retriever = SelfHealingRetriever(
            embedder=embedder,
            pinecone_client=pinecone_client,
        )
        rag_result = retriever.retrieve(diff=diff, pr_metadata=pr_metadata)
        rag_chunks = rag_result.chunks"""
new_retriever = """        critic = CompletenessCritic(GroqLLMClient())
        retriever = SelfHealingRetriever(
            embedder=embedder,
            pinecone_client=pinecone_client,
            critic=critic,
        )
        rag_result = retriever.retrieve(diff=diff, pr_metadata=pr_metadata)
        rag_chunks = rag_result.chunks
        critique_ran = rag_result.critique_ran
        critique_sufficient = rag_result.critique_sufficient"""
content = content.replace(old_retriever, new_retriever, 1)

# Patch 4: Pass critique args to header
old_header_call = """    header = _build_header(pr_metadata, intent_result, risk_result,
                           rag_context, len(rag_chunks), optimized["was_trimmed"])"""
new_header_call = """    header = _build_header(pr_metadata, intent_result, risk_result,
                           rag_context, len(rag_chunks), optimized["was_trimmed"],
                           critique_ran, critique_sufficient)"""
content = content.replace(old_header_call, new_header_call, 1)

# Patch 5: Update header function signature and add critique row
old_header_fn = """def _build_header(pr_metadata, intent_result, risk_result, rag_context,
                  chunk_count, was_trimmed):
    rag_status = f"✅ {chunk_count} chunks" if rag_context else "⚠️ Skipped"
    trim_note = " (trimmed)" if was_trimmed else ""
    return (
        f"<!-- pr-review-agent -->\\n"
        f"| Metric | Value |\\n"
        f"|--------|-------|\\n"
        f"| **PR** | #{pr_metadata['number']}: {pr_metadata['title']} |\\n"
        f"| **Author** | `{pr_metadata['author']}` |\\n"
        f"| **Risk Score** | {risk_result.emoji} **{risk_result.score}% {risk_result.level}** |\\n"
        f"| **Intent** | {intent_result.intent.value} ({intent_result.confidence:.0%}) |\\n"
        f"| **Scope** | +{pr_metadata['additions']} / -{pr_metadata['deletions']} lines, "
        f"{pr_metadata['changed_files']} file(s) |\\n"
        f"| **Review Engine** | Llama 3.3 70B via Groq |\\n"
        f"| **RAG Context** | {rag_status} (re-ranked){trim_note} |\\n"
    )"""
new_header_fn = """def _build_header(pr_metadata, intent_result, risk_result, rag_context,
                  chunk_count, was_trimmed, critique_ran=False, critique_sufficient=True):
    rag_status = f"✅ {chunk_count} chunks" if rag_context else "⚠️ Skipped"
    trim_note = " (trimmed)" if was_trimmed else ""
    if critique_ran:
        critique_status = "✅ Sufficient" if critique_sufficient else "🔄 Healed (was insufficient)"
    else:
        critique_status = "— Not run"
    return (
        f"<!-- pr-review-agent -->\\n"
        f"| Metric | Value |\\n"
        f"|--------|-------|\\n"
        f"| **PR** | #{pr_metadata['number']}: {pr_metadata['title']} |\\n"
        f"| **Author** | `{pr_metadata['author']}` |\\n"
        f"| **Risk Score** | {risk_result.emoji} **{risk_result.score}% {risk_result.level}** |\\n"
        f"| **Intent** | {intent_result.intent.value} ({intent_result.confidence:.0%}) |\\n"
        f"| **Scope** | +{pr_metadata['additions']} / -{pr_metadata['deletions']} lines, "
        f"{pr_metadata['changed_files']} file(s) |\\n"
        f"| **Review Engine** | Llama 3.3 70B via Groq |\\n"
        f"| **RAG Context** | {rag_status} (re-ranked){trim_note} |\\n"
        f"| **Completeness Critique** | {critique_status} |\\n"
    )"""
content = content.replace(old_header_fn, new_header_fn, 1)

with open('src/agent_core.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched agent_core.py")
print("has CompletenessCritic:", "CompletenessCritic" in content)
print("has critique_ran:", "critique_ran" in content)
print("has Completeness Critique:", "Completeness Critique" in content)
print("chars:", len(content))