import os

os.makedirs('src/ml', exist_ok=True)

# ── intent_router.py ──────────────────────────────────────────
with open('src/ml/intent_router.py', 'w', encoding='utf-8') as f:
    f.write(open(r'C:\Users\ASUS\Downloads\intent_router.py', encoding='utf-8').read())
print("intent_router.py written")

# ── reranker.py ───────────────────────────────────────────────
with open('src/ml/reranker.py', 'w', encoding='utf-8') as f:
    f.write(open(r'C:\Users\ASUS\Downloads\reranker.py', encoding='utf-8').read())
print("reranker.py written")

# ── groq_client.py ────────────────────────────────────────────
with open('src/llm/groq_client.py', 'w', encoding='utf-8') as f:
    f.write(open(r'C:\Users\ASUS\Downloads\groq_client.py', encoding='utf-8').read())
print("groq_client.py written")

# ── gh_client.py ──────────────────────────────────────────────
with open('src/gh/gh_client.py', 'w', encoding='utf-8') as f:
    f.write(open(r'C:\Users\ASUS\Downloads\gh_client.py', encoding='utf-8').read())
print("gh_client.py written")

# ── agent.py ──────────────────────────────────────────────────
with open('src/agent.py', 'w', encoding='utf-8') as f:
    f.write(open(r'C:\Users\ASUS\Downloads\agent.py', encoding='utf-8').read())
print("agent.py written")

print("\nAll Phase 3 files written successfully!")