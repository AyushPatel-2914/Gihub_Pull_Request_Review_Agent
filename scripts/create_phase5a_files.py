import os
src = r'C:\Users\ASUS\Downloads'  # adjust if files are elsewhere

mapping = {
    'risk_scorer.py': 'src/ml/risk_scorer.py',
    'token_optimizer.py': 'src/ml/token_optimizer.py',
    'agent.py': 'src/agent.py',
    'train_risk_model.py': 'scripts/train_risk_model.py',
}

for fname, dest in mapping.items():
    with open(os.path.join(src, fname), encoding='utf-8') as f:
        content = f.read()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Written: {dest} ({len(content)} chars)")

print("Done!")