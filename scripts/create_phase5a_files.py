import os
src = r'C:\Users\ASUS\Downloads'

mapping = {
    'agent_core.py': 'src/agent_core.py',
    'agent.py': 'src/agent.py',
    'webhook_server.py': 'src/server/webhook_server.py',
    'Dockerfile': 'Dockerfile',
}

for fname, dest in mapping.items():
    with open(os.path.join(src, fname), encoding='utf-8') as f:
        content = f.read()
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Written: {dest}")

# server needs an __init__.py
open('src/server/__init__.py', 'w').close()
print("Done!")