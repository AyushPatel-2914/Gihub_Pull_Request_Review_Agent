with open('.github/workflows/pr_review.yml', 'r', encoding='utf-8') as f:
    content = f.read()

# Add Supabase env vars after PINECONE_API_KEY line
old = "          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}"
new = """          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}

          # Supabase: for saving reviews to the dashboard
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}"""

if old in content:
    content = content.replace(old, new, 1)
    with open('.github/workflows/pr_review.yml', 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    print('SUCCESS - Supabase env vars added to workflow')
    print('has SUPABASE_URL:', 'SUPABASE_URL' in content)
    print('has SUPABASE_SERVICE_KEY:', 'SUPABASE_SERVICE_KEY' in content)
else:
    print('ERROR - could not find PINECONE line')