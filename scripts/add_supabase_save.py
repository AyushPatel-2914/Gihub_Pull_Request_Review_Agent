with open('src/agent_core.py', 'r', encoding='utf-8') as f:
    content = f.read()

save_block = """
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

"""

# Find the return statement
target = '    return {\n        "pr_number": pr_number,'

if target in content:
    content = content.replace(target, save_block + target, 1)
    with open('src/agent_core.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS - save_to_db block added')
else:
    print('Return statement not found. Last 20 lines:')
    print('\n'.join(content.splitlines()[-20:]))