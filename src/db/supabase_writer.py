import logging
import os

logger = logging.getLogger(__name__)


def save_review(pr_metadata, review_data, risk_result, intent_result,
                rag_result_chunks, critique_ran, critique_sufficient, was_healed):
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_SERVICE_KEY')

    if not supabase_url or not supabase_key:
        logger.info('Supabase not configured - skipping dashboard save')
        return None

    try:
        from supabase import create_client
        db = create_client(supabase_url, supabase_key)

        repo_full_name = pr_metadata.get('repo_full_name', os.environ.get('REPO_FULL_NAME', ''))

        review_row = {
            'repo_full_name': repo_full_name,
            'pr_number': pr_metadata['number'],
            'pr_title': pr_metadata['title'],
            'pr_author': pr_metadata['author'],
            'head_sha': pr_metadata.get('head_sha'),
            'base_branch': pr_metadata.get('base_branch'),
            'head_branch': pr_metadata.get('head_branch'),
            'additions': pr_metadata.get('additions', 0),
            'deletions': pr_metadata.get('deletions', 0),
            'changed_files': pr_metadata.get('changed_files', 0),
            'risk_score': risk_result.score,
            'risk_level': risk_result.level,
            'intent': intent_result.intent.value,
            'intent_confidence': intent_result.confidence,
            'verdict': review_data.get('verdict', 'NEEDS_DISCUSSION'),
            'summary': review_data.get('summary', ''),
            'rag_active': len(rag_result_chunks) > 0,
            'critique_ran': critique_ran,
            'critique_sufficient': critique_sufficient,
            'was_healed': was_healed,
        }

        result = db.table('pr_reviews').upsert(
            review_row, on_conflict='repo_full_name,pr_number,head_sha'
        ).execute()

        if not result.data:
            logger.error('Supabase upsert returned no data')
            return None

        review_id = result.data[0]['id']
        logger.info('Review saved to Supabase | id=%s', review_id)

        inline_rows = [
            {
                'review_id': review_id,
                'filepath': c['file'],
                'line_number': c.get('line'),
                'severity': c.get('severity'),
                'comment': c['comment'],
            }
            for c in review_data.get('inline_comments', [])
        ]
        if inline_rows:
            db.table('inline_comments').insert(inline_rows).execute()

        general_rows = [
            {'review_id': review_id, 'comment': c}
            for c in review_data.get('general_comments', [])
        ]
        if general_rows:
            db.table('general_comments').insert(general_rows).execute()

        factor_rows = [
            {'review_id': review_id, 'factor': f}
            for f in risk_result.factors
        ]
        if factor_rows:
            db.table('risk_factors').insert(factor_rows).execute()

        return review_id

    except Exception as e:
        logger.error('Failed to save review to Supabase: %s', e)
        return None
