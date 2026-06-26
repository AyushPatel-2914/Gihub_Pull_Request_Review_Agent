import os
import logging
from github import Github, GithubException

logger = logging.getLogger(__name__)


class GitHubClient:
    def __init__(self):
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise EnvironmentError("GITHUB_TOKEN is not set.")
        self.client = Github(token)
        repo_name = os.environ.get("REPO_FULL_NAME")
        if not repo_name:
            raise EnvironmentError("REPO_FULL_NAME is not set.")
        self.repo = self.client.get_repo(repo_name)
        self.pr_number = int(os.environ.get("PR_NUMBER", 0))
        self.pr = self.repo.get_pull(self.pr_number)
        logger.info("GitHubClient initialised | repo=%s | PR=#%d", repo_name, self.pr_number)

    # ──────────────────────────────────────────────────────────
    # PR METADATA
    # ──────────────────────────────────────────────────────────

    def get_pr_metadata(self):
        return {
            "number": self.pr.number,
            "title": self.pr.title,
            "body": self.pr.body or "",
            "author": self.pr.user.login,
            "base_branch": self.pr.base.ref,
            "head_branch": self.pr.head.ref,
            "head_sha": self.pr.head.sha,
            "base_sha": self.pr.base.sha,
            "additions": self.pr.additions,
            "deletions": self.pr.deletions,
            "changed_files": self.pr.changed_files,
        }

    # ──────────────────────────────────────────────────────────
    # DIFF FETCHING
    # ──────────────────────────────────────────────────────────

    def get_pr_diff(self):
        try:
            comparison = self.repo.compare(self.pr.base.sha, self.pr.head.sha)
            diff_parts = []
            for file in comparison.files:
                if file.patch:
                    diff_parts.append(
                        f"diff --git a/{file.filename} b/{file.filename}\n{file.patch}"
                    )
            full_diff = "\n".join(diff_parts)
            logger.info("Fetched diff | files=%d | characters=%d", len(diff_parts), len(full_diff))
            return full_diff
        except GithubException as e:
            logger.error("Failed to fetch PR diff: %s", e)
            return ""

    def get_changed_files(self):
        try:
            comparison = self.repo.compare(self.pr.base.sha, self.pr.head.sha)
            files = []
            for f in comparison.files:
                files.append({
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "patch": f.patch or "",
                })
            logger.info("Changed files count: %d", len(files))
            return files
        except GithubException as e:
            logger.error("Failed to fetch changed files: %s", e)
            return []

    def get_file_content(self, filepath):
        try:
            content_file = self.repo.get_contents(filepath, ref=self.pr.head.sha)
            return content_file.decoded_content.decode("utf-8", errors="replace")
        except GithubException as e:
            logger.warning("Could not fetch file content for %s: %s", filepath, e)
            return ""

    # ──────────────────────────────────────────────────────────
    # POSTING FEEDBACK — Phase 1 (plain comment)
    # ──────────────────────────────────────────────────────────

    def post_pr_comment(self, body):
        try:
            self.pr.create_issue_comment(body)
            logger.info("Posted PR comment successfully (PR #%d)", self.pr_number)
        except GithubException as e:
            logger.error("Failed to post PR comment: %s", e)

    # ──────────────────────────────────────────────────────────
    # POSTING FEEDBACK — Phase 3 (inline review comments)
    # ──────────────────────────────────────────────────────────

    def post_review_with_inline_comments(self, review_data, commit_sha=None):
        """
        Posts a single PR review containing a summary body and
        inline comments on specific changed lines via GitHub's
        Review API. Any comment that can't be placed on a valid
        diff line is appended to the summary body instead.
        """
        commit_sha = commit_sha or self.pr.head.sha
        valid_positions = self._build_valid_comment_positions()

        sev_emoji = {"CRITICAL": "🚨", "SUGGESTION": "💡", "NITPICK": "🔍"}
        review_comments = []
        unplaceable = []

        for c in review_data.get("inline_comments", []):
            filename = c["file"]
            line = c["line"]
            severity = c.get("severity", "SUGGESTION")
            body = f"{sev_emoji.get(severity, '💬')} **{severity}**: {c['comment']}"

            if (filename, line) in valid_positions:
                review_comments.append({
                    "path": filename,
                    "line": line,
                    "side": "RIGHT",
                    "body": body,
                })
            else:
                unplaceable.append(f"- `{filename}:{line}` {body}")

        summary_body = self._format_review_summary(review_data, unplaceable)

        try:
            self.pr.create_review(
                body=summary_body,
                event="COMMENT",
                comments=review_comments,
            )
            logger.info(
                "Posted review | inline=%d | unplaceable=%d | verdict=%s",
                len(review_comments), len(unplaceable),
                review_data.get("verdict"),
            )
        except GithubException as e:
            logger.error("create_review failed: %s — falling back to issue comment", e)
            fallback = summary_body
            if review_comments:
                fallback += "\n\n## Inline Findings\n"
                for rc in review_comments:
                    fallback += f"- `{rc['path']}:{rc['line']}` {rc['body']}\n"
            self.post_pr_comment(fallback)

    def _build_valid_comment_positions(self):
        """
        Returns a set of (filename, line_number) that are valid
        inline comment targets — i.e. lines present in the PR diff.
        """
        valid = set()
        try:
            files = self.pr.get_files()
            for f in files:
                if not f.patch:
                    continue
                filename = f.filename
                new_line = 0
                for patch_line in f.patch.splitlines():
                    if patch_line.startswith("@@"):
                        try:
                            plus_part = patch_line.split("+")[1].split(" ")[0]
                            new_line = int(plus_part.split(",")[0]) - 1
                        except (IndexError, ValueError):
                            new_line = 0
                    elif patch_line.startswith("+"):
                        new_line += 1
                        valid.add((filename, new_line))
                    elif patch_line.startswith("-"):
                        pass
                    else:
                        new_line += 1
                        valid.add((filename, new_line))
        except Exception as e:
            logger.warning("Could not build valid comment positions: %s", e)
        return valid

    def _format_review_summary(self, review_data, unplaceable):
        """Formats the top-level review summary as markdown."""
        verdict = review_data.get("verdict", "NEEDS_DISCUSSION")
        verdict_emoji = {
            "APPROVE": "✅",
            "REQUEST_CHANGES": "🔴",
            "NEEDS_DISCUSSION": "🟡",
        }.get(verdict, "🟡")

        parts = ["## 🤖 Automated PR Review\n"]

        parts.append("### 📋 Summary")
        parts.append(review_data.get("summary", "_No summary._") + "\n")

        positives = review_data.get("positives", [])
        if positives:
            parts.append("### ✅ What's Good")
            for p in positives:
                parts.append(f"- {p}")
            parts.append("")

        general = review_data.get("general_comments", [])
        if general:
            parts.append("### 💬 General Comments")
            for g in general:
                parts.append(f"- {g}")
            parts.append("")

        if unplaceable:
            parts.append("### 📌 Additional Findings")
            parts.append("_These relate to code outside the changed lines:_")
            parts.extend(unplaceable)
            parts.append("")

        parts.append("### 📊 Overall Verdict")
        parts.append(f"{verdict_emoji} **{verdict.replace('_', ' ')}**\n")
        parts.append("---")
        parts.append("*Automated review — use human judgment before merging.*")

        return "\n".join(parts)
