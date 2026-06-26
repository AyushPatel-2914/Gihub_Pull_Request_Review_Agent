"""
src/github/gh_client.py
─────────────────────────────────────────────────────────────
GitHub API Client
Phase 1 responsibility:
  - Authenticate with GitHub using GITHUB_TOKEN
  - Fetch the full unified diff for the PR
  - Fetch the list of changed files with metadata
  - Post a markdown summary comment to the PR thread

Later phases will extend this to post inline review comments.
─────────────────────────────────────────────────────────────
"""

import os
import logging
from github import Github, GithubException

logger = logging.getLogger(__name__)


class GitHubClient:
    """
    Wraps PyGithub to expose only the operations the agent needs.
    All credentials come from environment variables — never hardcoded.
    """

    def __init__(self):
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise EnvironmentError(
                "GITHUB_TOKEN is not set. "
                "In GitHub Actions it is auto-injected. "
                "For local dev, set it in your .env file."
            )

        self.client = Github(token)
        repo_name = os.environ.get("REPO_FULL_NAME")
        if not repo_name:
            raise EnvironmentError("REPO_FULL_NAME environment variable is not set.")

        self.repo = self.client.get_repo(repo_name)
        self.pr_number = int(os.environ.get("PR_NUMBER", 0))
        self.pr = self.repo.get_pull(self.pr_number)

        logger.info(
            "GitHubClient initialised | repo=%s | PR=#%d",
            repo_name,
            self.pr_number,
        )

    # ──────────────────────────────────────────────────────────
    # PR METADATA
    # ──────────────────────────────────────────────────────────

    def get_pr_metadata(self) -> dict:
        """
        Returns a flat dictionary of PR metadata that the agent
        uses throughout its pipeline.

        Fields returned:
          - number        : PR number (int)
          - title         : PR title string
          - body          : PR description / body text
          - author        : GitHub username of the PR author
          - base_branch   : branch being merged INTO (e.g. main)
          - head_branch   : branch being merged FROM (feature branch)
          - head_sha      : latest commit SHA on the PR branch
          - base_sha      : commit SHA of the base branch tip
          - additions     : total lines added across all files
          - deletions     : total lines deleted across all files
          - changed_files : count of files changed
        """
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

    def get_pr_diff(self) -> str:
        """
        Fetches the raw unified diff for the entire PR.

        The unified diff format looks like:
          diff --git a/src/foo.py b/src/foo.py
          --- a/src/foo.py
          +++ b/src/foo.py
          @@ -10,6 +10,8 @@
          -old line
          +new line

        This raw string is passed directly into the LLM in Phase 1,
        and used to identify changed files for vector DB updates in Phase 2.

        Returns:
          str: full unified diff, or empty string if fetch fails.
        """
        try:
            # PyGithub exposes the raw diff via the comparison object
            comparison = self.repo.compare(
                self.pr.base.sha, self.pr.head.sha
            )

            # Build the diff manually from file patches
            # (PyGithub does not expose one raw .diff string directly)
            diff_parts = []
            for file in comparison.files:
                if file.patch:  # binary files have no patch
                    diff_parts.append(
                        f"diff --git a/{file.filename} b/{file.filename}\n"
                        f"{file.patch}"
                    )

            full_diff = "\n".join(diff_parts)
            logger.info(
                "Fetched diff | files=%d | characters=%d",
                len(diff_parts),
                len(full_diff),
            )
            return full_diff

        except GithubException as e:
            logger.error("Failed to fetch PR diff: %s", e)
            return ""

    def get_changed_files(self) -> list[dict]:
        """
        Returns structured metadata for every file changed in the PR.

        Each dict contains:
          - filename    : relative path from repo root
          - status      : 'added' | 'modified' | 'removed' | 'renamed'
          - additions   : lines added in this file
          - deletions   : lines deleted in this file
          - patch       : the raw unified diff patch for this file only

        Used in Phase 2 to decide which Pinecone vectors to
        delete and re-index.
        """
        try:
            comparison = self.repo.compare(
                self.pr.base.sha, self.pr.head.sha
            )
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

    def get_file_content(self, filepath: str) -> str:
        """
        Fetches the current content of a file from the HEAD commit
        of the PR branch. Used in Phase 2 to re-index changed files.

        Args:
          filepath: relative path from repo root (e.g. 'src/utils.py')

        Returns:
          str: decoded file content, or empty string if not found.
        """
        try:
            content_file = self.repo.get_contents(
                filepath, ref=self.pr.head.sha
            )
            # content_file.decoded_content returns bytes
            return content_file.decoded_content.decode("utf-8", errors="replace")

        except GithubException as e:
            logger.warning("Could not fetch file content for %s: %s", filepath, e)
            return ""

    # ──────────────────────────────────────────────────────────
    # POSTING FEEDBACK
    # ──────────────────────────────────────────────────────────

    def post_pr_comment(self, body: str) -> None:
        """
        Posts a top-level markdown comment to the PR thread.
        This is the Phase 1 feedback delivery mechanism.

        In Phase 3 this will be supplemented with inline comments
        posted directly on specific diff lines.

        Args:
          body: Markdown-formatted string to post as the comment.
        """
        try:
            self.pr.create_issue_comment(body)
            logger.info("Posted PR comment successfully (PR #%d)", self.pr_number)
        except GithubException as e:
            logger.error("Failed to post PR comment: %s", e)

    def post_inline_comment(
        self,
        commit_sha: str,
        filepath: str,
        line: int,
        body: str,
    ) -> None:
        """
        Posts an inline review comment on a specific line of the diff.
        Reserved for Phase 3 — stubbed here so imports don't break.

        Args:
          commit_sha : the HEAD commit SHA of the PR
          filepath   : relative file path (e.g. 'src/foo.py')
          line       : line number in the NEW file version
          body       : markdown comment body
        """
        try:
            commit = self.repo.get_commit(commit_sha)
            self.pr.create_review_comment(
                body=body,
                commit=commit,
                path=filepath,
                line=line,
            )
            logger.info(
                "Posted inline comment | file=%s | line=%d", filepath, line
            )
        except GithubException as e:
            logger.error(
                "Failed to post inline comment on %s:%d — %s", filepath, line, e
            )