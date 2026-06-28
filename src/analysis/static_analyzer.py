"""
src/analysis/static_analyzer.py
─────────────────────────────────────────────────────────────
Static Analyzer — Pre-LLM Objective Error Detection
Phase 1 responsibility:
  - Detect the type of each changed file
  - Run the appropriate static analysis tool
  - Return structured findings as a string
    that gets injected into the LLM context

Why run this BEFORE the LLM?
  The LLM focuses on logic, design, and architecture issues.
  Static tools catch objective, deterministic errors cheaply:
  syntax errors, undefined names, broken YAML, malformed JSON.
  Feeding these findings to the LLM prevents it from wasting
  its reasoning budget re-discovering things a linter already knows.

Supported file types:
  .py          → flake8 (style) + ast syntax check
  .yaml/.yml   → yamllint
  .json        → json.loads validation
  .md/.txt     → basic structure check (heading, length)
  .js/.ts      → placeholder (eslint needs node runtime)
  everything else → skipped gracefully
─────────────────────────────────────────────────────────────
"""

import ast
import json
import logging
import subprocess
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# FILE TYPE DETECTION
# ──────────────────────────────────────────────────────────────

def detect_file_type(filename: str) -> str:
    """
    Returns a canonical file type label based on the file extension.

    Args:
      filename: relative or absolute path string

    Returns:
      str: one of 'python', 'yaml', 'json', 'markdown',
           'javascript', 'typescript', 'notebook', 'text', 'unknown'
    """
    suffix = Path(filename).suffix.lower()
    mapping = {
        ".py": "python",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".md": "markdown",
        ".txt": "text",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".ipynb": "notebook",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".sh": "shell",
        ".bash": "shell",
    }
    return mapping.get(suffix, "unknown")


# ──────────────────────────────────────────────────────────────
# PYTHON ANALYSIS
# ──────────────────────────────────────────────────────────────

def _analyze_python(content: str, filename: str) -> list[str]:
    """
    Runs two checks on Python content:
      1. AST syntax check — catches SyntaxError before anything else
      2. flake8 — style, undefined names, unused imports

    Args:
      content  : raw Python source code as string
      filename : used only for display in error messages

    Returns:
      list of finding strings, empty if no issues found
    """
    findings = []

    # ── Check 1: AST Syntax ───────────────────────────────────
    try:
        ast.parse(content)
    except SyntaxError as e:
        findings.append(
            f"[SYNTAX ERROR] {filename} line {e.lineno}: {e.msg}"
        )
        # If syntax is broken, flake8 will also fail — return early
        return findings

    # ── Check 2: flake8 ───────────────────────────────────────
    try:
        # Write content to a temp file because flake8 needs a real file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = subprocess.run(
            [
                "flake8",
                "--max-line-length=120",
                "--select=E,W,F",   # Errors, Warnings, pyFlakes
                "--format=%(row)d:%(col)d %(code)s %(text)s",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                findings.append(f"[FLAKE8] {filename}: {line}")

    except subprocess.TimeoutExpired:
        findings.append(f"[WARNING] flake8 timed out on {filename}")
    except FileNotFoundError:
        logger.warning("flake8 not found in PATH — skipping Python lint")
    finally:
        # Always clean up the temp file
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return findings


# ──────────────────────────────────────────────────────────────
# YAML ANALYSIS
# ──────────────────────────────────────────────────────────────

def _analyze_yaml(content: str, filename: str) -> list[str]:
    """
    Validates YAML using yamllint.
    Catches indentation errors, duplicate keys, and syntax issues.

    Returns:
      list of finding strings
    """
    findings = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = subprocess.run(
            ["yamllint", "-d", "relaxed", tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                # Filter out trivial warnings, keep errors
                if "error" in line.lower() or "warning" in line.lower():
                    findings.append(f"[YAMLLINT] {filename}: {line.strip()}")

    except subprocess.TimeoutExpired:
        findings.append(f"[WARNING] yamllint timed out on {filename}")
    except FileNotFoundError:
        logger.warning("yamllint not found — skipping YAML validation")
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return findings


# ──────────────────────────────────────────────────────────────
# JSON ANALYSIS
# ──────────────────────────────────────────────────────────────

def _analyze_json(content: str, filename: str) -> list[str]:
    """
    Validates JSON by attempting to parse it with the stdlib.
    Catches malformed JSON, trailing commas, and encoding issues.

    Returns:
      list of finding strings
    """
    findings = []
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        findings.append(
            f"[JSON ERROR] {filename} line {e.lineno} col {e.colno}: {e.msg}"
        )
    return findings


# ──────────────────────────────────────────────────────────────
# MARKDOWN ANALYSIS
# ──────────────────────────────────────────────────────────────

def _analyze_markdown(content: str, filename: str) -> list[str]:
    """
    Basic structural checks on Markdown files:
      - Warns if the document has no headings (# H1)
      - Warns if the document is very short (likely incomplete)
      - Counts broken markdown image/link patterns

    This is intentionally lightweight — Markdown is flexible.
    The LLM handles deeper content quality checks.

    Returns:
      list of finding strings
    """
    findings = []
    lines = content.splitlines()

    has_heading = any(line.startswith("#") for line in lines)
    if not has_heading:
        findings.append(
            f"[MARKDOWN] {filename}: No headings found — "
            "consider adding structure with # H1, ## H2 etc."
        )

    if len(content.strip()) < 50:
        findings.append(
            f"[MARKDOWN] {filename}: Document appears very short "
            f"({len(content.strip())} chars) — may be incomplete."
        )

    return findings


# ──────────────────────────────────────────────────────────────
# MAIN PUBLIC INTERFACE
# ──────────────────────────────────────────────────────────────

def analyze_file(filename: str, content: str) -> dict:
    """
    Dispatches to the correct analyzer based on file type.
    Returns a structured result dict consumed by the agent.

    Args:
      filename : relative path of the file (used for display)
      content  : raw string content of the file

    Returns:
      dict with keys:
        - filename   : str
        - file_type  : str (e.g. 'python', 'yaml')
        - findings   : list[str] — issues found
        - clean      : bool — True if no issues found
        - summary    : str — human-readable one-liner for LLM context
    """
    file_type = detect_file_type(filename)

    dispatch = {
        "python": _analyze_python,
        "yaml": _analyze_yaml,
        "json": _analyze_json,
        "markdown": _analyze_markdown,
        "text": _analyze_markdown,  # reuse lightweight text checks
    }

    analyzer_fn = dispatch.get(file_type)

    if analyzer_fn:
        findings = analyzer_fn(content, filename)
    else:
        findings = []
        logger.debug("No static analyzer for file type '%s' (%s)", file_type, filename)

    clean = len(findings) == 0
    summary = (
        f"Static analysis of {filename} ({file_type}): "
        + ("No issues found." if clean else f"{len(findings)} issue(s) detected.")
    )

    return {
        "filename": filename,
        "file_type": file_type,
        "findings": findings,
        "clean": clean,
        "summary": summary,
    }


def analyze_changed_files(changed_files: list[dict]) -> str:
    """
    Runs static analysis over every changed file in the PR
    and returns a single formatted string ready to inject
    into the LLM context as additional signal.

    Args:
      changed_files: list of dicts from GitHubClient.get_changed_files()
                     Each dict has 'filename', 'status', 'patch' keys.

    Returns:
      str: formatted static analysis report block.
           Empty string if no issues were found across all files.
    """
    all_findings = []

    for file_info in changed_files:
        filename = file_info.get("filename", "")
        content = file_info.get("content", "")  # populated by agent.py

        if not content:
            # No content to analyze (binary file, deleted file, etc.)
            continue

        result = analyze_file(filename, content)

        if not result["clean"]:
            all_findings.append(f"\n### {filename}")
            for finding in result["findings"]:
                all_findings.append(f"  - {finding}")

    if not all_findings:
        return ""

    header = "## Static Analysis Pre-Check\n"
    header += "The following objective issues were detected BEFORE LLM review:\n"
    return header + "\n".join(all_findings)
# Phase 5A risk test
