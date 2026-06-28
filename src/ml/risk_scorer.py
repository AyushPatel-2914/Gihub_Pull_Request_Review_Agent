"""
src/ml/risk_scorer.py
─────────────────────────────────────────────────────────────
Predictive Risk Scorer — 0-100% PR Risk Score

WHAT IT DOES:
  Extracts numerical features from a PR (size, file types, security
  keywords, test coverage, etc.) and produces a 0-100% risk score
  so an engineering manager can triage at a glance.

WHY A HEURISTIC MODEL FIRST:
  A trained XGBoost model needs historical labeled data
  (PRs labeled "caused a bug" vs "was fine"). You don't have
  that yet on a fresh repo. So this ships with a well-calibrated
  HEURISTIC scorer that works immediately, AND the plumbing to
  load a trained XGBoost model later via the RISK_MODEL_PATH env var.

  When you've collected enough labeled PRs, train an XGBoost model
  with scripts/train_risk_model.py and drop it in — the scorer
  switches automatically.

THE FEATURES (these matter for risk):
  - total_changes      : additions + deletions (bigger = riskier)
  - files_changed      : more files = wider blast radius
  - file_density       : changes concentrated vs spread out
  - has_security_terms : touches auth/crypto/secrets/permissions
  - has_tests          : does the PR add/modify tests? (lowers risk)
  - touches_config     : config/CI/dependency changes (raises risk)
  - touches_migrations : DB migrations (raises risk significantly)
  - deletion_ratio     : mostly deletions can mean removed safeguards
  - max_file_changes   : one giant file change is risky
─────────────────────────────────────────────────────────────
"""

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Optional trained model path — if set and file exists, use XGBoost
RISK_MODEL_PATH = os.environ.get("RISK_MODEL_PATH", "")

# Security-sensitive keywords that elevate risk when touched
SECURITY_KEYWORDS = [
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "auth", "authenticate", "authorization", "credential", "private_key",
    "encrypt", "decrypt", "crypto", "hash", "salt", "jwt", "oauth",
    "permission", "privilege", "admin", "sudo", "root",
    "sql", "query", "exec", "eval", "subprocess", "os.system",
    "pickle", "deserialize", "yaml.load",
]

# File patterns that indicate elevated risk
CONFIG_PATTERNS = [
    r"\.ya?ml$", r"\.json$", r"\.toml$", r"\.ini$", r"\.cfg$",
    r"dockerfile", r"requirements\.txt$", r"package\.json$",
    r"\.github/", r"\.gitlab", r"Makefile",
]

MIGRATION_PATTERNS = [
    r"migration", r"migrate", r"schema", r"alembic", r"\.sql$",
]

TEST_PATTERNS = [
    r"test_", r"_test\.", r"\.test\.", r"/tests?/", r"spec\.",
]


@dataclass
class RiskFeatures:
    """Numerical features extracted from a PR for risk scoring."""
    total_changes: int = 0
    additions: int = 0
    deletions: int = 0
    files_changed: int = 0
    max_file_changes: int = 0
    has_security_terms: int = 0       # 0 or 1
    security_term_count: int = 0
    has_tests: int = 0                # 0 or 1
    touches_config: int = 0          # 0 or 1
    touches_migrations: int = 0      # 0 or 1
    deletion_ratio: float = 0.0

    def to_vector(self) -> list[float]:
        """Returns features as an ordered vector for the ML model."""
        return [
            self.total_changes,
            self.additions,
            self.deletions,
            self.files_changed,
            self.max_file_changes,
            self.has_security_terms,
            self.security_term_count,
            self.has_tests,
            self.touches_config,
            self.touches_migrations,
            self.deletion_ratio,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "total_changes", "additions", "deletions", "files_changed",
            "max_file_changes", "has_security_terms", "security_term_count",
            "has_tests", "touches_config", "touches_migrations", "deletion_ratio",
        ]


@dataclass
class RiskResult:
    """Output of risk scoring."""
    score: int                        # 0-100
    level: str                        # LOW / MEDIUM / HIGH / CRITICAL
    emoji: str                        # visual indicator
    factors: list[str] = field(default_factory=list)  # human-readable drivers
    features: RiskFeatures = None
    method: str = "heuristic"         # "heuristic" or "xgboost"


# ──────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────

def extract_features(pr_metadata: dict, changed_files: list, diff: str) -> RiskFeatures:
    """
    Extracts risk features from PR data.

    Args:
      pr_metadata   : dict with additions, deletions, changed_files
      changed_files : list of changed file dicts
      diff          : raw unified diff string

    Returns:
      RiskFeatures
    """
    additions = pr_metadata.get("additions", 0)
    deletions = pr_metadata.get("deletions", 0)
    total = additions + deletions
    files_changed = len(changed_files)

    # Per-file max change size
    max_file_changes = 0
    for f in changed_files:
        file_changes = f.get("additions", 0) + f.get("deletions", 0)
        max_file_changes = max(max_file_changes, file_changes)

    # Scan filenames and diff content for risk signals
    filenames = [f.get("filename", "").lower() for f in changed_files]
    all_filenames = " ".join(filenames)

    # Security: scan the ADDED lines of the diff for sensitive terms
    added_content = "\n".join(
        line.lower() for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    security_hits = [kw for kw in SECURITY_KEYWORDS if kw in added_content]
    security_term_count = len(security_hits)

    # File-type signals
    touches_config = int(any(
        re.search(p, all_filenames) for p in CONFIG_PATTERNS
    ))
    touches_migrations = int(any(
        re.search(p, all_filenames) for p in MIGRATION_PATTERNS
    ))
    has_tests = int(any(
        re.search(p, all_filenames) for p in TEST_PATTERNS
    ))

    deletion_ratio = (deletions / total) if total > 0 else 0.0

    return RiskFeatures(
        total_changes=total,
        additions=additions,
        deletions=deletions,
        files_changed=files_changed,
        max_file_changes=max_file_changes,
        has_security_terms=int(security_term_count > 0),
        security_term_count=security_term_count,
        has_tests=has_tests,
        touches_config=touches_config,
        touches_migrations=touches_migrations,
        deletion_ratio=round(deletion_ratio, 2),
    )


# ──────────────────────────────────────────────────────────────
# HEURISTIC SCORER
# ──────────────────────────────────────────────────────────────

def _score_heuristic(features: RiskFeatures) -> RiskResult:
    """
    Computes a 0-100 risk score using a transparent, well-calibrated
    point system. Each risk factor adds points; tests subtract points.

    This works immediately with no training data and is explainable —
    every point is tied to a named factor shown to the user.
    """
    score = 0.0
    factors = []

    # ── Size-based risk ───────────────────────────────────────
    total = features.total_changes
    if total > 1000:
        score += 30
        factors.append(f"Very large PR ({total} lines changed)")
    elif total > 500:
        score += 20
        factors.append(f"Large PR ({total} lines changed)")
    elif total > 200:
        score += 10
        factors.append(f"Moderate size ({total} lines changed)")
    elif total > 50:
        score += 5

    # ── File spread ───────────────────────────────────────────
    if features.files_changed > 20:
        score += 15
        factors.append(f"Wide blast radius ({features.files_changed} files)")
    elif features.files_changed > 10:
        score += 10
        factors.append(f"Many files touched ({features.files_changed})")
    elif features.files_changed > 5:
        score += 5

    # ── One giant file ────────────────────────────────────────
    if features.max_file_changes > 500:
        score += 10
        factors.append(f"Single file with {features.max_file_changes} changes")

    # ── Security signals ──────────────────────────────────────
    if features.security_term_count > 0:
        sec_points = min(features.security_term_count * 8, 25)
        score += sec_points
        factors.append(
            f"Touches security-sensitive code "
            f"({features.security_term_count} sensitive term(s))"
        )

    # ── Migrations (high risk) ────────────────────────────────
    if features.touches_migrations:
        score += 20
        factors.append("Modifies database migrations/schema")

    # ── Config changes ────────────────────────────────────────
    if features.touches_config:
        score += 10
        factors.append("Changes configuration/CI/dependencies")

    # ── Deletion-heavy ────────────────────────────────────────
    if features.deletion_ratio > 0.7 and features.total_changes > 50:
        score += 10
        factors.append(
            f"Mostly deletions ({features.deletion_ratio:.0%}) — "
            "verify no safeguards removed"
        )

    # ── Tests REDUCE risk ─────────────────────────────────────
    if features.has_tests:
        score -= 10
        factors.append("✓ Includes test changes (reduces risk)")
    elif features.total_changes > 100:
        # Large PR with NO tests is itself a risk
        score += 10
        factors.append("No test changes despite significant code changes")

    # Clamp to 0-100
    score = max(0, min(100, int(round(score))))

    level, emoji = _risk_level(score)

    return RiskResult(
        score=score,
        level=level,
        emoji=emoji,
        factors=factors or ["No significant risk factors detected"],
        features=features,
        method="heuristic",
    )


def _risk_level(score: int) -> tuple[str, str]:
    """Maps a numeric score to a level label and emoji."""
    if score >= 75:
        return "CRITICAL", "🔴"
    elif score >= 50:
        return "HIGH", "🟠"
    elif score >= 25:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🟢"


# ──────────────────────────────────────────────────────────────
# XGBOOST SCORER (loads if a trained model exists)
# ──────────────────────────────────────────────────────────────

def _score_xgboost(features: RiskFeatures, model_path: str) -> RiskResult:
    """
    Scores using a trained XGBoost model. Falls back to heuristic
    if the model can't be loaded.

    The model is trained by scripts/train_risk_model.py once you
    have labeled historical PR data.
    """
    try:
        import xgboost as xgb
        import numpy as np

        model = xgb.XGBClassifier()
        model.load_model(model_path)

        X = np.array([features.to_vector()])
        # Probability of the "risky" class → 0-100 score
        proba = model.predict_proba(X)[0][1]
        score = int(round(proba * 100))

        level, emoji = _risk_level(score)

        # Still surface heuristic factors for explainability
        heuristic = _score_heuristic(features)

        logger.info("Risk scored via XGBoost | score=%d", score)
        return RiskResult(
            score=score,
            level=level,
            emoji=emoji,
            factors=heuristic.factors,
            features=features,
            method="xgboost",
        )
    except Exception as e:
        logger.error("XGBoost scoring failed: %s — using heuristic", e)
        return _score_heuristic(features)


# ──────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ──────────────────────────────────────────────────────────────

def score_pr_risk(pr_metadata: dict, changed_files: list, diff: str) -> RiskResult:
    """
    Main entry point. Extracts features and computes the risk score.

    Uses a trained XGBoost model if RISK_MODEL_PATH is set and the
    file exists; otherwise uses the heuristic scorer.

    Args:
      pr_metadata   : PR metadata dict
      changed_files : list of changed file dicts
      diff          : raw unified diff

    Returns:
      RiskResult
    """
    features = extract_features(pr_metadata, changed_files, diff)
    logger.info(
        "Risk features | changes=%d | files=%d | security=%d | tests=%d | config=%d",
        features.total_changes, features.files_changed,
        features.security_term_count, features.has_tests, features.touches_config,
    )

    if RISK_MODEL_PATH and os.path.exists(RISK_MODEL_PATH):
        result = _score_xgboost(features, RISK_MODEL_PATH)
    else:
        result = _score_heuristic(features)

    logger.info(
        "Risk score: %d%% (%s) via %s",
        result.score, result.level, result.method,
    )
    return result


def format_risk_section(result: RiskResult) -> str:
    """
    Formats the risk result as a markdown section for the PR comment.
    """
    parts = [
        f"### {result.emoji} Risk Assessment: {result.score}% ({result.level})",
        "",
        "**Risk factors:**",
    ]
    for factor in result.factors:
        parts.append(f"- {factor}")
    return "\n".join(parts)

# testing RAG retrieval
