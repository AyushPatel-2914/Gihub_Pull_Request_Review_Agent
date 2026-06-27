"""
scripts/train_risk_model.py
─────────────────────────────────────────────────────────────
XGBoost Risk Model Trainer (OPTIONAL — for later)

WHEN TO USE THIS:
  The heuristic scorer in src/ml/risk_scorer.py works immediately
  with no training. But once you've collected historical PR data
  labeled as "risky" (caused a bug/revert/incident) vs "safe",
  you can train an XGBoost model that learns YOUR repo's patterns.

DATA FORMAT:
  Provide a CSV with these columns (one row per historical PR):
    total_changes, additions, deletions, files_changed,
    max_file_changes, has_security_terms, security_term_count,
    has_tests, touches_config, touches_migrations, deletion_ratio,
    label
  where label = 1 (risky) or 0 (safe).

USAGE:
  python scripts/train_risk_model.py --data historical_prs.csv --out models/risk_model.json

THEN:
  Set the env var so the agent uses it:
    RISK_MODEL_PATH=models/risk_model.json
  (add it as a GitHub Actions secret/variable, or in .env locally)

  The scorer auto-detects the model and switches from heuristic
  to XGBoost — no code changes needed.
─────────────────────────────────────────────────────────────
"""

import argparse
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_risk")

FEATURE_COLUMNS = [
    "total_changes", "additions", "deletions", "files_changed",
    "max_file_changes", "has_security_terms", "security_term_count",
    "has_tests", "touches_config", "touches_migrations", "deletion_ratio",
]


def train(data_path: str, out_path: str):
    """Trains an XGBoost classifier on labeled PR data."""
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score

    logger.info("Loading training data from %s", data_path)
    df = pd.read_csv(data_path)

    missing_cols = [c for c in FEATURE_COLUMNS + ["label"] if c not in df.columns]
    if missing_cols:
        logger.error("Missing required columns: %s", missing_cols)
        sys.exit(1)

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    logger.info("Dataset: %d PRs | %d risky | %d safe",
                len(df), int(y.sum()), int((1 - y).sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    logger.info("\n%s", classification_report(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, y_proba)
        logger.info("ROC-AUC: %.3f", auc)
    except ValueError:
        logger.warning("Could not compute AUC (single class in test set)")

    # Feature importance
    logger.info("Feature importances:")
    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    for name, imp in importances:
        logger.info("  %-22s %.3f", name, imp)

    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save_model(out_path)
    logger.info("Model saved to %s", out_path)
    logger.info("Set RISK_MODEL_PATH=%s to activate it in the agent.", out_path)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost PR risk model.")
    parser.add_argument("--data", required=True, help="Path to labeled CSV data")
    parser.add_argument("--out", default="models/risk_model.json", help="Output model path")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
