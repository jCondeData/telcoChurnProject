import os
import sys
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Ensure we can import src.utils
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import read_yaml

def main():
    print("[train] starting...")
    cfg_path = os.path.join(HERE, "config.yml")
    cfg = read_yaml(cfg_path)
    print("[train] cfg:", cfg)
    if not cfg or "paths" not in cfg:
        raise ValueError("Config not loaded. Check src/config.yml is valid YAML.")
    paths = cfg["paths"]

    proc_abs   = os.path.join(ROOT, paths["processed_csv"])
    model_abs  = os.path.join(ROOT, paths["model_path"])
    metrics_abs= os.path.join(ROOT, paths["metrics_path"])

    if not os.path.exists(proc_abs):
        raise FileNotFoundError(f"Processed CSV not found at {proc_abs}. Run clean.py first.")

    print(f"[train] reading processed: {paths['processed_csv']}")
    df = pd.read_csv(proc_abs)

    if "Churn" not in df.columns:
        raise ValueError("Processed data has no 'Churn' column. Check cleaning step.")

    # Features: numeric & bool; drop target
    X = df.select_dtypes(include=["int64","float64","bool"]).drop(columns=["Churn"], errors="ignore")
    y = df["Churn"].astype(int)

    print(f"[train] X shape: {X.shape} | y shape: {y.shape}")
    if X.empty:
        raise ValueError("No numeric/bool features found after processing.")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    print("[train] fitting RandomForest...")
    clf.fit(X_train, y_train)

    # Metrics (val split)
    y_proba = clf.predict_proba(X_val)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        "val_roc_auc": float(roc_auc_score(y_val, y_proba)),
        "val_accuracy": float(accuracy_score(y_val, y_pred)),
        "val_f1": float(f1_score(y_val, y_pred)),
        "n_features": int(X.shape[1]),
    }

    # CV (optional, quick)
    print("[train] 5-fold ROC-AUC CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    metrics["cv_roc_auc_mean"] = float(cv_auc.mean())
    metrics["cv_roc_auc_std"]  = float(cv_auc.std())

    # Ensure output dir and save
    os.makedirs(os.path.dirname(model_abs), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_abs), exist_ok=True)

    joblib.dump(clf, model_abs)
    with open(metrics_abs, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] saved model -> {paths['model_path']}")
    print(f"[train] saved metrics -> {paths['metrics_path']}")
    print("[train] done. Metrics:", metrics)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[train] ERROR:", repr(e))
        raise
