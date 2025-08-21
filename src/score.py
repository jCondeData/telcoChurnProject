import os
import sys
import joblib
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import read_yaml, reconstruct_contract_from_dummies

def main():
    cfg = read_yaml(os.path.join(HERE, "config.yml"))
    if not cfg or "paths" not in cfg:
        raise ValueError("Config not loaded. Check src/config.yml is valid YAML.")
    paths = cfg["paths"]

    model_abs  = os.path.join(ROOT, paths["model_path"])
    proc_abs   = os.path.join(ROOT, paths["processed_csv"])
    scored_abs = os.path.join(ROOT, paths["scored_csv"])

    print("[score] loading model and data...")
    model = joblib.load(model_abs)
    df = pd.read_csv(proc_abs)

    X = df.select_dtypes(include=["int64","float64","bool"]).drop(columns=["Churn"], errors="ignore")
    print("[score] scoring...")
    df["Churn_Prob"] = model.predict_proba(X)[:, 1]

    df = reconstruct_contract_from_dummies(df)

    os.makedirs(os.path.dirname(scored_abs), exist_ok=True)
    df.to_csv(scored_abs, index=False)
    print(f"[score] wrote -> {paths['scored_csv']} | shape: {df.shape}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[score] ERROR:", repr(e))
        raise
