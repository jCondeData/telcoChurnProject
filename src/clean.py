import os
import sys
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import read_yaml, basic_schema_check, yesno_to_binary

def main():
    cfg_path = os.path.join(HERE, "config.yml")
    print("[clean] cfg_path:", cfg_path)
    cfg = read_yaml(cfg_path)
    print("[clean] cfg loaded:", cfg)
    if not cfg or "paths" not in cfg:
        raise ValueError("Config not loaded. Check src/config.yml is valid YAML.")

    raw_csv = cfg["paths"]["raw_csv"]
    out_csv = cfg["paths"]["processed_csv"]

    raw_csv_abs = os.path.join(ROOT, raw_csv)
    out_csv_abs = os.path.join(ROOT, out_csv)

    os.makedirs(os.path.dirname(out_csv_abs), exist_ok=True)

    print(f"[clean] reading: {raw_csv}")
    if not os.path.exists(raw_csv_abs):
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv}")

    df = pd.read_csv(raw_csv_abs)

    # light type fixes BEFORE schema check
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    yesno_cols = [
        "Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
        "PaperlessBilling","Churn"
    ]
    df = yesno_to_binary(df, yesno_cols)

    if "SeniorCitizen" in df.columns:
        try:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
        except Exception:
            df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # ✅ Schema check while raw columns (like `gender`) still exist
    basic_schema_check(df)

    # One-hot after schema check
    cat_cols = [c for c in ["InternetService","Contract","PaymentMethod","gender"] if c in df.columns]
    print(f"[clean] one-hot cols: {cat_cols}")
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    df.to_csv(out_csv_abs, index=False)
    print(f"[clean] wrote -> {out_csv} | shape: {df.shape}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[clean] ERROR:", repr(e))
        raise
