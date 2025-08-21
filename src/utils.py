import pandas as pd
import yaml

def read_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# columns we expect after cleaning (baseline check; adjust if your dataset differs)
REQUIRED_COLUMNS = [
    "gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "PaperlessBilling","MonthlyCharges","TotalCharges","Churn"
]

def basic_schema_check(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

def yesno_to_binary(df: pd.DataFrame, cols):
    """Map 'Yes'/'No'/'No phone service' -> 1/0/0 inplace for given columns."""
    for c in cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map({"Yes": 1, "No": 0, "No phone service": 0})
            )
    return df

def reconstruct_contract_from_dummies(df: pd.DataFrame):
    """Create human-readable 'Contract' from one-hot columns if needed."""
    if "Contract" not in df.columns and "Contract_One year" in df.columns:
        df["Contract"] = "Month-to-month"
        one = df["Contract_One year"]
        two = df["Contract_Two year"]
        if one.dtype == bool:
            df.loc[one, "Contract"] = "One year"
            df.loc[two, "Contract"] = "Two year"
        else:
            df.loc[one == 1, "Contract"] = "One year"
            df.loc[two == 1, "Contract"] = "Two year"
    return df
