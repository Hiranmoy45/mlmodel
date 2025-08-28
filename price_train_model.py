import re
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "listings.csv"
MODEL_PATH = Path(__file__).resolve().parent / "price-mode.joblib"

def parse_price(s: str) -> float:
    """
    Convert '₹1.99 Cr' -> 1.99e7, '₹55 L' -> 5.5e6, '₹80,00,000' -> 8e6, '₹75 K' -> 7.5e4.
    Returns rupees as float.
    """
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    # fix common mojibake for ₹
    s = s.replace("â‚¹", "₹")
    # remove commas/spaces
    s_clean = s.replace(",", "").replace(" ", "")
    m = re.search(r"([0-9]*\.?[0-9]+)", s_clean)
    if not m:
        return np.nan
    val = float(m.group(1))
    if "Cr" in s_clean or "crore" in s.lower():
        val *= 1e7
    elif "L" in s_clean or "lac" in s.lower() or "lakh" in s.lower():
        val *= 1e5
    elif re.search(r"[Kk]\b", s):
        val *= 1e3
    # else assume rupees already
    return val

def extract_bhk(title: str) -> float:
    """Extract the leading number before 'BHK'."""
    if pd.isna(title):
        return np.nan
    m = re.search(r"(\d+)\s*BHK", str(title))
    return float(m.group(1)) if m else np.nan

def yes_no_to_int(x) -> int:
    if pd.isna(x): return 0
    s = str(x).strip().lower()
    return 1 if s in {"yes", "y", "true", "1"} else 0

def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")  # handles stray BOM
    # Standardize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Derive / clean fields
    if "Price" in df.columns:
        df["price_inr"] = df["Price"].apply(parse_price)
    else:
        raise ValueError("Missing 'Price' column")

    if "BHK" not in df.columns:
        df["BHK"] = df.get("Property_Title", df.get("Property_Title", "")).apply(extract_bhk)

    df["BalconyFlag"] = df.get("Balcony", 0).apply(yes_no_to_int)
    df["Baths"] = pd.to_numeric(df.get("Baths", np.nan), errors="coerce")
    df["Total_Area"] = pd.to_numeric(df.get("Total_Area", np.nan), errors="coerce")

    # Keep only useful columns
    use_cols = ["Total_Area", "BHK", "Baths", "BalconyFlag", "Location", "price_inr"]
    df = df[use_cols].dropna()

    # Filter obvious outliers (optional)
    df = df[(df["Total_Area"] > 150) & (df["Total_Area"] < 15000)]
    df = df[(df["price_inr"] > 1e5) & (df["price_inr"] < 5e8)]

    # Features / target
    X = df.drop(columns=["price_inr"])
    y = np.log1p(df["price_inr"])   # log to stabilize

    numeric = ["Total_Area", "BHK", "Baths", "BalconyFlag"]
    categorical = ["Location"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("rf", model),
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)

    y_pred = pipe.predict(Xte)
    # Bring back to rupees
    y_pred_r = np.expm1(y_pred)
    y_true_r = np.expm1(yte)

    r2 = r2_score(y_true_r, y_pred_r)
    mae = mean_absolute_error(y_true_r, y_pred_r)
    print(f"R2: {r2:.3f} | MAE: ₹{mae:,.0f}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
