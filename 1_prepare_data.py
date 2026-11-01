"""STEP 1: Get the data, clean it, and store it in a database (SQLite).
Run:  python 1_prepare_data.py

- If you downloaded the real Kaggle file "water_potability.csv" and put it in
  this folder, the code uses it.
- If not, it creates practice data with the SAME columns."""
import os, sqlite3
import numpy as np
import pandas as pd

if os.path.exists("water_potability.csv"):
    df = pd.read_csv("water_potability.csv")
    print("Using REAL dataset: water_potability.csv")
else:
    print("Real file not found -> creating synthetic practice data")
    rng = np.random.default_rng(42)
    n = 3000
    df = pd.DataFrame({
        "ph":              rng.normal(7.1, 1.5, n).clip(0, 14),
        "Hardness":        rng.normal(196, 32, n).clip(47, 323),
        "Solids":          rng.normal(21000, 8500, n).clip(320, 61000),
        "Chloramines":     rng.normal(7.1, 1.6, n).clip(0.3, 13),
        "Sulfate":         rng.normal(333, 40, n).clip(129, 481),
        "Conductivity":    rng.normal(426, 80, n).clip(181, 753),
        "Organic_carbon":  rng.normal(14.3, 3.3, n).clip(2, 28),
        "Trihalomethanes": rng.normal(66, 16, n).clip(0, 124),
        "Turbidity":       rng.normal(3.97, 0.78, n).clip(1.4, 6.7),
    })
    # Safe-water rule (loosely based on drinking-water guidelines) + randomness
    score = (
        (df["ph"].between(6.5, 8.5)).astype(int) * 2
        + (df["Turbidity"] < 4.0).astype(int) * 2
        + (df["Solids"] < 25000).astype(int) * 2
        + (df["Chloramines"] < 8).astype(int)
        + (df["Trihalomethanes"] < 80).astype(int)
        + (df["Organic_carbon"] < 16).astype(int)
        + (df["Hardness"].between(120, 250)).astype(int)
    )
    prob = 1 / (1 + np.exp(-(score - 6.0) * 1.3))
    df["Potability"] = (rng.random(n) < prob).astype(int)
    # real data has missing values, so add some to practise cleaning
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        df.loc[rng.random(n) < 0.08, col] = np.nan

print("\nMissing values before cleaning:\n", df.isna().sum()[df.isna().sum() > 0])
df = df.fillna(df.median(numeric_only=True))      # CLEANING: fill gaps with the median
print("\nClass balance (1 = safe to drink):\n", df["Potability"].value_counts())

df.to_csv("water_clean.csv", index=False)
with sqlite3.connect("water.db") as con:           # the DATABASE
    df.to_sql("water_quality", con, if_exists="replace", index=False)
print("\nSaved water_clean.csv and water.db  | rows:", len(df))
