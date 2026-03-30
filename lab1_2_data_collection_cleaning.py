"""
LAB 1 & 2 — Data Collection, Import & Cleaning
================================================
DSE3231 | Manipal University Jaipur

Covers:
  - Importing from CSV, Excel, JSON
  - Handling missing values
  - Removing duplicates
  - Data type conversions
"""

import pandas as pd
import numpy as np
import json
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("data",    exist_ok=True)

# ══════════════════════════════════════════════════════════
# LAB 1: IMPORT DATA FROM MULTIPLE SOURCES
# ══════════════════════════════════════════════════════════
print("="*55)
print("  LAB 1 — Data Collection & Import")
print("="*55)

# 1A — CSV (real Kaggle Superstore dataset)
df = pd.read_csv("data/orders.csv", encoding="latin-1")
print(f"\n[CSV] Orders loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"Columns: {df.columns.tolist()}")

# 1B — Excel: generate customer info from the real data
customers_df = df[["Customer ID", "Customer Name", "Segment", "Region"]].drop_duplicates("Customer ID").copy()
customers_df.to_excel("data/customers.xlsx", index=False)
customers = pd.read_excel("data/customers.xlsx")
print(f"\n[Excel] Customers saved & loaded: {customers.shape}")

# 1C — JSON: generate a dummy returns file
#      (Kaggle Superstore doesn't include returns — we simulate it)
np.random.seed(42)
sample_orders = df["Order ID"].sample(80, replace=True).tolist()
reasons       = np.random.choice(
    ["Wrong item", "Damaged", "Not as described", "Changed mind"], 80).tolist()
returns_data  = {"returns": [{"Order ID": o, "Reason": r}
                              for o, r in zip(sample_orders, reasons)]}
with open("data/returns.json", "w") as f:
    json.dump(returns_data, f, indent=2)

with open("data/returns.json") as f:
    returns_raw = json.load(f)
returns_df = pd.DataFrame(returns_raw["returns"])
print(f"[JSON] Returns loaded: {returns_df.shape}")

# Flag returned orders
df["Returned"] = df["Order ID"].isin(returns_df["Order ID"]).astype(int)
print(f"\nOrders flagged as returned: {df['Returned'].sum()}")


# ══════════════════════════════════════════════════════════
# LAB 2: DATA CLEANING
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  LAB 2 — Data Cleaning")
print("="*55)

# 2A — Inspect
print(f"\n[Before Cleaning]")
print(f"  Shape          : {df.shape}")
print(f"  Duplicates     : {df.duplicated().sum()}")
missing = df.isnull().sum()
print(f"  Missing values :\n{missing[missing > 0] if missing.sum() > 0 else '  None ✅'}")

# 2B — Fix date columns
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=False)
df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=False)
print(f"\n  Order Date range: {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")

# 2C — Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"\n  Removed {before - len(df)} duplicate rows. Shape: {df.shape}")

# 2D — Handle missing values (Kaggle Superstore is mostly clean,
#       but we apply defensive handling anyway)
for col in ["Profit", "Sales", "Quantity", "Discount"]:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

for col in ["Customer Name", "Customer ID"]:
    if df[col].isnull().sum() > 0:
        df[col].fillna("Unknown", inplace=True)

# 2E — Ensure correct numeric types
df["Sales"]    = pd.to_numeric(df["Sales"],    errors="coerce")
df["Profit"]   = pd.to_numeric(df["Profit"],   errors="coerce")
df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

# 2F — Final check
print(f"\n[After Cleaning]")
print(f"  Missing values : {df.isnull().sum().sum()}")
print(f"  Final shape    : {df.shape}")
print(f"\n[Final dtypes]")
print(df.dtypes)

# Save
df.to_csv("data/orders_clean.csv", index=False)
print("\n✅ Cleaned data saved → data/orders_clean.csv")