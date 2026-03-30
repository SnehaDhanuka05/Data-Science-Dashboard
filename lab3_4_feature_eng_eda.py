"""
LAB 3 & 4 — Feature Engineering & EDA
========================================
DSE3231 | Manipal University Jaipur

Covers:
  - Feature selection, scaling, transformation
  - RFM feature creation
  - Univariate & multivariate analysis
  - Correlation analysis
  - Descriptive statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/orders_clean.csv", parse_dates=["Order Date", "Ship Date"])

print("="*55)
print("  LAB 3 — Feature Engineering & Transformation")
print("="*55)

# ══════════════════════════════════════════════════════════
# 3A — Derived / New Features
# ══════════════════════════════════════════════════════════

# Shipping duration (days)
df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

# Profit Margin (%)
df["Profit Margin %"] = (df["Profit"] / df["Sales"].replace(0, np.nan) * 100).round(2)

# Revenue per unit
df["Revenue per Unit"] = (df["Sales"] / df["Quantity"]).round(2)

# Is Discounted
df["Is Discounted"] = (df["Discount"] > 0).astype(int)

# Loss Order flag
df["Is Loss"] = (df["Profit"] < 0).astype(int)

# Month & Year from Order Date
df["Order Month"] = df["Order Date"].dt.month
df["Order Year"]  = df["Order Date"].dt.year
df["Order Quarter"] = df["Order Date"].dt.quarter

print("\n[New features added]")
new_feats = ["Ship Days", "Profit Margin %", "Revenue per Unit",
             "Is Discounted", "Is Loss", "Order Month", "Order Quarter"]
print(df[new_feats].describe().round(2))


# ══════════════════════════════════════════════════════════
# 3B — RFM Feature Engineering (Customer Level)
# ══════════════════════════════════════════════════════════
print("\n[RFM Feature Engineering]")
snapshot = df["Order Date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg(
    Recency   = ("Order Date",  lambda x: (snapshot - x.max()).days),
    Frequency = ("Order ID",    "count"),
    Monetary  = ("Sales",       "sum")
).reset_index()

# RFM Score (1-4 quartile bins)
rfm["R_Score"] = pd.qcut(rfm["Recency"],   4, labels=[4,3,2,1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"],  4, labels=[1,2,3,4]).astype(int)
rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

# Customer Segment from RFM
def rfm_segment(score):
    if score >= 10: return "Champions"
    elif score >= 7: return "Loyal"
    elif score >= 5: return "Potential"
    else: return "At Risk"

rfm["Segment"] = rfm["RFM_Total"].apply(rfm_segment)
print(rfm[["Customer ID","Recency","Frequency","Monetary","RFM_Total","Segment"]].head(10))
print(f"\nSegment distribution:\n{rfm['Segment'].value_counts()}")
rfm.to_csv("data/rfm.csv", index=False)


# ══════════════════════════════════════════════════════════
# 3C — Encoding & Scaling
# ══════════════════════════════════════════════════════════
print("\n[Encoding & Scaling]")
le = LabelEncoder()
df["Region_enc"]   = le.fit_transform(df["Region"])
df["Segment_enc"]  = le.fit_transform(df["Segment"])
df["Category_enc"] = le.fit_transform(df["Category"])

num_feats = ["Sales", "Quantity", "Discount", "Profit", "Ship Days", "Revenue per Unit"]

scaler_std = StandardScaler()
scaler_mm  = MinMaxScaler()

df_std = pd.DataFrame(scaler_std.fit_transform(df[num_feats]), columns=[f"{c}_std" for c in num_feats])
df_mm  = pd.DataFrame(scaler_mm.fit_transform(df[num_feats]),  columns=[f"{c}_mm"  for c in num_feats])

print("\nStandard Scaled (first 3 rows):")
print(df_std.head(3).round(3))

print("\nMin-Max Scaled (first 3 rows):")
print(df_mm.head(3).round(3))


# ══════════════════════════════════════════════════════════
# 3D — Feature Selection
# ══════════════════════════════════════════════════════════
X = df[["Sales", "Quantity", "Discount", "Ship Days",
        "Region_enc", "Segment_enc", "Category_enc"]].dropna()
y = df.loc[X.index, "Profit"]

selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X, y)

feat_scores = pd.DataFrame({
    "Feature": X.columns,
    "Score":   selector.scores_.round(2)
}).sort_values("Score", ascending=False)

print("\n[Feature Selection — F-Regression scores for predicting Profit]")
print(feat_scores.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(feat_scores["Feature"], feat_scores["Score"], color="#4C72B0")
ax.set_title("Feature Importance (F-Score vs Profit)", fontweight="bold")
ax.set_xlabel("F-Score")
plt.tight_layout()
plt.savefig("outputs/lab3_feature_importance.png", dpi=150)
plt.close()
print("\n✅ Saved → outputs/lab3_feature_importance.png")


# ══════════════════════════════════════════════════════════
# LAB 4 — EDA
# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  LAB 4 — Exploratory Data Analysis")
print("="*55)

# 4A — Descriptive Statistics
print("\n[Descriptive Statistics]")
print(df[["Sales","Quantity","Discount","Profit","Profit Margin %"]].describe().round(2))

# 4B — Skewness & Kurtosis
print("\n[Skewness & Kurtosis]")
for col in ["Sales", "Profit", "Discount", "Quantity"]:
    print(f"  {col:<12} Skew: {df[col].skew():.2f}   Kurt: {df[col].kurtosis():.2f}")

# 4C — Univariate: Sales distribution per category
print("\n[Category-wise Sales Summary]")
print(df.groupby("Category")["Sales"].agg(["mean","median","std","sum"]).round(2))

# 4D — Multivariate: Profit by Region & Segment
pivot = df.pivot_table(values="Profit", index="Region", columns="Segment", aggfunc="mean").round(2)
print("\n[Avg Profit: Region × Segment]")
print(pivot)

# 4E — Correlation Matrix
corr_cols = ["Sales","Quantity","Discount","Profit","Ship Days",
             "Revenue per Unit","Profit Margin %"]
corr = df[corr_cols].corr().round(2)
print("\n[Correlation Matrix]")
print(corr)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix — Key Numeric Features", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/lab4_correlation_matrix.png", dpi=150)
plt.close()

# 4F — Top 5 profitable Sub-Categories
top_sub = df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False).head(5)
print(f"\n[Top 5 Profitable Sub-Categories]\n{top_sub.round(2)}")

# 4G — Monthly Sales trend
monthly = df.groupby("Order Month")["Sales"].sum()
print(f"\n[Monthly Sales]\n{monthly.round(2)}")

print("\n✅ Saved → outputs/lab4_correlation_matrix.png")
df.to_csv("data/orders_featured.csv", index=False)
print("✅ Feature-engineered data saved → data/orders_featured.csv")
