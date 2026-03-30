"""
LAB 5 — Static Data Visualization
====================================
DSE3231 | Manipal University Jaipur

Covers:
  - Bar charts, Histograms, Pie charts
  - Box plots, Heatmaps
  - Pair plots, Scatter plots
  - Matplotlib & Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv("data/orders_featured.csv", parse_dates=["Order Date", "Ship Date"])

print("="*55)
print("  LAB 5 — Static Data Visualization")
print("="*55)

# ══════════════════════════════════════════════════════════
# FIGURE 1 — Sales Overview (2×3 grid)
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.suptitle("E-Commerce Superstore — Sales Overview", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1A — Bar chart: Total Sales by Category
ax1 = fig.add_subplot(gs[0, 0])
cat_sales = df.groupby("Category")["Sales"].sum().sort_values()
colors = ["#4C72B0", "#DD8452", "#55A868"]
ax1.barh(cat_sales.index, cat_sales.values, color=colors)
ax1.set_title("Total Sales by Category", fontweight="bold")
ax1.set_xlabel("Sales ($)")
for i, v in enumerate(cat_sales.values):
    ax1.text(v + 500, i, f"${v:,.0f}", va="center", fontsize=9)

# 1B — Pie chart: Sales share by Region
ax2 = fig.add_subplot(gs[0, 1])
region_sales = df.groupby("Region")["Sales"].sum()
wedge_props = dict(linewidth=1.5, edgecolor="white")
ax2.pie(region_sales.values, labels=region_sales.index,
        autopct="%1.1f%%", startangle=90,
        colors=sns.color_palette("pastel"), wedgeprops=wedge_props)
ax2.set_title("Sales Share by Region", fontweight="bold")

# 1C — Histogram: Sales distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df["Sales"].clip(upper=1500), bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
ax3.axvline(df["Sales"].median(), color="red", linestyle="--", label=f"Median: ${df['Sales'].median():.0f}")
ax3.set_title("Sales Distribution", fontweight="bold")
ax3.set_xlabel("Sales ($)"); ax3.set_ylabel("Frequency")
ax3.legend()

# 1D — Box plot: Profit by Category
ax4 = fig.add_subplot(gs[1, 0])
cats = df["Category"].unique()
data_for_box = [df[df["Category"] == c]["Profit"].values for c in cats]
bp = ax4.boxplot(data_for_box, labels=cats, patch_artist=True,
                 medianprops=dict(color="red", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax4.set_title("Profit Distribution by Category", fontweight="bold")
ax4.set_ylabel("Profit ($)")

# 1E — Bar chart: Avg Profit Margin by Sub-Category (top 8)
ax5 = fig.add_subplot(gs[1, 1])
sub_margin = df.groupby("Sub-Category")["Profit Margin %"].mean().sort_values(ascending=False).head(8)
bar_colors = ["#55A868" if v >= 0 else "#C44E52" for v in sub_margin.values]
ax5.bar(sub_margin.index, sub_margin.values, color=bar_colors)
ax5.set_title("Avg Profit Margin % — Top 8 Sub-Categories", fontweight="bold")
ax5.set_xlabel("Sub-Category"); ax5.set_ylabel("Profit Margin (%)")
ax5.tick_params(axis="x", rotation=40)
ax5.axhline(0, color="black", linewidth=0.8)

# 1F — Monthly Sales trend (line)
ax6 = fig.add_subplot(gs[1, 2])
monthly = df.groupby("Order Month")["Sales"].sum()
ax6.plot(monthly.index, monthly.values, marker="o", linewidth=2,
         color="#4C72B0", markersize=6)
ax6.fill_between(monthly.index, monthly.values, alpha=0.15, color="#4C72B0")
ax6.set_title("Monthly Sales Trend", fontweight="bold")
ax6.set_xlabel("Month"); ax6.set_ylabel("Total Sales ($)")
ax6.set_xticks(range(1, 13))
ax6.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=30)

plt.savefig("outputs/lab5_sales_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved → outputs/lab5_sales_overview.png")


# ══════════════════════════════════════════════════════════
# FIGURE 2 — Heatmap + Pairplot
# ══════════════════════════════════════════════════════════

# 2A — Pivot Heatmap: Avg Sales — Region × Category
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle("Advanced Visualizations", fontsize=14, fontweight="bold")

pivot_heat = df.pivot_table(values="Sales", index="Region",
                             columns="Category", aggfunc="mean").round(0)
sns.heatmap(pivot_heat, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, ax=axes[0])
axes[0].set_title("Avg Sales: Region × Category", fontweight="bold")

# 2B — Stacked bar: Segment × Shipping Mode
ship_seg = df.groupby(["Segment", "Ship Mode"]).size().unstack(fill_value=0)
ship_seg.plot(kind="bar", stacked=True, ax=axes[1],
              colormap="tab10", edgecolor="white", linewidth=0.5)
axes[1].set_title("Shipping Mode by Customer Segment", fontweight="bold")
axes[1].set_xlabel("Segment"); axes[1].set_ylabel("Order Count")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Ship Mode", bbox_to_anchor=(1.01, 1))

plt.tight_layout()
plt.savefig("outputs/lab5_advanced_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved → outputs/lab5_advanced_plots.png")


# ══════════════════════════════════════════════════════════
# FIGURE 3 — Scatter Plots (Sales vs Profit by category)
# ══════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(9, 6))
category_colors = {"Furniture":"#4C72B0", "Office Supplies":"#DD8452", "Technology":"#55A868"}

for cat, color in category_colors.items():
    subset = df[df["Category"] == cat]
    ax.scatter(subset["Sales"], subset["Profit"], alpha=0.4, s=25,
               color=color, label=cat)

ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="Break-even")
ax.set_title("Sales vs Profit by Category", fontsize=13, fontweight="bold")
ax.set_xlabel("Sales ($)"); ax.set_ylabel("Profit ($)")
ax.legend(); ax.set_xlim(0, 2000)

plt.tight_layout()
plt.savefig("outputs/lab5_scatter_sales_profit.png", dpi=150)
plt.close()
print("✅ Saved → outputs/lab5_scatter_sales_profit.png")

print("\n✅ Lab 5 complete — 3 figures saved to outputs/")
