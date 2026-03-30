"""
run_all.py — Capstone: Run full DS pipeline end-to-end
========================================================
DSE3231 | Manipal University Jaipur

Executes Labs 1–9 in sequence, then prints a final summary.

Usage:
    python run_all.py
"""

import subprocess
import sys
import time

labs = [
    ("Lab 1 & 2 — Data Collection & Cleaning",  "lab1_2_data_collection_cleaning.py"),
    ("Lab 3 & 4 — Feature Engineering & EDA",   "lab3_4_feature_eng_eda.py"),
    ("Lab 5    — Static Visualization",          "lab5_static_visualization.py"),
    ("Lab 7 & 8 — Regression & Classification",  "lab7_8_regression_classification.py"),
    ("Lab 9    — Clustering",                    "lab9_clustering.py"),
]

print("\n" + "="*65)
print("  DSE3231 — Superstore Analytics: Full Pipeline")
print("  Manipal University Jaipur | Jan-May 2026")
print("="*65 + "\n")

start_total = time.time()

for label, script in labs:
    print(f"\n{'─'*65}")
    print(f"  ▶ Running: {label}")
    print(f"{'─'*65}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    status = "✅ Done" if result.returncode == 0 else "❌ Failed"
    print(f"\n  {status}  ({elapsed:.1f}s)")

total = time.time() - start_total

print("\n" + "="*65)
print("  PIPELINE COMPLETE")
print(f"  Total time : {total:.1f}s")
print("="*65)
print("""
  Output files:
  ─────────────────────────────────────────────────────────
  data/
    orders.csv              ← Raw CSV (Lab 1)
    customers.xlsx          ← Excel (Lab 1)
    returns.json            ← JSON  (Lab 1)
    orders_clean.csv        ← Cleaned (Lab 2)
    orders_featured.csv     ← With new features (Lab 3)
    rfm.csv                 ← RFM table (Lab 3)
    rfm_clustered.csv       ← With cluster labels (Lab 9)

  outputs/
    lab3_feature_importance.png
    lab4_correlation_matrix.png
    lab5_sales_overview.png
    lab5_advanced_plots.png
    lab5_scatter_sales_profit.png
    lab6_animated_bar.html      ← Open in browser
    lab6_treemap.html
    lab6_scatter.html
    lab6_rfm_bubble.html
    lab7a_linear_regression.png
    lab7b_logistic_regression.png
    lab8_classification.png
    lab8_decision_tree.png
    lab9_clustering.png

  For interactive dashboard (Lab 6):
    python lab6_dashboard.py
    Open http://127.0.0.1:8050
""")
