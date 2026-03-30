#  E-Commerce Superstore Analytics
### DSE3231 — Data Science Lab | Manipal University Jaipur | Jan-May 2026

A complete **end-to-end data science project** covering all 10 lab sessions from the course handout.

---

##  Project Summary

| Detail | Info |
|---|---|
| **Dataset** | Superstore Sales (synthetic, Kaggle-inspired) |
| **Domain** | E-Commerce / Retail Analytics |
| **Stack** | Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Plotly Dash |
| **Labs covered** | Lab 1 through Lab 10 (Capstone) |

---

##  File Structure

```
superstore_project/
│
├── run_all.py                          ← Run full pipeline at once
│
├── lab1_2_data_collection_cleaning.py  ← Lab 1 & 2
├── lab3_4_feature_eng_eda.py           ← Lab 3 & 4
├── lab5_static_visualization.py        ← Lab 5
├── lab6_dashboard.py                   ← Lab 6 (Dash app)
├── lab7_8_regression_classification.py ← Lab 7 & 8
├── lab9_clustering.py                  ← Lab 9
│
├── data/                               ← Generated data files
├── outputs/                            ← Generated plots & HTML charts
├── models/                             ← (optional) saved models
└── requirements.txt
```

---

##  Handout → Code Mapping

| Lab | Topic | Script | Key Output |
|---|---|---|---|
| 1 | Data Collection (CSV, Excel, JSON) | `lab1_2_...py` | `orders_clean.csv` |
| 2 | Data Cleaning | `lab1_2_...py` | Missing values fixed, duplicates removed |
| 3 | Feature Engineering | `lab3_4_...py` | RFM scores, Profit Margin, Ship Days |
| 4 | EDA | `lab3_4_...py` | Correlation matrix, descriptive stats |
| 5 | Static Visualization | `lab5_...py` | 3 PNG figures (charts & plots) |
| 6 | Interactive Dashboard | `lab6_...py` | Plotly Dash + 4 HTML charts |
| 7 | Regression Models | `lab7_8_...py` | Linear + Logistic Regression |
| 8 | Classification Models | `lab7_8_...py` | Decision Tree, Random Forest, SVM |
| 9 | Clustering | `lab9_...py` | K-Means + Hierarchical + Dendrogram |
| 10 | Capstone | `run_all.py` | Full pipeline + all outputs |

---

##  Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything in one go
python run_all.py

# 3. Or run individual labs
python lab1_2_data_collection_cleaning.py
python lab3_4_feature_eng_eda.py
python lab5_static_visualization.py
python lab7_8_regression_classification.py
python lab9_clustering.py

# 4. Launch interactive dashboard (Lab 6)
python lab6_dashboard.py
# Open http://127.0.0.1:8050
```

---

##  ML Models & Results (Expected)

### Regression (Lab 7)
| Model | Target | Metric |
|---|---|---|
| Linear Regression | Sales | R² ≈ 0.55–0.70 |
| Logistic Regression | Loss Order | Accuracy ≈ 70–80% |

### Classification (Lab 8)
| Model | Accuracy |
|---|---|
| Decision Tree | ~72–78% |
| Random Forest | ~78–85% |
| SVM | ~74–80% |

### Clustering (Lab 9)
- K-Means: 3–4 customer segments identified via Elbow + Silhouette
- Hierarchical: Dendrogram with Ward linkage

---

## 📊 Dashboard Features (Lab 6)
- **Global filters**: Region, Category, Segment
- **KPI cards**: Total Sales, Profit, Orders, Margin, Return Rate
- **Tab 1 — Sales**: Monthly trend + Top sub-categories
- **Tab 2 — Profit**: Sales vs Profit scatter + Shipping mode analysis
- **Tab 3 — Customers**: RFM bubble chart + Segment pie chart

---

##  Course Outcomes Addressed
| CO | Description | Labs |
|---|---|---|
| CO1 | Data cleaning, transformation, integration | 1, 2, 3 |
| CO2 | Statistical summaries, charts, graphs | 4, 5 |
| CO3 | ML models + evaluation metrics | 7, 8, 9 |
| CO4 | Interactive visualization / dashboards | 6 |
| CO5 | End-to-end data science solution | 10 (Capstone) |
