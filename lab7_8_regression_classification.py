"""
LAB 7 & 8 — Regression & Classification Models
================================================
DSE3231 | Manipal University Jaipur

Covers:
  Lab 7 — Linear Regression (predict Sales)
           Logistic Regression (predict profit/loss)
  Lab 8 — Decision Tree, Random Forest, SVM (predict churn)
           Full evaluation: accuracy, confusion matrix, ROC, F1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.linear_model       import LinearRegression, LogisticRegression
from sklearn.tree               import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble           import RandomForestClassifier
from sklearn.svm                import SVC
from sklearn.metrics            import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

df = pd.read_csv("data/orders_featured.csv", parse_dates=["Order Date"])

# ── Encode categoricals ─────────────────────────────────────
le = LabelEncoder()
for col in ["Region", "Category", "Segment", "Ship Mode"]:
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))


# ══════════════════════════════════════════════════════════
# LAB 7A — LINEAR REGRESSION (Predict Sales)
# ══════════════════════════════════════════════════════════
print("="*60)
print("  LAB 7A — Linear Regression: Predicting Sales")
print("="*60)

features_lr = ["Quantity", "Discount", "Ship Days",
               "Region_enc", "Category_enc", "Segment_enc"]
target_lr   = "Sales"

X_lr = df[features_lr].dropna()
y_lr = df.loc[X_lr.index, target_lr]

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_lr, y_lr, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_lr_s = scaler.fit_transform(X_train_lr)
X_test_lr_s  = scaler.transform(X_test_lr)

lr_model = LinearRegression()
lr_model.fit(X_train_lr_s, y_train_lr)
y_pred_lr = lr_model.predict(X_test_lr_s)

rmse = np.sqrt(mean_squared_error(y_test_lr, y_pred_lr))
mae  = mean_absolute_error(y_test_lr, y_pred_lr)
r2   = r2_score(y_test_lr, y_pred_lr)

print(f"\n  RMSE : {rmse:.2f}")
print(f"  MAE  : {mae:.2f}")
print(f"  R²   : {r2:.4f}")

# Coefficient table
coef_df = pd.DataFrame({
    "Feature": features_lr,
    "Coefficient": lr_model.coef_.round(3)
}).sort_values("Coefficient", key=abs, ascending=False)
print(f"\n  Coefficients:\n{coef_df.to_string(index=False)}")

# Plot: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Lab 7A — Linear Regression: Sales Prediction", fontweight="bold")

axes[0].scatter(y_test_lr, y_pred_lr, alpha=0.4, color="#4C72B0", s=20)
lims = [min(y_test_lr.min(), y_pred_lr.min()), max(y_test_lr.max(), y_pred_lr.max())]
axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
axes[0].set_xlabel("Actual Sales"); axes[0].set_ylabel("Predicted Sales")
axes[0].set_title(f"Actual vs Predicted  (R²={r2:.3f})")
axes[0].legend()

residuals = y_test_lr - y_pred_lr
axes[1].hist(residuals, bins=40, color="#DD8452", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Residual Distribution"); axes[1].set_xlabel("Residual")

plt.tight_layout()
plt.savefig("outputs/lab7a_linear_regression.png", dpi=150)
plt.close()
print("✅ Saved → outputs/lab7a_linear_regression.png")


# ══════════════════════════════════════════════════════════
# LAB 7B — LOGISTIC REGRESSION (Predict Profit/Loss)
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  LAB 7B — Logistic Regression: Predicting Loss Orders")
print("="*60)

features_logr = ["Sales", "Quantity", "Discount", "Ship Days",
                 "Region_enc", "Category_enc", "Segment_enc"]
target_logr   = "Is Loss"

X_logr = df[features_logr].dropna()
y_logr = df.loc[X_logr.index, target_logr]

X_train_logr, X_test_logr, y_train_logr, y_test_logr = train_test_split(
    X_logr, y_logr, test_size=0.2, random_state=42, stratify=y_logr)

scaler2 = StandardScaler()
X_train_logr_s = scaler2.fit_transform(X_train_logr)
X_test_logr_s  = scaler2.transform(X_test_logr)

logr_model = LogisticRegression(max_iter=1000, random_state=42)
logr_model.fit(X_train_logr_s, y_train_logr)
y_pred_logr = logr_model.predict(X_test_logr_s)

print(f"\n  Accuracy : {accuracy_score(y_test_logr, y_pred_logr):.4f}")
print(f"\n  Classification Report:\n{classification_report(y_test_logr, y_pred_logr, target_names=['Profit','Loss'])}")

# ROC Curve
y_prob_logr = logr_model.predict_proba(X_test_logr_s)[:, 1]
fpr, tpr, _ = roc_curve(y_test_logr, y_prob_logr)
roc_auc     = auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Lab 7B — Logistic Regression: Loss Prediction", fontweight="bold")

# Confusion matrix
cm = confusion_matrix(y_test_logr, y_pred_logr)
ConfusionMatrixDisplay(cm, display_labels=["Profit", "Loss"]).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix")

# ROC
axes[1].plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"AUC = {roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], "r--", linewidth=1)
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve"); axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/lab7b_logistic_regression.png", dpi=150)
plt.close()
print("✅ Saved → outputs/lab7b_logistic_regression.png")


# ══════════════════════════════════════════════════════════
# LAB 8 — CLASSIFICATION (Predict High-Value Customer)
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  LAB 8 — Classification: Decision Tree / RF / SVM")
print("="*60)

# Target: Is the order returned? (binary classification)
features_clf = ["Sales", "Quantity", "Discount", "Profit",
                "Ship Days", "Region_enc", "Category_enc",
                "Segment_enc", "Ship Mode_enc"]
target_clf   = "Returned"

X_clf = df[features_clf].dropna()
y_clf = df.loc[X_clf.index, target_clf]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

scaler3 = StandardScaler()
X_train_cs = scaler3.fit_transform(X_train_c)
X_test_cs  = scaler3.transform(X_test_c)

models = {
    "Decision Tree":  DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM":            SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_cs, y_train_c)
    y_pred = model.predict(X_test_cs)
    acc    = accuracy_score(y_test_c, y_pred)
    cv_acc = cross_val_score(model, X_clf, y_clf, cv=5, scoring="accuracy").mean()
    results[name] = {"model": model, "preds": y_pred, "accuracy": acc, "cv_accuracy": cv_acc}
    print(f"\n  [{name}]")
    print(f"    Test Accuracy : {acc:.4f}")
    print(f"    CV Accuracy   : {cv_acc:.4f}")
    print(f"\n{classification_report(y_test_c, y_pred, target_names=['Not Returned','Returned'])}")


# ── Model Comparison Plot ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Lab 8 — Classification Model Comparison", fontweight="bold", fontsize=14)

model_names = list(results.keys())
test_accs   = [results[m]["accuracy"]    for m in model_names]
cv_accs     = [results[m]["cv_accuracy"] for m in model_names]

x = np.arange(len(model_names))
axes[0, 0].bar(x - 0.2, test_accs, 0.35, label="Test Acc",  color="#4C72B0")
axes[0, 0].bar(x + 0.2, cv_accs,   0.35, label="CV Acc",    color="#DD8452")
axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(model_names)
axes[0, 0].set_ylim(0, 1.1); axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_title("Accuracy Comparison"); axes[0, 0].legend()
for i, (ta, cv) in enumerate(zip(test_accs, cv_accs)):
    axes[0, 0].text(i - 0.2, ta + 0.01, f"{ta:.3f}", ha="center", fontsize=9)
    axes[0, 0].text(i + 0.2, cv + 0.01, f"{cv:.3f}", ha="center", fontsize=9)

# Confusion matrices for RF (best model)
best_model_name = max(results, key=lambda m: results[m]["accuracy"])
ConfusionMatrixDisplay(
    confusion_matrix(y_test_c, results[best_model_name]["preds"]),
    display_labels=["Not Returned", "Returned"]
).plot(ax=axes[0, 1], cmap="Blues")
axes[0, 1].set_title(f"Confusion Matrix — {best_model_name}")

# ROC curves for all models
for name, res in results.items():
    y_prob = res["model"].predict_proba(X_test_cs)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_c, y_prob)
    axes[1, 0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
axes[1, 0].plot([0,1],[0,1],"r--",linewidth=1)
axes[1, 0].set_xlabel("FPR"); axes[1, 0].set_ylabel("TPR")
axes[1, 0].set_title("ROC Curves — All Models"); axes[1, 0].legend(fontsize=9)

# Feature importance from Random Forest
rf = results["Random Forest"]["model"]
feat_imp = pd.Series(rf.feature_importances_, index=features_clf).sort_values(ascending=True)
feat_imp.tail(8).plot(kind="barh", ax=axes[1, 1], color="#55A868")
axes[1, 1].set_title("Random Forest Feature Importance")
axes[1, 1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("outputs/lab8_classification.png", dpi=150)
plt.close()
print("\n✅ Saved → outputs/lab8_classification.png")

# Decision Tree visualisation
fig_dt, ax_dt = plt.subplots(figsize=(16, 8))
plot_tree(results["Decision Tree"]["model"],
          feature_names=features_clf,
          class_names=["Not Returned", "Returned"],
          filled=True, rounded=True, fontsize=8, ax=ax_dt, max_depth=3)
ax_dt.set_title("Decision Tree (max_depth=3 shown)", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/lab8_decision_tree.png", dpi=150)
plt.close()
print("✅ Saved → outputs/lab8_decision_tree.png")

print("\n✅ Lab 7 & 8 complete!")
