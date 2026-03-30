"""
LAB 9 — Clustering Techniques
================================
DSE3231 | Manipal University Jaipur

Covers:
  - K-Means clustering (with Elbow method)
  - Hierarchical / Agglomerative clustering
  - Dendrogram
  - Cluster interpretation & profiling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing      import StandardScaler
from sklearn.cluster            import KMeans, AgglomerativeClustering
from sklearn.metrics            import silhouette_score
from scipy.cluster.hierarchy    import dendrogram, linkage

os.makedirs("outputs", exist_ok=True)

df  = pd.read_csv("data/orders_featured.csv")
rfm = pd.read_csv("data/rfm.csv")

print("="*55)
print("  LAB 9 — Clustering Techniques")
print("="*55)

# Use RFM features for clustering
X_cluster = rfm[["Recency", "Frequency", "Monetary"]].copy()
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_cluster)


# ══════════════════════════════════════════════════════════
# 9A — K-MEANS: Elbow Method + Silhouette Score
# ══════════════════════════════════════════════════════════
print("\n[K-Means — Elbow Method]")
inertias    = []
sil_scores  = []
K_range     = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"  k={k}  Inertia={km.inertia_:.1f}  Silhouette={sil_scores[-1]:.4f}")

# Best k by silhouette
best_k = K_range[np.argmax(sil_scores)]
print(f"\n  Best k (by Silhouette) = {best_k}")


# ══════════════════════════════════════════════════════════
# 9B — Fit Final K-Means with best k
# ══════════════════════════════════════════════════════════
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm["KMeans_Cluster"] = km_final.fit_predict(X_scaled)

print(f"\n[K-Means Cluster Sizes]\n{rfm['KMeans_Cluster'].value_counts().sort_index()}")

# Cluster Profiles
profile = rfm.groupby("KMeans_Cluster")[["Recency","Frequency","Monetary"]].mean().round(1)
print(f"\n[K-Means Cluster Profiles]\n{profile}")


# ══════════════════════════════════════════════════════════
# 9C — HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════
print("\n[Hierarchical Clustering]")

# Use a sample for dendrogram (all rows makes it unreadable)
sample_idx  = np.random.choice(len(X_scaled), min(100, len(X_scaled)), replace=False)
X_sample    = X_scaled[sample_idx]

Z = linkage(X_sample, method="ward")

hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
rfm["Hier_Cluster"] = hier.fit_predict(X_scaled)

print(f"[Hierarchical Cluster Sizes]\n{rfm['Hier_Cluster'].value_counts().sort_index()}")

hier_profile = rfm.groupby("Hier_Cluster")[["Recency","Frequency","Monetary"]].mean().round(1)
print(f"\n[Hierarchical Cluster Profiles]\n{hier_profile}")


# ══════════════════════════════════════════════════════════
# 9D — VISUALISATION
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Lab 9 — Customer Clustering Analysis", fontsize=15, fontweight="bold")

# ── Row 1 ──────────────────────────────────────────────────
# Elbow curve
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(K_range, inertias, marker="o", color="#4C72B0", linewidth=2)
ax1.set_title("Elbow Method (Inertia)", fontweight="bold")
ax1.set_xlabel("Number of Clusters (k)"); ax1.set_ylabel("Inertia")
ax1.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
ax1.legend()

# Silhouette scores
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(K_range, sil_scores, marker="s", color="#DD8452", linewidth=2)
ax2.set_title("Silhouette Scores", fontweight="bold")
ax2.set_xlabel("Number of Clusters (k)"); ax2.set_ylabel("Silhouette Score")
ax2.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
ax2.legend()

# K-Means scatter: Recency vs Monetary
ax3 = fig.add_subplot(3, 3, 3)
palette = sns.color_palette("tab10", best_k)
for c in range(best_k):
    sub = rfm[rfm["KMeans_Cluster"] == c]
    ax3.scatter(sub["Recency"], sub["Monetary"], alpha=0.5, s=25,
                color=palette[c], label=f"Cluster {c}")
ax3.set_xlabel("Recency (days)"); ax3.set_ylabel("Monetary ($)")
ax3.set_title("K-Means: Recency vs Monetary", fontweight="bold")
ax3.legend(fontsize=8)

# ── Row 2 ──────────────────────────────────────────────────
# Cluster profile heatmap (K-Means)
ax4 = fig.add_subplot(3, 3, 4)
profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
sns.heatmap(profile_norm.T, annot=profile.T, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, ax=ax4)
ax4.set_title("K-Means Cluster Profiles (normalised)", fontweight="bold")
ax4.set_xlabel("Cluster")

# Cluster size bar chart
ax5 = fig.add_subplot(3, 3, 5)
cluster_counts = rfm["KMeans_Cluster"].value_counts().sort_index()
ax5.bar([f"C{i}" for i in cluster_counts.index], cluster_counts.values,
        color=palette[:len(cluster_counts)])
ax5.set_title("K-Means Cluster Sizes", fontweight="bold")
ax5.set_ylabel("Customer Count")
for i, v in enumerate(cluster_counts.values):
    ax5.text(i, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")

# Frequency vs Monetary scatter
ax6 = fig.add_subplot(3, 3, 6)
for c in range(best_k):
    sub = rfm[rfm["KMeans_Cluster"] == c]
    ax6.scatter(sub["Frequency"], sub["Monetary"], alpha=0.5, s=25,
                color=palette[c], label=f"Cluster {c}")
ax6.set_xlabel("Frequency (orders)"); ax6.set_ylabel("Monetary ($)")
ax6.set_title("K-Means: Frequency vs Monetary", fontweight="bold")
ax6.legend(fontsize=8)

# ── Row 3 ──────────────────────────────────────────────────
# Dendrogram
ax7 = fig.add_subplot(3, 1, 3)
dendrogram(Z, ax=ax7, color_threshold=0.7 * max(Z[:, 2]),
           leaf_rotation=90, leaf_font_size=7,
           above_threshold_color="gray")
ax7.set_title("Hierarchical Clustering Dendrogram (Ward linkage, 100 sample)", fontweight="bold")
ax7.set_xlabel("Sample Index"); ax7.set_ylabel("Distance")
ax7.axhline(y=sorted(Z[:,2])[-best_k+1], color="red",
             linestyle="--", label=f"Cut for {best_k} clusters")
ax7.legend()

plt.tight_layout()
plt.savefig("outputs/lab9_clustering.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved → outputs/lab9_clustering.png")

# Save RFM with cluster assignments
rfm.to_csv("data/rfm_clustered.csv", index=False)
print("✅ Saved → data/rfm_clustered.csv")
print("\n✅ Lab 9 complete!")
