import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
import seaborn as sns 
import numpy as np

# ----------------------------------------------------------------------
# Evaluation metrics and confusion matrices for different PCA settings
# ---------------------------------------------------------------------
pca_components = [2, 10, 25, 50, 64]

# Test set metrics
accuracy = [0.6843, 0.7013, 0.6994, 0.7119, 0.7128]
precision = [0.6694, 0.6839, 0.6733, 0.6841, 0.6851]
recall = [0.7851, 0.7991, 0.8281, 0.8360, 0.8360]
f1_score = [0.7226, 0.7371, 0.7427, 0.7525, 0.7531]
roc_auc = [0.7672, 0.7600, 0.7896, 0.7993, 0.7993]

# Cross-validation accuracy
cv_accuracy = [0.7263, 0.7531, 0.7707, 0.7781, 0.7783]

# Confusion matrices: [[TN, FP], [FN, TP]]
conf_matrices = {
    2:  np.array([[594, 442],
                  [245, 895]]),
    10: np.array([[615, 421],
                  [229, 911]]),
    25: np.array([[578, 458],
                  [196, 944]]),
    50: np.array([[596, 440],
                  [187, 953]]),
    64: np.array([[598, 438],
                  [187, 953]])
}

# -----------------------------
# Style
# -----------------------------
sns.set_theme(style="whitegrid")

# -----------------------------
# Figure 1: PCA vs performance
# -----------------------------
plt.figure(figsize=(9, 6))
plt.plot(pca_components, accuracy, marker="o", linewidth=2, label="Test Accuracy")
plt.plot(pca_components, f1_score, marker="s", linewidth=2, label="Test F1 Score")
plt.plot(pca_components, roc_auc, marker="^", linewidth=2, label="Test ROC AUC")
plt.plot(pca_components, cv_accuracy, marker="d", linewidth=2, linestyle="--", label="CV Accuracy")

plt.xlabel("Number of PCA Components")
plt.ylabel("Score")
plt.title("Model Performance Across PCA Settings")
plt.xticks(pca_components)
plt.ylim(0.65, 0.82)
plt.legend()
plt.tight_layout()
plt.savefig("pca_performance.png")
plt.close()
# -----------------------------
# Figure 2: Precision vs Recall
# -----------------------------
plt.figure(figsize=(9, 6))
plt.plot(pca_components, precision, marker="o", linewidth=2, label="Precision")
plt.plot(pca_components, recall, marker="s", linewidth=2, label="Recall")
plt.plot(pca_components, f1_score, marker="^", linewidth=2, label="F1 Score")

plt.xlabel("Number of PCA Components")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1 Score Across PCA Settings")
plt.xticks(pca_components)
plt.ylim(0.65, 0.86)
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall.png")
plt.close()
# -----------------------------
# Figure 3: Confusion matrices
# -----------------------------
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, (pca, cm) in zip(axes, conf_matrices.items()):
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Pred Non-Coding", "Pred Coding"],
        yticklabels=["Actual Non-Coding", "Actual Coding"]
    )
    ax.set_title(f"PCA = {pca}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle("Confusion Matrices Across PCA Settings", y=1.05, fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
# -----------------------------
# Figure 4: Final model metrics
# -----------------------------
final_metrics = {
    "Accuracy": 0.7128,
    "Precision": 0.6851,
    "Recall": 0.8360,
    "F1 Score": 0.7531,
    "ROC AUC": 0.7993
}

plt.figure(figsize=(8, 6))
bars = plt.bar(list(final_metrics.keys()), list(final_metrics.values()))
plt.ylim(0.60, 0.90)
plt.ylabel("Score")
plt.title("Final Model Metrics (PCA = 64)")

# Add value labels on top of bars
for bar, value in zip(bars, final_metrics.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.005,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("final_metrics.png")
plt.close()