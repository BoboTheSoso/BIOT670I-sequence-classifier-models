"""
DNA Coding vs Non-Coding Classification
---------------------------------------

This script:

1. Loads preprocessed train/val/test datasets
2. Generates 3-mer (k=3) frequency features
3. Scales features
4. Applies PCA (dimensionality reduction)
5. Trains three SVM models:
      - Linear kernel
      - RBF kernel
      - Polynomial kernel
6. Evaluates models using:
      - Accuracy
      - Precision
      - Recall
      - F1-score
      - Confusion Matrix
7. Saves trained models using joblib to avoid retraining in the future.
"""

import os
import numpy as np
import pandas as pd
import joblib
from itertools import product
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =====================================================
# STEP 1: K-MER FEATURE EXTRACTION
# =====================================================

def generate_all_kmers(k=3): #Generate all possible k-mers using A,C,G,T. For k=3 → 4^3 = 64 features.
    
    return [''.join(p) for p in product('ACGT', repeat=k)]


ALL_KMERS = generate_all_kmers(3)


def k_mer_features(sequence, k=3): #Convert a DNA sequence into normalized k-mer frequency features.
    
    counts = dict.fromkeys(ALL_KMERS, 0)

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in counts:
            counts[kmer] += 1

    total = sum(counts.values())

    # Normalize to frequency
    if total > 0:
        for kmer in counts:
            counts[kmer] /= total

    return list(counts.values())


# =====================================================
# STEP 2: LOAD DATA
# =====================================================

TRAIN_PATH = "Data/processed/train.csv"
VAL_PATH   = "Data/processed/val.csv"
TEST_PATH  = "Data/processed/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

# Extract features and labels
X_train = np.array([k_mer_features(seq) for seq in train_df["sequence"]])
y_train = train_df["label"].values

X_val = np.array([k_mer_features(seq) for seq in val_df["sequence"]])
y_val = val_df["label"].values

X_test = np.array([k_mer_features(seq) for seq in test_df["sequence"]])
y_test = test_df["label"].values


# =====================================================
# STEP 3: SCALE FEATURES
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


# =====================================================
# STEP 4: PCA DIMENSIONALITY REDUCTION
# =====================================================

pca = PCA(n_components=32)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print("Total Variance Retained by PCA:", np.sum(pca.explained_variance_ratio_))


# =====================================================
# STEP 5: TRAIN SVM MODELS
# =====================================================

# Linear Kernel
svm_linear = SVC(kernel='linear', C=1, probability=True)
svm_linear.fit(X_train_pca, y_train)

# RBF Kernel
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_rbf.fit(X_train_pca, y_train)

# Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3, C=1, probability=True)
svm_poly.fit(X_train_pca, y_train)


# =====================================================
# STEP 6: EVALUATION FUNCTION
# =====================================================

def evaluate_model(model, X, y, name="Model"): #Evaluate model performance and print metrics.
    
    y_pred = model.predict(X)

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec  = recall_score(y, y_pred)
    f1   = f1_score(y, y_pred)
    cm   = confusion_matrix(y, y_pred)

    print("\n==============================")
    print(f"{name} Performance")
    print("==============================")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))
    print("Confusion Matrix:\n", cm)

    return acc, prec, rec, f1


# =====================================================
# STEP 7: VALIDATION PERFORMANCE
# =====================================================

evaluate_model(svm_linear, X_val_pca, y_val, "Linear SVM (Validation)")
evaluate_model(svm_rbf, X_val_pca, y_val, "RBF SVM (Validation)")
evaluate_model(svm_poly, X_val_pca, y_val, "Polynomial SVM (Validation)")




# =====================================================
# STEP 8: SAVE MODELS USING JOBLIB
# =====================================================

os.makedirs("saved_models", exist_ok=True)

joblib.dump(svm_linear, "saved_models/svm_linear.pkl")
joblib.dump(svm_rbf, "saved_models/svm_rbf.pkl")
joblib.dump(svm_poly, "saved_models/svm_poly.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(pca, "saved_models/pca.pkl")

print("\nModels saved successfully.")