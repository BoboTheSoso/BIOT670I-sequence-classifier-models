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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
# =====================================================
# STEP 1: LOAD PREPROCESSED K-MER FEATURES
# =====================================================

DATA_DIR = "Data/processed/kmer_k3"

# Load feature matrices
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))

# Load labels
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Shape check
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)
# =====================================================
# STEP 2: SCALE FEATURES
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


# =====================================================
# STEP 4: PCA DIMENSIONALITY REDUCTION
# =====================================================

pca = PCA(n_components=42)
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

#-----------------------------------------------------------
# Pipeline

#-----------------------------------------------------------
# Define a pipeline for scaling, PCA, and SVM (for future use in GridSearchCV)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=42)), #reduce to 42 components for speed
    ('svc', SVC(probability=True))
])


#-----------------------------------------------------------
# Parameter grid
#-----------------------------------------------------------

param_grid = [
    {'svc__kernel': ['linear'], 'svc__C': [0.1, 1, 10]},
    {'svc__kernel': ['rbf'], 'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 0.01, 0.1]},
    {'svc__kernel': ['poly'], 'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 0.01, 0.1], 'svc__degree': [2, 3]}
]


#-----------------------------------------------------------
# Nested Cross-validation with GridSearchCV
#-----------------------------------------------------------

#Outer CV for model evaluation, inner CV for hyperparameter tuning
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_scores = []
best_params_list = []

for outer_train_idx, outer_val_idx in outer_cv.split(X_train, y_train):
    X_train_outer, y_train_outer = X_train[outer_train_idx], y_train[outer_train_idx]
    X_val_outer, y_val_outer = X_train[outer_val_idx], y_train[outer_val_idx]

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1) #check roc auc types options for binary vs multiclass
    grid_search.fit(X_train_outer, y_train_outer)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best params for this fold: {best_params}, Best inner CV AUC: {best_score:.4f}")

    best_params_list.append(best_params)

    # Evaluate the best model on the outer validation set
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val_outer)
    val_accuracy = accuracy_score(y_val_outer, y_val_pred)
    print(f"Outer CV accuracy for this fold: {val_accuracy:.4f}")
    outer_scores.append(val_accuracy)

print(f"\nOverall Outer CV Accuracy: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")

#-----------------------------------------------------------
# Train model
#-----------------------------------------------------------

final_grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
final_grid.fit(X_train, y_train)
final_model = final_grid.best_estimator_
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
# STEP 8: SAVE MODELS USING JOBLIB
# =====================================================

os.makedirs("saved_models", exist_ok=True)

joblib.dump(svm_linear, "saved_models/svm_linear.pkl")
joblib.dump(svm_rbf, "saved_models/svm_rbf.pkl")
joblib.dump(svm_poly, "saved_models/svm_poly.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(pca, "saved_models/pca.pkl")

print("\nModels saved successfully.")