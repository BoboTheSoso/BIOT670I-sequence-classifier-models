'''
PCA + SVM Training with Nested Cross-validation
Name: pca_svm_training.py
Purpose:
Train a Support Vector Machine (SVM) classifier on k-mer features with PCA dimensionality reduction.
Includes nested cross-validation for robust hyperparameter tuning and model evaluation.
Steps:
1. Load preprocessed k-mer features and labels from .npy files generated in previous steps
2. Create a machine learning pipeline that includes:
    - StandardScaler for feature scaling
    - PCA for dimensionality reduction (n_components=50 for speed)
    - SVC for classification
3. Define a parameter grid for hyperparameter tuning of the SVM (linear, RBF, polynomial kernels)
4. Perform nested cross-validation:
   - Outer loop for model evaluation
   - Inner loop for hyperparameter tuning using GridSearchCV
5. Train the final model on the entire training set using the best hyperparameters
6. Evaluate the final model on the validation and test sets using various metrics (accuracy, precision, recall, F1 score, ROC AUC)
7. Save the trained model using joblib for future use

Input:
- X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy (k-mer features and labels)
Output:
- Trained SVM model saved as pca_svm_model.joblib

'''

#Import core libraries
import numpy as np
import joblib #for saving the model after training?
import os
from pathlib import Path
import json

#Processing and PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#model
from sklearn.svm import SVC #supports the 3 kernel methods

#pipeline
from sklearn.pipeline import Pipeline

#model selection for cross val
from sklearn.model_selection import GridSearchCV, StratifiedKFold

#metrics and visualization
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, RocCurveDisplay
import matplotlib.pyplot as plt


#-----------------------------------------------------------
#Define all paths
#-----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "kmer_k3"
print(PROJECT_ROOT)
print(f"kmer dir path: {DATA_DIR}")

X_TRAIN_PATH = DATA_DIR / "X_Train.npy"
Y_TRAIN_PATH = DATA_DIR / "y_Train.npy"
X_VAL_PATH = DATA_DIR / "X_Validation.npy"
Y_VAL_PATH = DATA_DIR / "y_Validation.npy"

def train_model():
    #-----------------------------------------------------------
    #Load Data from the previous step + param grid
    #-----------------------------------------------------------

    #Load features into matrix
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)

    X_val = np.load(X_VAL_PATH)
    y_val = np.load(Y_VAL_PATH)

    X_test = np.load(DATA_DIR / "X_Testing.npy")
    y_test = np.load(DATA_DIR / "y_Testing.npy")

    #Double checking shapes
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    #-----------------------------------------------------------
    # Pipeline
    #-----------------------------------------------------------
    #Create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=50)), #reduce to 50 components for speed while retaining high accuracy and precision
        ('svc', SVC(probability=True))
    ])
    print("Pipeline created.")

    #-----------------------------------------------------------
    # Parameter grid
    #-----------------------------------------------------------

    param_grid = [
        {'svc__kernel': ['linear'], 'svc__C': [0.1, 1, 10]}, #3
        {'svc__kernel': ['rbf'], 'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 0.01, 0.1]}, #9
        {'svc__kernel': ['poly'], 'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 0.01, 0.1], 'svc__degree': [2, 3]} #27
    ]
    print("Parameter grid created.")

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

        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
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
    print(f"Best parameters based on evaluation: {best_params_list}")

    #-----------------------------------------------------------
    # Train model
    #-----------------------------------------------------------

    final_grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
    final_grid.fit(X_train, y_train)
    final_model = final_grid.best_estimator_
    print("Final model training completed. Moving to evaluation steps.")

    #-----------------------------------------------------------
    # Evaluation method + Saving results
    #-----------------------------------------------------------
    #File path for metrics (and later Model)
    MODEL_DIR = PROJECT_ROOT / "Models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    #List of all results
    results = {}

    def evaluate_step(split_name, y_true, y_pred, y_proba=None):
        results['best_params'] = final_grid.best_params_
        results['accuracy'] = float(accuracy_score(y_true, y_pred))
        results['precision'] = float(precision_score(y_true, y_pred, zero_division = 0))
        results['recall'] = float(recall_score(y_true, y_pred, zero_division = 0))
        results['f1_Score'] = float(f1_score(y_true, y_pred, zero_division = 0))

        if y_proba is not None:
            results['ROC AUC'] = float(roc_auc_score(y_true, y_proba))
        
        if y_proba is not None:
            #ROC Curve plot
            plt.figure()
            RocCurveDisplay.from_predictions(y_true, y_proba)
            plt.title(f'Roc Curve for {split_name}')
            plt.savefig(MODEL_DIR / f'{split_name}_roc_curve.png') #saved as png
            plt.close()


        #Saving metrics in Json file
        with open(MODEL_DIR / f'{split_name}_metrics.json', 'w') as f:
            json.dump(results, f, indent = 4)

        #saving confusion matrix as a numpy matrix file
        cm =  confusion_matrix(y_true, y_pred)
        np.save(MODEL_DIR / f'{split_name}_confusion_matrix.npy', cm)

        #Saving the classification report as a text file
        cr = classification_report(y_true, y_pred)
        with open(MODEL_DIR / f'{split_name}_classification_report.txt', 'w') as f:
            f.write(cr)

        print(f'{split_name} results saved.')

    #-----------------------------------------------------------
    # Validation evaluation
    #-----------------------------------------------------------

    y_val_pred = final_model.predict(X_val)
    y_val_proba = final_model.predict_proba(X_val)[:, 1] #probabilities for ROC AUC

    #Evaluation metrics
    print("Validation set evaluation:")
    evaluate_step('Validation_Set', y_val, y_val_pred, y_val_proba)

    #-----------------------------------------------------------
    # Test evaluation
    #-----------------------------------------------------------
 
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1] #probabilities for ROC AUC

    print("Test set evaluation:")
    evaluate_step('Test set', y_test, y_test_pred, y_test_proba)


    #-----------------------------------------------------------
    # SAVE MODEL WITH JOBLIB
    #-----------------------------------------------------------

    #Target directory will depend on GitHub structure and where you want to save the model
    
    joblib.dump(final_model, f"{MODEL_DIR}/pca_svm_model.joblib")
    print(f"Model saved in path: {MODEL_DIR}")
