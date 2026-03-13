#Import core libraries
import numpy as np
import joblib #for saving the model after training?

#Processing and PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#model
from sklearn.svm import SVC #supports the 3 kernel methods

#pipeline
from sklearn.pipeline import Pipeline

#model selection for cross val
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid, cross_val_score

#metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#-----------------------------------------------------------
#Load Data from the previous step + param grid
#-----------------------------------------------------------

DATA_DIR = "Data/processed/kmer=3"

#Load features into matrix
X_train = np.load(f"{DATA_DIR}/X_train.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")

X_val = np.load(f"{DATA_DIR}/X_val.npy")
y_val = np.load(f"{DATA_DIR}/y_val.npy")

X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

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
    ('pca', PCA(n_components=50)), #reduce to 50 components for speed
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

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='roc_auc') #check roc auc types options for binary vs multiclass
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

final_grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='roc_auc')
final_grid.fit(X_train, y_train)
final_model = final_grid.best_estimator_

#-----------------------------------------------------------
# Evaluation method
#-----------------------------------------------------------

def evaluate_step(model, set_name, set_pred):
    print('Accuracy: ', round(accuracy_score(set_name, set_pred), 4))
    print('Precision: ', round(precision_score(set_name, set_pred), 4))
    print('Recall: ', round(recall_score(set_name, set_pred), 4))
    print('F1 Score: ', round(f1_score(set_name, set_pred), 4))
    print('ROC AUC: ', round(roc_auc_score(set_name, set_pred), 4))

    print('\nConfusion Matrix:\n', confusion_matrix(set_name, set_pred))
    print('\nClassification Report:\n', classification_report(set_name, set_pred))



#-----------------------------------------------------------
# Validation evaluation
#-----------------------------------------------------------

y_val_pred = final_model.predict(X_val)
y_val_proba = final_model.predict_proba(X_val)[:, 1] #probabilities for ROC AUC

#Evaluation metrics
print("Validation set evaluation:")
evaluate_step(y_val, y_val_pred)

#-----------------------------------------------------------
# Test evaluation
#-----------------------------------------------------------
 
y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)[:, 1] #probabilities for ROC AUC

print("Test set evaluation:")
evaluate_step(y_test, y_test_pred)


#-----------------------------------------------------------
# SAVE MODEL WITH JOBLIB
#-----------------------------------------------------------

#Target directory will depend on GitHub structure and where you want to save the model
MODEL_DIR = "Models"
joblib.dump(final_model, f"{MODEL_DIR}/pca_svm_model.joblib")
