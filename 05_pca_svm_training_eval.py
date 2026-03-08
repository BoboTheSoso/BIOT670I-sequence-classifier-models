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

#---------------------------------------------
#Load Data from the previous step + param grid
#---------------------------------------------

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

#Make a parameter grid for tuning the SVM hyperparameters
param_grid = [
    {'svc_kernel': ['linear'], 'svc_C': [0.1, 1, 10]},
    {'svc_kernel': ['rbf'], 'svc_C': [0.1, 1, 10], 'svc_gamma': ['scale', 0.01, 0.1]},
    {'svc_kernel': ['poly'], 'svc_C': [0.1, 1, 10], 'svc_gamma': ['scale', 0.01, 0.1], 'svc_degree': [2, 3]}
]

#-----------------------
#Set up cross-validation
#-----------------------

#Outer CV for model evaluation, inner CV for hyperparameter tuning
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#function for nested cross-validation with GridSearchCV on inner loop
def inner_cv_grid_search(X, y, pipeline, param_combination):
    scores = []
    for train_idx, val_idx in inner_cv.split(X, y):
        X_train_inner, y_train_inner = X[train_idx], y[train_idx]
        X_val_inner, y_val_inner = X[val_idx], y[val_idx]
        
        pipeline.set_params(**param_combination)
        pipeline.fit(X_train_inner, y_train_inner)
        y_pred_inner = pipeline.predict(X_val_inner)
        scores.append(accuracy_score(y_val_inner, y_pred_inner))
    return np.mean(scores)



#----------------------------------------------------------
# Define the parameter grid for SVM hyperparameters to test
#-----------------------------------------------------------

#Outer loop for cv
for outer_train_idx, outer_val_idx in outer_cv.split(X_train, y_train):
    X_train_outer, y_train_outer = X_train[outer_train_idx], y_train[outer_train_idx]
    X_val_outer, y_val_outer = X_train[outer_val_idx], y_train[outer_val_idx]

    best_score = 0
    best_params = None

    #Inner loop for hyperparameter tuning
    for params in ParameterGrid(param_grid):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=50)), #reduce to 50 components for speed
            ('svc', SVC(probability=True))
        ])
       #pipeline.set_params(**params)
        score = inner_cv_grid_search(X_train_outer, y_train_outer, pipeline, params)
        if score > best_score:
            best_score = score
            best_params = params

    print(f"Best parameters for this fold: {best_params} with inner CV accuracy: {best_score:.4f}")

#for loop to test each kernel type and its hyperparameters
for params in ParameterGrid(param_grid):
    print(f"\nTraining SVM with parameters: {params}")
    
    #Create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)), #reduce to 50 components for speed
        ('svc', SVC(probability=True))
    ])
    #hyperparameters
    pipeline.set_params(**params)

    
    #----------------------------------
    #Nested Stratified Cross-validation
    #----------------------------------

    outer_scores = cross_val_score(pipeline, X_train, y_train, cv=outer_cv, scoring='accuracy')
    print(f"Outer CV accuracy: {outer_scores.mean():.4f} ± {outer_scores.std():.4f}")

    #---------------
    #Train the model
    #---------------

    pipeline.fit(X_train, y_train)
    
    #----------
    #Evaluation
    #----------

    #Evaluate on validation set
    y_val_pred = pipeline.predict(X_val)
    print("Validation set:")
    print(classification_report(y_val, y_val_pred))
    
    #Evaluate on test set
    y_test_pred = pipeline.predict(X_test)
    print("Test set:")
    print(classification_report(y_test, y_test_pred))
