#Import core libraries
import numpy as np
import joblib #for saving the model after training

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

#-------------------------------------------
#Load Data from the previous step
#-------------------------------------------

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

#-------------------------------------------
# Pipeline with StandardScaler, PCA, and SVM
#-------------------------------------------

#Make a parameter grid for tuning the SVM hyperparameters
param_grid = [
    {'svc_kernel': ['linear'], 'svc_C': [0.1, 1, 10]},
    {'svc_kernel': ['rbf'], 'svc_C': [0.1, 1, 10], 'svc_gamma': ['scale', 0.01, 0.1]},
    {'svc_kernel': ['poly'], 'svc_C': [0.1, 1, 10], 'svc_gamma': ['scale', 0.01, 0.1], 'svc_degree': [2, 3]}
]

#Outer CV for model evaluation, inner CV for hyperparameter tuning
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

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
    
    #Save the model
    model_filename = f"svm_{params['svc_kernel']}_C{params['svc_C']}"
    if 'svc_gamma' in params:
        model_filename += f"_gamma{params['svc_gamma']}"
    if 'svc_degree' in params:
        model_filename += f"_degree{params['svc_degree']}"
    
    joblib.dump(pipeline, f"Models/{model_filename}.joblib")
    print(f"Saved model to Models/{model_filename}.joblib")
