'''
This script is only to run the preprocessing script and the pca+svm training script.
Add a GUI
'''

import subprocess

# Run the preprocessing script
subprocess.run(["python", "01-04_Data_preprocessing_Scripts.py"])

# Run the PCA + SVM training script
subprocess.run(["python", "05_pca_svm_training.py"])
