'''
This script is only to run the preprocessing script and the pca+svm training script.
Add a GUI
'''

import subprocess
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import joblib


# Run the preprocessing script
#subprocess.run(["python", "01-04_Data_preprocessing_Scripts.py"])

# Run the PCA + SVM training script
#subprocess.run(["python", "05_pca_svm_training.py"])

#Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "pca_svm_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model not found or failed to load: {e}")

def dna_feat(sequence, k=3):
    sequence = sequence.upper()
    kmers={}

    for i in range(len(sequence)-k+1):
        kmer=sequence[i:i+k]
        if kmer not in kmers:
            kmers[kmer]=0
        kmers[kmer]+=1
    
    all_kmers = sorted(kmers.keys())
    feature_vec = np.array([kmers[k] for k in all_kmers])

    return feature_vec.reshape(1,-1)
