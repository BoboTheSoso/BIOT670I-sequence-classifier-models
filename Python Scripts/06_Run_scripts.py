'''
This script is only to run the preprocessing script and the pca+svm training script.
Add a GUI
'''
import os
import subprocess
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.filedialog import askopenfilename
import joblib
import itertools
from collections import Counter

# Run the preprocessing script
#subprocess.run(["python", "01-04_Data_preprocessing_Scripts.py"])

# Run the PCA + SVM training script
#subprocess.run(["python", "05_pca_svm_training.py"])

#Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "pca_svm_model.joblib"

#-----------------------------------------------------------
# Load model
#-----------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model not found or failed to load: {e}")

#-----------------------------------------------------------
# Method for kmer vector-ing
#-----------------------------------------------------------

K = 3
ALL_KMERS = [''.join(p) for p in itertools.product("ACGT", repeat=K)]
KMER_INDEX = {kmer: i for i, kmer in enumerate(ALL_KMERS)}

def kmer_vector(seq: str) -> np.ndarray:

    seq = str(seq).upper()
    vec = np.zeros(len(ALL_KMERS), dtype=float)


    if len(seq) < K:
        return vec


    counts = Counter(
    seq[i:i+K]
    for i in range(len(seq) - K + 1)
    if set(seq[i:i+K]).issubset({"A", "C", "G", "T"})
    )


    total = sum(counts.values())
    if total == 0:
        return vec


    for mer, c in counts.items():
        idx = KMER_INDEX.get(mer)
        if idx is not None:
            vec[idx] = c / total

    return vec

#-----------------------------------------------------------
# File check + read
#-----------------------------------------------------------

def fileCheck (input_seq):
    if not os.path.exists(input_seq):
        raise FileNotFoundError(f"File not found: {input_seq}")
    with open(input_seq, "r") as file:
        contents = file.read().strip()
    #check for 'AGCT'
    if not all(base in 'ATCG' for base in contents): #Do we include blank characters? '-', '*', ' ' and all that
        raise ValueError("Not a DNA sequence")
    return contents

def classify_seq():
    #root = tk.Tk()
    input_seq = filedialog.askopenfilename()

    try:
        sequence = fileCheck(input_seq)
        features = kmer_vector(sequence)

        prediction = model.predict(features)
        print("Prediction for input sequence: " + prediction)

    except Exception as e:
        messagebox.showerror("Error", str(e))

#-----------------------------------------------------------
# GUI time
#-----------------------------------------------------------
root = tk.Tk()
root.title("DNA Classifier for CDS and NCDS")

label = tk.Label(root, text="DNA Sequence Classifier")
label.pack()
button = tk.Button(root, text = "Select input sequence", command= classify_seq)

root.mainloop


'''
Pseudocode:

Get all paths
GUI pops up to let user select file from their machine
scripts processes the file: kmer breakage np for model plug-in
uses pre-saved model from joblib to give a prediction
test evaluation grid thing + final prediction coding vs non-coding
'''
