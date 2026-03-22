'''
GUI script. 
Name: 06_Run_scripts.py

Purpose:
GUI script to launch a window and use the pre-trained model from script 05 to predict a sequence inputed by the user.
Sequence is broken into 3-mers and loaded into the model, then a prediction is made alongside a confidence level.

250bp window check with a 100bp sliding window step
-Option to train the model using scripts 01-04 and 05 (in case the joblib does not exist)



'''
import os
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.filedialog import askopenfilename
import joblib
import itertools
from collections import Counter


#Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "Models" / "pca_svm_model.joblib"


#-----------------------------------------------------------
# Method to train/re-train the model
#-----------------------------------------------------------

def load_or_train_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except:
        result_label.config(text="No existing model found, initiating pre-processing script. This will take time.")
        root.update()
        with open("01-04_Data_preprocessing_Scripts.py") as preprocessing_step:
            exec(preprocessing_step.read())
            result_label.config(text="Preprocessing step completed. Initiating model training step.")
            root.update()
        with open("05_pca_svm_training.py") as training_step:
            exec(training_step.read())
            result_label.config(text="Model training completed, thank you for your patience. Loading model now...")
            root.update()

        model = joblib.load(MODEL_PATH)


#-----------------------------------------------------------
# Window sliding and kmer function calling
#-----------------------------------------------------------    

K = 3
WINDOW_SIZE = 250
STEP_SIZE = 100
ALL_KMERS = [''.join(p) for p in itertools.product("ACGT", repeat=K)]
KMER_INDEX = {kmer: i for i, kmer in enumerate(ALL_KMERS)}

def windowed_kmer_preds(seq:str, wind=WINDOW_SIZE, step=STEP_SIZE):
    if len(seq)<wind:
        raise ValueError(f"Sequence too short. Must be at least {wind}bp, current length: {len(seq)}")
    wind_preds = []
    wind_probs = []
    wind_count=0
    for start in range(0, len(seq)-wind+1, step):
        end = start+wind
        wind_seq = seq[start:end]
        features = kmer_vector(wind_seq).reshape(1, -1)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]
        wind_preds.append(pred)
        wind_probs.append(prob)
        wind_count+=1

    majority_label = int(np.round(np.mean(wind_preds)))
    avg_prob = float(np.mean(wind_probs))
    return majority_label, avg_prob, wind_count


#-----------------------------------------------------------
# Break input into kmer size 3
#-----------------------------------------------------------

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
        lines = file.readlines()
    
    #removes any FASTA headers + join sequences
    seq = "".join(line.strip() for line in lines if not line.startswith(">")).upper()
    
    
    #check for 'AGCT'
    while not all(base in 'ACGT' for base in seq):
        raise ValueError("No valid DNA sequence found")
    return seq


#-----------------------------------------------------------
# Sequence classification time
#-----------------------------------------------------------

def classify_seq():
    
    input_seq = filedialog.askopenfilename()

    try:
        filename = os.path.basename(input_seq)
        sequence = fileCheck(input_seq)
        label_num, prob, count = windowed_kmer_preds(sequence)
        label = "Coding" if label_num == 1 else "Non-coding"
        result_label.config(text=f"File Name: {filename}\nSequence length: {len(sequence)}\nWindows evaluated: {count}\nPrediction: {label}\nConfidence: {prob:.3f}")
        root.update()
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

#-----------------------------------------------------------
# GUI time
#-----------------------------------------------------------
root = tk.Tk()
load_or_train_model()
root.title("DNA Classifier for CDS and NCDS")
root.geometry("500x300")
root.resizable(False,False)
label = tk.Label(root, text="DNA Sequence Classifier")
label.pack(pady=10)
button2 = tk.Button(root, text = "Train the model (this will take a while...)", command= load_or_train_model)
button = tk.Button(root, text = "Select input sequence", command= classify_seq)
button.pack(pady=10)
result_label = tk.Label(root, text="")
result_label.pack(pady=20)
root.mainloop()
