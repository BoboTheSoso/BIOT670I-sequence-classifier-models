"""
GUI Script with Visualization
Name: 06_Runscripts_GUI_Visual.py

Features:
- Loads or trains model automatically
- Accepts FASTA input
- Uses sliding window (250bp, step 100)
- Predicts coding vs non-coding
- Displays confidence
- Visualizes predictions across sequence
"""

import os
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog
import joblib
import itertools
from collections import Counter
import threading
import matplotlib.pyplot as plt

# Custom modules
import BIO670SVM as train
import prep

prep.main()
# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "Models" / "BIO670SVM.joblib"

# -----------------------------------------------------------
# Global model
# -----------------------------------------------------------
model = None

# -----------------------------------------------------------
# Loading Animation
# -----------------------------------------------------------
animation_running = False

def animate_label(label, i=0):
    spinner = ['|', '/', '-', '\\']
    if animation_running:
        label.config(text=f"Loading {spinner[i % len(spinner)]}")
        label.after(100, animate_label, label, i+1)

def start_spinner():
    global animation_running
    animation_running = True
    animate_label(result_label_animation)

def stop_spinner():
    global animation_running
    animation_running = False
    result_label_animation.config(text="Model Ready")

# -----------------------------------------------------------
# Load or Train Model
# -----------------------------------------------------------
def load_or_train_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        result_label.config(text="Model loaded successfully. Ready to classify.")
    except:
        start_spinner()

        def task():
            global model
            try:
                result_label.after(0, lambda: result_label.config(
                    text="No model found.\nRunning preprocessing..."
                ))

                prep.main()

                result_label.after(0, lambda: result_label.config(
                    text="Preprocessing done.\nTraining model (this may take time)..."
                ))

                train.train_model()

                model = joblib.load(MODEL_PATH)

                result_label.after(0, lambda: result_label.config(
                    text="Model trained and loaded.\nReady to classify."
                ))

            except Exception as e:
                result_label.after(0, lambda: messagebox.showerror("Error", str(e)))

            finally:
                result_label.after(0, stop_spinner)

        threading.Thread(target=task, daemon=True).start()

    root.after(0, lambda: button.config(state="normal"))

# -----------------------------------------------------------
# K-mer Setup
# -----------------------------------------------------------
K = 3
WINDOW_SIZE = 250
STEP_SIZE = 100

ALL_KMERS = [''.join(p) for p in itertools.product("ACGT", repeat=K)]
KMER_INDEX = {kmer: i for i, kmer in enumerate(ALL_KMERS)}

# -----------------------------------------------------------
# K-mer Feature Vector
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# Sliding Window Prediction
# -----------------------------------------------------------
def windowed_kmer_preds(seq: str):
    if len(seq) < WINDOW_SIZE:
        raise ValueError(f"Sequence must be at least {WINDOW_SIZE} bp.")

    preds, probs, positions = [], [], []

    for start in range(0, len(seq) - WINDOW_SIZE + 1, STEP_SIZE):
        window = seq[start:start + WINDOW_SIZE]

        features = kmer_vector(window).reshape(1, -1)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]

        preds.append(pred)
        probs.append(prob)
        positions.append(start)

    final_label = int(np.round(np.mean(preds)))
    confidence = float(np.mean(probs))

    return final_label, confidence, len(preds), positions, preds

# -----------------------------------------------------------
# Plot Visualization
# -----------------------------------------------------------
def plot_predictions(positions, preds):
    plt.figure()
    plt.scatter(positions, preds)
    plt.yticks([0, 1], ["Non-coding", "Coding"])
    plt.xlabel("Sequence Position (bp)")
    plt.ylabel("Prediction")
    plt.title("Sliding Window Classification Across DNA Sequence")
    plt.grid()
    plt.show()

# -----------------------------------------------------------
# File Handling
# -----------------------------------------------------------
def fileCheck(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError("File not found.")

    with open(filepath, "r") as f:
        lines = f.readlines()

    seq = "".join(line.strip() for line in lines if not line.startswith(">")).upper()

    if not all(base in "ACGT" for base in seq):
        raise ValueError("Invalid DNA sequence detected.")

    return seq

# -----------------------------------------------------------
# Classification Function
# -----------------------------------------------------------
def classify_seq():
    filepath = filedialog.askopenfilename()

    try:
        filename = os.path.basename(filepath)
        sequence = fileCheck(filepath)

        label_num, prob, count, positions, preds = windowed_kmer_preds(sequence)
        label = "Coding" if label_num == 1 else "Non-coding"

        result_label.config(
            text=f"File: {filename}\n"
                 f"Length: {len(sequence)} bp\n"
                 f"Windows: {count}\n"
                 f"Prediction: {label}\n"
                 f"Confidence: {prob:.3f}"
        )

        # 📊 Show visualization
        plot_predictions(positions, preds)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# -----------------------------------------------------------
# GUI Setup
# -----------------------------------------------------------
root = tk.Tk()
root.title("DNA Classifier (CDS vs NCDS)")
root.geometry("500x320")
root.resizable(False, False)

title = tk.Label(root, text="DNA Sequence Classifier", font=("Arial", 16))
title.pack(pady=10)

result_label_animation = tk.Label(root, text="")
result_label_animation.pack(pady=5)

result_label = tk.Label(root, text="", justify="left")
result_label.pack(pady=10)

button = tk.Button(root, text="Select FASTA File", command=classify_seq)
button.pack(pady=10)
button.config(state="disabled")

# Load model
load_or_train_model()

root.mainloop()