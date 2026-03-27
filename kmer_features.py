
"""
Purpose:
*Load train/val/test CSVs produced by 03_balance_and_split.py
*Convert DNA sequences into k-mer frequency vectors 
*Save X/y arrays to .npy files for SVM training
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import itertools
from collections import Counter

#File Paths
IN_TRAIN = "Data/processed/train.csv" #Balanced training windows
IN_VAL   = "Data/processed/val.csv" #Validation window
IN_TEST  = "Data/processed/test.csv" #Test windows

#Output Folder for K-mer features
OUT_DIR = "Data/processed/kmer_k3"
#Create folder / directory
os.makedirs(OUT_DIR, exist_ok=True)

#Implement Kmer
K = 3
#From A C G T, generate all possible combinations
ALL_KMERS = [''.join(p) for p in itertools.product("ACGT", repeat=K)]
#Create dictionary to align each k-mer to numeric index position
KMER_INDEX = {kmer: i for i, kmer in enumerate(ALL_KMERS)}

#Function to convert sequence to k-mer vector 
def kmer_vector(seq: str) -> np.ndarray:
    """Convert a DNA sequence into a normalized k-mer frequency vector.
    Output will be a NumPy array of length 4^k to display relative frequency of k-mer
    """
    #Ensure sequence is string and uppercase
    seq = str(seq).upper()

    #Initialize vector with zeros
    vec = np.zeros(len(ALL_KMERS), dtype=float)
    #If sequence length less than k, halts k-mer formation
    if len(seq) < K:
        return vec

    #Count valid k-mers 
    counts = Counter(
        #Extract substring of k length
        seq[i:i+K] 
        for i in range(len(seq) - K + 1)
        #Skip any k-mer containing N or non-ACGT
        if set(seq[i:i+K]).issubset({"A", "C", "G", "T"})
    )

    #Store total number of valid k-mers in sequence
    total = sum(counts.values())

    #If no valid k-mers found, return zero vector
    if total == 0:
        return vec

    #Add vectors with normalize frequencies
    for mer, c in counts.items():
        #Retreive numeric index
        idx = KMER_INDEX.get(mer)
        #Store normalized frequency 
        if idx is not None:
            vec[idx] = c / total 

    return vec

def featurize_split(csv_path: str, split_name: str) -> None:
    #Load CSV file (train/val/test)
    df = pd.read_csv(csv_path)

    #Ensure valid columns exist
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'sequence' and 'label' columns. Found: {list(df.columns)}")

    #Extract sequences as list
    sequences = df["sequence"].astype(str).tolist()
    #Extract labels as numeric array
    y = df["label"].astype(int).to_numpy()

    #Convert each sequence into a k-mer feature vector and put in matrix shape 
    X = np.vstack([kmer_vector(s) for s in sequences])

    #Save feature matrix and labels
    np.save(os.path.join(OUT_DIR, f"X_{split_name}.npy"), X)
    np.save(os.path.join(OUT_DIR, f"y_{split_name}.npy"), y)

    #Display dataset summary
    print(f"{split_name}: X shape = {X.shape}, y shape = {y.shape}, class balance = {np.bincount(y)}")

def main():
    featurize_split(IN_TRAIN, "train")
    featurize_split(IN_VAL, "val")
    featurize_split(IN_TEST, "test")
    print("Saved k-mer features to:", OUT_DIR)

if __name__ == "__main__":
    main()
