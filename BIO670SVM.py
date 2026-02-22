# Import the necessary models and library
import numpy as np 
import tkinter as tk
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

# Read Fasta file
def read_fasta_file(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequences = ""
        for line in file:
            if line.startswith('>'):
                if sequences != "":
                    sequences.append(sequences)
                    sequences = ""
            else:
                sequences += line.strip()
        if sequences != "":
            sequences.append(sequences.upper())
    return sequences

# One-hot encoding of sequences
def one_hot_encode(sequences, max_length = 250):
    encoding = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
    }
encoded_sequences = []
    for nucleotide in sequences:
        encoded_sequences.extend(encoding.get(nucleotide, [0, 0, 0, 0]))