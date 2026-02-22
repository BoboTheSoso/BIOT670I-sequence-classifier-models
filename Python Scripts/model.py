import itertools
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from LoadingData import sequence_coding_regions

######
# Feature Extraction
######
def generate_kmers(k):
    return [''.join(p) for p in itertools.product('ACGT', repeat=k)]

def kmer_frequency(sequence, k=3):
    kmers = generate_kmers(k)
    counts = dict.fromkeys(kmers, 0)

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in counts:
            counts[kmer] += 1

    total_kmers = sum(counts.values())
    if total_kmers > 0:
        for key in counts:
            counts[key] /= total_kmers
    return list(counts.values())

def gc_content(sequence):
    return (sequence.count('G') + sequence.count('C')) / len(sequence) if len(sequence) > 0 else 0

def extract_features(df):
    kmer_feat = kmer_frequency(sequence, k=3)
    gc_feat = [gc_content(sequence)]
    return kmer_feat + gc_feat

######
# 1. Data Preparation
######

#Get sequence from system file path GUI
file_path = 'x' #open a window to select a file
sequence = open(file_path).read().strip()
labels = sequence_coding_regions['type'].values
#Pull the data from the LoadingData.py file and extract features
#also add GC content and ORF length as features

x = np.array([extract_features(seq) for seq in sequence_coding_regions['attributes']])
y = np.array(labels)

######
# 2. Traint/Test Split
######

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify = y)


######
# 3. Pipeline scaling PCA and SVM
######

#Parameters for SVM
kernels = ['linear', 'rbf', 'poly']
C_values = [0.1, 1, 10]

results = []


for kernel in kernels:
    for C in C_values:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)), #retains 95% of variance
            ('svm', svm.SVC(kernel=kernel, C=C, gamma='scale')) #SVM with RBF kernel
        ])

        #cross validation accuracy
        cv_scores = cross_val_score(pipeline, x, y, cv=5)
        
        ######
        # 4. Model Training and Testing
        ######

        pipeline.fit(X_train, y_train)
        #test predictions
        y_pred = pipeline.predict(X_test)

        #Results
        results.append({
            'kernel': kernel,
            'C': C,
            'cv_accuracy': cv_scores.mean(),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        })
#Print results
for res in results:
    print(f"Kernel: {res['kernel']}, C: {res['C']}, CV Accuracy: {res['cv_accuracy']:.4f}, Test Accuracy: {res['test_accuracy']:.4f}, Precision: {res['precision']:.4f}, Recall: {res['recall']:.4f}, F1 Score: {res['f1_score']:.4f}")
