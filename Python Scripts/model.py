import itertools
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


######
# 3. Pipeline scaling PCA and SVM
######

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)), #retains 95% of variance
    ('svm', svm.SVC(kernel='rbf', C=1, gamma='scale')) #SVM with RBF kernel
])

######
# 4. Training
######

pipeline.fit(X_train, y_train)

######
# 5. Evaluation
######

y_pred = pipeline.predict(X_test)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

######
# 6. Cross validation
######

scores = cross_val_score(pipeline, x, y, cv=5)
print('\nCross-validation accuracy: ', scores.mean())
