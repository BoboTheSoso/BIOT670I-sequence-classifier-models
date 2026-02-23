# Import the necessary models and library
import numpy as np 
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import product

# Function that Read Fasta file
def read_fasta_file(file_path):
    sequences = []              
    current_sequence = ""       
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                if current_sequence != "":
                    sequences.append(current_sequence.upper())
                current_sequence = ""
            else:
                current_sequence += line.strip()

        if current_sequence != "":
            sequences.append(current_sequence.upper())

    return sequences

# Check for Open Reading Frames (ORFs)
def has_orf(sequence, min_length=90):
    stop_codons = {"TAA", "TAG", "TGA"}
    
    # Check all 3 reading frames
    for frame in range(3):
        for i in range(frame, len(sequence) - 2, 3):
            if sequence[i:i+3] == "ATG":
                for j in range(i+3, len(sequence) - 2, 3):
                    if sequence[j:j+3] in stop_codons:
                        orf_length = j - i
                        if orf_length >= min_length:
                            return True
                        break
    return False

# Generate k-mer features
def k_mers(sequence, k=3):
    return [''.join(p) for p in product('ACGT', repeat=k)]

ALL_K_MERS = k_mers('ACGT', k=3)

def k_mer_features(sequence, k=3):
    counts = dict.fromkeys(ALL_K_MERS, 0)
    for i in range(len(sequence) - k + 1):
        k_mer = sequence[i:i+k]
        if k_mer in counts:
            counts[k_mer] += 1

    total = sum(counts.values())
    if total > 0:
        for k_mer in counts:
            counts[k_mer] /= total

    return list(counts.values())

# Global variable to store the trained model
svm_model = None
scaler = None
pca = None

# Load and train the model
def train_model():
    global svm_model, scaler, pca
    
    # Load the Dataset
    coding_seq = read_fasta_file("coding_sequences.fasta")
    non_coding_seq = read_fasta_file("noncoding_sequences.fasta") 

    sequences = coding_seq + non_coding_seq
    labels = [1] * len(coding_seq) + [0] * len(non_coding_seq)

    features = np.array([k_mer_features(seq) for seq in sequences]) 

    # Split the Dataset into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=27, stratify=labels)

    # Scale the Dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 

    # Dimensionality Reduction using PCA
    pca = PCA(n_components=32) # reduce to 32 dimensions
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    #Training Linear SVM model
    svm_model_linear = SVC(kernel='linear', C=1, gamma='scale', probability=True)
    svm_model_linear.fit(X_train_pca, y_train)
    y_pred_linear = svm_model_linear.predict(X_test_pca)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    print("Linear SVM Model Accuracy:", accuracy_linear)

    # Train the rbf SVM model
    svm_model_rbf = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    svm_model_rbf.fit(X_train_pca, y_train)

    y_pred = svm_model_rbf.predict(X_test_pca)
    accuracy_rbf = accuracy_score(y_test, y_pred)
    print("RBF SVM Model Accuracy:", accuracy_rbf)

    Conf_mat = confusion_matrix(y_test, y_pred)

    return svm_model_linear,svm_model_rbf, scaler, pca, accuracy_rbf, accuracy_linear, Conf_mat

# Create a simple GUI using Steamlit
st.title("🧬 DNA Coding vs Non-Coding Classifier")

svm_model_linear, svm_model_rbf, scaler, pca, accuracy_rbf, accuracy_linear, Conf_mat = train_model()

st.subheader("Model Performance")
st.write(f"Linear SVM Accuracy: {accuracy_linear:.4f}")
st.write(f"RBF SVM Accuracy: {accuracy_rbf:.4f}")
st.write("This model uses SVM + PCA to classify DNA sequences.")

st.subheader("Confusion Matrix")
st.write(Conf_mat)

st.subheader("PCA Explained Variance Ratio")
st.write(f"Total Variance Retained: {np.sum(pca.explained_variance_ratio_):.4f}")


sequence_input = st.text_area("Please, Enter DNA Sequence (A, C, G, T only):")

if st.button("Classify"):
    sequence_input = sequence_input.strip().upper()

    if sequence_input == "":
        st.warning("Please enter a DNA sequence.")
    
    elif not all(nuc in "ACGT" for nuc in sequence_input):
        st.error("Invalid characters entered! Only A, C, G, T are allowed.")
    
    elif not has_orf(sequence_input):
        st.warning("No Open Reading Frame (ORF) detected! The sequence is likely non-coding.")


    else:
        features = np.array(k_mer_features(sequence_input)).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        result = svm_model_rbf.predict(features_pca)[0]
        probability = svm_model_rbf.predict_proba(features_pca)[0][result]

        if result == 1:
            st.success("Prediction: Coding")
        else:
            st.error("Prediction: Non-Coding")

        st.write(f"Confidence: {probability:.4f}")