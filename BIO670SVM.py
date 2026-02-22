# Import the necessary models and library
import numpy as np 
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Global variable to store the trained model
svm_model = None
scaler = None
pca = None
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

    required_length = max_length * 4
    if len(encoded_sequences) < required_length:
        encoded_sequences += [0] * (required_length - len(encoded_sequences))
    else:            
        encoded_sequences = encoded_sequences[:required_length]
    return encoded_sequences

# Load and train the model
def train_model():
    global svm_model, scaler, pca
    
    # Load the Dataset
    coding_seq = read_fasta_file("coding_sequences.fasta")
    non_coding_seq = read_fasta_file("noncoding_sequences.fasta") 

    sequences = coding_seq + non_coding_seq
    labels = [1] * len(coding_seq) + [0] * len(non_coding_seq)

    features = np.array([one_hot_encode(seq) for seq in sequences]) 

    # Scale the Dataset
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features) 

    # Dimensionality Reduction using PCA
    pca = PCA(n_components=100) # reduce to 100 dimensions

    features_pca = pca.fit_transform(features_scaled)   

    print("Original features shape:", features.shape[1])
    print("Reduced features shape after PCA:", features_pca.shape[1])


    # Split the Dataset into Training and Testing sets  
    X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.2, random_state=27, stratify=labels)

    # Train the SVM model
    svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("SVM Model Accuracy:", accuracy)

    return svm_model, scaler, pca, accuracy 

# Create a simple GUI using Tkinter
st.title("🧬 DNA Coding vs Non-Coding Classifier")

st.write("This model uses SVM + PCA to classify DNA sequences.")

svm_model, scaler, pca, accuracy = train_model()

st.success(f"Model Accuracy: {accuracy:.4f}")

sequence_input = st.text_area("Enter DNA Sequence (A, C, G, T only):")

if st.button("Classify"):
    if sequence_input.strip() == "":
        st.warning("Please enter a DNA sequence.")
    else:
        encoded = np.array(one_hot_encode(sequence_input.upper())).reshape(1, -1)
        encoded_scaled = scaler.transform(encoded)
        encoded_pca = pca.transform(encoded_scaled)

        result = svm_model.predict(encoded_pca)[0]
        probability = svm_model.predict_proba(encoded_pca)[0][result]

        if result == 1:
            st.success(f"Prediction: Coding")
        else:
            st.error(f"Prediction: Non-Coding")

        st.write(f"Confidence: {probability:.4f}")
