from Bio import SeqIO
import pandas as pd
import random
import numpy as np
from collections import Counter
import itertools

##Load Data###
#Read in Chromosome 22 Sequence (FASTA)
chrom_seq = SeqIO.read("chr22.fna", "fasta")
sequence = str(chrom_seq.seq)

#Read in Chromosome 22 (GFF)
seq_gff = pd.read_csv("chr22.gff", sep="\t", comment="#", header=None,
                     names=["seqid","source","type","start","end","score","strand","phase","attributes"])

##Remove duplicates, NaNs, and extract CDS##
#Filter for only coding regions (CDS)
sequence_coding_regions = seq_gff[seq_gff["type"] == "CDS"]

#Remove duplicates
sequence_coding_regions = sequence_coding_regions.drop_duplicates(subset=["start", "end"])

#Remove NaNs
sequence_coding_regions = sequence_coding_regions.dropna(subset=["start", "end"])

print(sequence_coding_regions.head())
print(sequence_coding_regions.shape)

##Remove invalid genomic ranges and convert to integers##

#Convert start and end coordinates to numeric to ensure valid numbers
sequence_coding_regions["start"] = pd.to_numeric(sequence_coding_regions["start"], errors="coerce")
sequence_coding_regions["end"] = pd.to_numeric(sequence_coding_regions["end"], errors="coerce")

#Drop any rows that became NaN after conversion
sequence_coding_regions = sequence_coding_regions.dropna(subset=["start", "end"])

#Convert to integers
sequence_coding_regions["start"] = sequence_coding_regions["start"].astype(int)
sequence_coding_regions["end"] = sequence_coding_regions["end"].astype(int)

#Remove biologically invalid ranges (such as start positions that are 0 or negative and ensure valid end position)
sequence_coding_regions = sequence_coding_regions[
    (sequence_coding_regions["start"] > 0) &
    (sequence_coding_regions["end"] > sequence_coding_regions["start"])
]
#Sort by coordinates (smallest genomic position to largest)
sequence_coding_regions = sequence_coding_regions.sort_values(["start", "end"]).reset_index(drop=True)
print(" CDS rows after cleaning:")
print(sequence_coding_regions.head())
print(sequence_coding_regions.shape)

#Extract coding sequences
coding_sequences = []

for index, row in sequence_coding_regions.iterrows():
    start = row["start"]
    end = row["end"]

    # GFF is 1 based indexing
    cds_seq = sequence[start-1:end]

    coding_sequences.append(cds_seq)

print("Total CDS sequences extracted:")
print(len(coding_sequences))

#Extract non-coding sequences
noncoding_sequences = []

cds_ranges = list(zip(sequence_coding_regions["start"], sequence_coding_regions["end"]))

prev_end = 1

for start, end in sorted(cds_ranges):
    if start > prev_end:
        noncoding_seq = sequence[prev_end-1:start-1]
        if len(noncoding_seq) > 50:
            noncoding_sequences.append(noncoding_seq)
    prev_end = end + 1

print("Total Non-Coding sequences extracted:")
print(len(noncoding_sequences))

# Make coding class equal to noncoding class
random.seed(42) #To keep consistent when testing
coding_sequences_balanced = random.sample(coding_sequences, len(noncoding_sequences))

print("Balanced Coding Sequences:")
print(len(coding_sequences_balanced))

print("Non-Coding Sequences:")
print(len(noncoding_sequences))


#Combine sequences and labels
all_sequences = coding_sequences_balanced + noncoding_sequences
y = np.array([1]*len(coding_sequences_balanced) + [0]*len(noncoding_sequences))

#Build kmer
k = 3
all_kmers = [''.join(p) for p in itertools.product("ACGT", repeat=k)] #generates all possible combinations of length 3
kmer_index = {kmer: i for i, kmer in enumerate(all_kmers)} #creates dictionary 

def kmer_vector(seq, k=3): #define kmer vector function
    seq = seq.upper().replace("N", "")  #converts to uppercase and dros unknown bases if present
    counts = Counter(seq[i:i+k] for i in range(len(seq)-k+1) if len(seq[i:i+k]) == k) #extracts 3 mers and counts frequency of when it appers in sequence
    vec = np.zeros(len(all_kmers), dtype=float) #creates zero vector 
    total = sum(counts.values()) #sum of total kmers founds
    if total > 0:
        for mer, c in counts.items():
            if mer in kmer_index:
                vec[kmer_index[mer]] = c / total   #stores normalized frequency
    return vec

#Convert all sequences into feature matrix X
X = np.vstack([kmer_vector(seq, k) for seq in all_sequences])

print("Feature matrix shape (X):", X.shape)
print("Label vector shape (y):", y.shape)
print("Class balance:", np.bincount(y))
