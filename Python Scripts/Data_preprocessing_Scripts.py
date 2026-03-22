"""
* Step 01: Extract and merge CDS intervals from chr22.gff
* Step 02: Create fixed-length windows from chr22.fna and label by CDS overlap
* Step 03: Balance classes and split into train/validation/testing sets
* Step 04: Convert DNA sequences into k-mer frequency vectors for SVM training


Input Files:
* data/ncbi_dataset_GRCh38/ncbi_dataset/data/GCF_000001405.40/chr22.fna
* data/ncbi_dataset_GRCh38/ncbi_dataset/data/GCF_000001405.40/chr22.gff


Output Files:
* data/processed/chr22_cds_merged.csv
* data/processed/chr22_windows_250_labeled_all.csv
* data/processed/chr22_windows_250_labeled_balanced.csv
* data/processed/train.csv
* data/processed/val.csv
* data/processed/test.csv
* data/processed/kmer_k3/X_train.npy
* data/processed/kmer_k3/y_train.npy
* data/processed/kmer_k3/X_val.npy
* data/processed/kmer_k3/y_val.npy
* data/processed/kmer_k3/X_test.npy
* data/processed/kmer_k3/y_test.npy
"""


from __future__ import annotations


from pathlib import Path
import itertools
from collections import Counter


import numpy as np
import pandas as pd
from Bio import SeqIO



PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATA_DIR = PROJECT_ROOT / "data"
GENOME_DIR = DATA_DIR / "ncbi_dataset_GRCh38" / "ncbi_dataset" / "data" / "GCF_000001405.40"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


GFF_PATH = GENOME_DIR / "chr22.gff"
FASTA_PATH = GENOME_DIR / "chr22.fna"


print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_DIR:", DATA_DIR)
print("GENOME_DIR:", GENOME_DIR)
print("GFF_PATH:", GFF_PATH)
print("FASTA_PATH:", FASTA_PATH)


OUT_CDS = PROCESSED_DIR / "chr22_cds_merged.csv"


OUT_ALL = PROCESSED_DIR / "chr22_windows_250_labeled_all.csv"


OUT_BAL = PROCESSED_DIR / "chr22_windows_250_labeled_balanced.csv"
OUT_TRAIN = PROCESSED_DIR / "train.csv"
OUT_VAL = PROCESSED_DIR / "val.csv"
OUT_TEST = PROCESSED_DIR / "test.csv"


KMER_OUT_DIR = PROCESSED_DIR / "kmer_k3"
KMER_OUT_DIR.mkdir(parents=True, exist_ok=True)


#Ensure valid file loaded
def check_required_inputs() -> None:
    if not GFF_PATH.exists():
        raise FileNotFoundError(f"GFF file Not Found: {GFF_PATH}")
    if not FASTA_PATH.exists():
        raise FileNotFoundError(f"FASTA file Not Found: {FASTA_PATH}")



#Load gff file into DataFrame (skipping comment lines)
def load_gff(path: Path) -> pd.DataFrame:
    return pd.read_csv(
    path,
    sep="\t",
    comment="#",
    header=None,
    names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"],
)


#Convert GFF coordinates (1-based inclusive) to Python coordinates (0-based half-open)
def gff_to_python_coords(start_1based: int, end_1based_inclusive: int) -> tuple[int, int]:
    start0 = start_1based - 1
    end0 = end_1based_inclusive
    return start0, end0


# Merge overlapping / adjacent intervals
#Input intervals must already be sorted by start coordinates.
def merge_intervals(sorted_intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for s, e in sorted_intervals:
        if not merged:
            merged.append([s, e])
            continue


        prev_s, prev_e = merged[-1]

        # If this interval starts after the previous one ends, start a new merged interval
        if s > prev_e:
            merged.append([s, e])
        else:
            # Otherwise overlap/adjacent: extend previous interval end
            merged[-1][1] = max(prev_e, e)


    return [(s, e) for s, e in merged]


#Read chr22.gff file, extract CDS features, merge CDS intervals
def run_01_extract_cds() -> None:

    gff = load_gff(GFF_PATH)
    #filter to CDS only
    cds = gff[gff["type"] == "CDS"].dropna(subset=["start", "end"]).copy()

    
    cds["start"] = pd.to_numeric(cds["start"], errors="coerce")
    cds["end"] = pd.to_numeric(cds["end"], errors="coerce")
    cds = cds.dropna(subset=["start", "end"])

    #Convert coordinates and store
    cds["start"] = cds["start"].astype(int)
    cds["end"] = cds["end"].astype(int)


    #Remove invalid intervals
    cds = cds[(cds["start"] > 0) & (cds["end"] > cds["start"])].copy()


    cds["start0"], cds["end0"] = zip(*cds.apply(lambda r: gff_to_python_coords(r["start"], r["end"]), axis=1))


    cds_intervals = cds[["start0", "end0"]].sort_values(["start0", "end0"]).reset_index(drop=True)


    intervals_list = list(cds_intervals.itertuples(index=False, name=None))
    merged_list = merge_intervals(intervals_list)


    merged_df = pd.DataFrame(merged_list, columns=["start0", "end0"])
    merged_df.to_csv(OUT_CDS, index=False)


    print("Step 01 Complete")
    print("Loaded GFF:", GFF_PATH)
    print("Raw CDS Rows:", len(cds_intervals))
    print("Merged CDS Intervals:", len(merged_df))
    print("Saved:", OUT_CDS)
    print()


#CHECKPOINT
#Define fixed-length, non-overlapping windows
WINDOW = 250
STEP = 250 
MAX_N_FRAC = 0.05

#Load FASTA file ensuring all DNA string charachters are uppercase 
def load_sequence(fasta_path: Path) -> str:
    record = SeqIO.read(fasta_path, "fasta")
    return str(record.seq).upper()


#Label each window (including CDS overlap)
def label_windows_by_overlap(windows: np.ndarray, cds_intervals: np.ndarray) -> np.ndarray:

    #Overlap when window [ws, we) overlaps cds [cs, ce) if cs < we AND ce > ws
    labels = np.zeros(len(windows), dtype=np.int8)


    j = 0
    for i, (ws, we) in enumerate(windows):
        while j < len(cds_intervals) and cds_intervals[j][1] <= ws:
            j += 1


        k = j
        while k < len(cds_intervals) and cds_intervals[k][0] < we:
            cs, ce = cds_intervals[k]
            if cs < we and ce > ws:
                labels[i] = 1
                break
            k += 1


    return labels


#Creation and labeling of fixed-length windows
def run_02_windows_labels() -> None:
    seq = load_sequence(FASTA_PATH)
    L = len(seq)


    cds_df = pd.read_csv(OUT_CDS)
    cds_intervals = cds_df[["start0", "end0"]].to_numpy()


    starts = np.arange(0, L - WINDOW + 1, STEP, dtype=int)
    ends = starts + WINDOW
    win_arr = np.column_stack([starts, ends])


    labels = label_windows_by_overlap(win_arr, cds_intervals)


    df = pd.DataFrame({
        "window_id": [f"chr22:{s}-{e}" for s, e in zip(starts, ends)],
        "start0": starts,
        "end0": ends,
        "label": labels,
    })


    df["sequence"] = [seq[s:e] for s, e in zip(df["start0"], df["end0"])]


    n_frac = df["sequence"].str.count("N") / WINDOW
    df = df[n_frac <= MAX_N_FRAC].reset_index(drop=True)


    df.to_csv(OUT_ALL, index=False)


    print("Step 02 Complete")
    print("Loaded FASTA:", FASTA_PATH)
    print("Sequence Length (base pairs):", L)
    print("Saved:", OUT_ALL)
    print("Label Counts:", df["label"].value_counts().to_dict())
    print()



#Train/Vali/Test Set Splits
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_STATE = 42


REQUIRED_COLS = {"window_id", "start0", "end0", "label", "sequence"}


#Ensure and validate the input DataFram
def assert_expected_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Found: {list(df.columns)}")


#Balance classes by undersampling majority class
def balance_by_undersampling(df: pd.DataFrame) -> pd.DataFrame:
    coding = df[df["label"] == 1]
    noncoding = df[df["label"] == 0]


    if len(coding) == 0 or len(noncoding) == 0:
        raise ValueError(f"Cannot balance. coding={len(coding)} noncoding={len(noncoding)}")


    if len(noncoding) > len(coding):
        noncoding_sample = noncoding.sample(n=len(coding), random_state=RANDOM_STATE)
        balanced = pd.concat([coding, noncoding_sample], ignore_index=True)
    else:
        coding_sample = coding.sample(n=len(noncoding), random_state=RANDOM_STATE)
        balanced = pd.concat([coding_sample, noncoding], ignore_index=True)


        balanced = balanced.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return balanced


#Split into Train/Val/Test based on genomic coordinates order
def split_by_coordinate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("start0").reset_index(drop=True)
    n = len(df_sorted)


    n_train = int(TRAIN_FRAC * n)
    n_val_end = int((TRAIN_FRAC + VAL_FRAC) * n)


    train = df_sorted.iloc[:n_train].reset_index(drop=True)
    val = df_sorted.iloc[n_train:n_val_end].reset_index(drop=True)
    test = df_sorted.iloc[n_val_end:].reset_index(drop=True)


    return train, val, test


#Balance labeled windows and creation of Train/Valid/Test splits 
def run_03_balance_and_split() -> None:
    df = pd.read_csv(OUT_ALL)
    assert_expected_columns(df)


    df["label"] = df["label"].astype(int)
    df["start0"] = df["start0"].astype(int)
    df["end0"] = df["end0"].astype(int)


    print("Loaded:", OUT_ALL)
    print("Total Windows:", len(df))
    print("Class Counts (before):", df["label"].value_counts().to_dict())


    balanced = balance_by_undersampling(df)
    print("Class Counts (balanced):", balanced["label"].value_counts().to_dict())


    balanced.to_csv(OUT_BAL, index=False)
    print("Saved:", OUT_BAL)


    train, val, test = split_by_coordinate(balanced)


    train.to_csv(OUT_TRAIN, index=False)
    val.to_csv(OUT_VAL, index=False)
    test.to_csv(OUT_TEST, index=False)


    print("Step 03 Complete")
    print("Saved Splits:")
    print(" Train:", OUT_TRAIN, "rows:", len(train))
    print(" Val: ", OUT_VAL, "rows:", len(val))
    print(" Test: ", OUT_TEST, "rows:", len(test))
    print("Training Label Counts:", train["label"].value_counts().to_dict())
    print("Validation Label Counts:", val["label"].value_counts().to_dict())
    print("Testing Label Counts:", test["label"].value_counts().to_dict())
    print()



#Defining k-mer features
K = 3
ALL_KMERS = [''.join(p) for p in itertools.product("ACGT", repeat=K)]
KMER_INDEX = {kmer: i for i, kmer in enumerate(ALL_KMERS)}


IN_TRAIN = OUT_TRAIN
IN_VAL = OUT_VAL
IN_TEST = OUT_TEST


#Conversion of DNA sequences into normalized k-mer frequency vector 
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


#Loading of each split, conversion of sequences into k-mer featurers
def featurize_split(csv_path: Path, split_name: str) -> None:
    df = pd.read_csv(csv_path)


    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'sequence' and 'label' columns. Found: {list(df.columns)}")


    sequences = df["sequence"].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()


    X = np.vstack([kmer_vector(seq) for seq in sequences])


    np.save(KMER_OUT_DIR / f"X_{split_name}.npy", X)
    np.save(KMER_OUT_DIR / f"y_{split_name}.npy", y)


    print(f"{split_name}: X shape = {X.shape}, y shape = {y.shape}, class balance = {np.bincount(y)}")


#Conversion of train/val/test windows sequences into k-mer feature vector matrices
def run_04_kmers() -> None:
    featurize_split(IN_TRAIN, "Train")
    featurize_split(IN_VAL, "Validation")
    featurize_split(IN_TEST, "Testing")


    print("Step 04 Complete")
    print("Saved k-mer features to:", KMER_OUT_DIR)
    print()



#Main Functions
def main() -> None:
    check_required_inputs()
    run_01_extract_cds()
    run_02_windows_labels()
    run_03_balance_and_split()
    run_04_kmers()
    print("✅ Full preprocessing pipeline complete.")



if __name__ == "__main__":
    main() 
