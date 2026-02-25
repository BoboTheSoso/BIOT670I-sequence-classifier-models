"""
02_windows_labels.py

Purpose:
- Read chr22.fna (FASTA sequence)
- Read merged CDS intervals from 01_extract_cds.py
- Create fixed-length windows across chr22 (250 bp)
- Label each window:
    label=1 if it overlaps ANY CDS interval
    label=0 otherwise
- Save labeled windows to CSV

Inputs:
- Data/chr22/chr22.fna
- Data/processed/chr22_cds_merged.csv

Output:
- Data/processed/chr22_windows_250_labeled_all.csv
"""

import os
import numpy as np
import pandas as pd
from Bio import SeqIO

FASTA_PATH = "Data/chr22/chr22.fna"
CDS_PATH = "Data/processed/chr22_cds_merged.csv"
OUT_DIR = "Data/processed"
OUT_ALL = os.path.join(OUT_DIR, "chr22_windows_250_labeled_all.csv")

WINDOW = 250
STEP = 250          # STEP=WINDOW means NON-overlapping windows
MAX_N_FRAC = 0.05   # drop windows that are >5% 'N' (ambiguous bases)


def load_sequence(fasta_path: str) -> str:
    """Load a single FASTA record as an uppercase DNA string."""
    record = SeqIO.read(fasta_path, "fasta")
    return str(record.seq).upper()


def label_windows_by_overlap(windows: np.ndarray, cds_intervals: np.ndarray) -> np.ndarray:
    """
    Label each window using CDS overlap.

    windows: Nx2 array of [start0, end0]
    cds_intervals: Mx2 array of [start0, end0] (merged, sorted)

    Overlap rule:
      window [ws,we) overlaps cds [cs,ce) if cs < we AND ce > ws
    """
    labels = np.zeros(len(windows), dtype=np.int8)

    # Two-pointer sweep for efficiency
    j = 0  # CDS interval pointer
    for i, (ws, we) in enumerate(windows):
        # Advance CDS pointer if CDS ends before window starts
        while j < len(cds_intervals) and cds_intervals[j][1] <= ws:
            j += 1

        # Check overlap with current CDS interval and following ones that might overlap
        k = j
        while k < len(cds_intervals) and cds_intervals[k][0] < we:
            cs, ce = cds_intervals[k]
            if cs < we and ce > ws:
                labels[i] = 1
                break
            k += 1

    return labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load chr22 DNA sequence
    seq = load_sequence(FASTA_PATH)
    L = len(seq)

    # Load merged CDS intervals
    cds_df = pd.read_csv(CDS_PATH)
    cds_intervals = cds_df[["start0", "end0"]].to_numpy()

    # Create windows across chr22
    starts = np.arange(0, L - WINDOW + 1, STEP, dtype=int)
    ends = starts + WINDOW
    win_arr = np.column_stack([starts, ends])

    # Label windows
    labels = label_windows_by_overlap(win_arr, cds_intervals)

    # Build output DataFrame
    df = pd.DataFrame({
        "window_id": [f"chr22:{s}-{e}" for s, e in zip(starts, ends)],
        "start0": starts,
        "end0": ends,
        "label": labels
    })

    # Extract the actual DNA sequence for each window
    df["sequence"] = [seq[s:e] for s, e in zip(df["start0"], df["end0"])]

    # Filter out windows with too many Ns (ambiguous bases)
    n_frac = df["sequence"].str.count("N") / WINDOW
    df = df[n_frac <= MAX_N_FRAC].reset_index(drop=True)

    df.to_csv(OUT_ALL, index=False)

    print("Loaded FASTA:", FASTA_PATH)
    print("Sequence length (bp):", L)
    print("Saved:", OUT_ALL)
    print("Label counts:", df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()