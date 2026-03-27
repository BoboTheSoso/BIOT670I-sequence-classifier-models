"""
01_extract_cds.py

Purpose:
- Read chr22.gff
- Keep only CDS (coding DNA sequence) features
- Convert GFF coordinates (1-based, inclusive) to Python coordinates (0-based, half-open)
- Merge overlapping CDS intervals
- Save merged CDS intervals to a CSV

Input:
- Data/chr22/chr22.gff

Output:
- Data/processed/chr22_cds_merged.csv

Why merge?
- GFF often contains many CDS entries that overlap (exons, transcripts, isoforms).
- Merging gives a clean set of "coding regions" for labeling.
"""

import os
import pandas as pd

GFF_PATH = "data/ncbi_dataset_GRCh38/ncbi_dataset/data/GCF_000001405.40/chr22.gff"
OUT_DIR = "Data/processed"
OUT_CDS = os.path.join(OUT_DIR, "chr22_cds_merged.csv")


def load_gff(path: str) -> pd.DataFrame:
    """Load a GFF file into a pandas DataFrame (skipping # comment lines)."""
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"],
    )


def gff_to_python_coords(start_1based: int, end_1based_inclusive: int) -> tuple[int, int]:
    """
    Convert GFF coordinates to Python slicing coordinates.

    GFF:    1-based inclusive [start, end]
    Python: 0-based half-open [start0, end0)

    Example:
      GFF start=1 end=3 => Python start0=0 end0=3 (slice seq[0:3] gives 3 bases)
    """
    start0 = start_1based - 1
    end0 = end_1based_inclusive  # end stays the same to become half-open
    return start0, end0


def merge_intervals(sorted_intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge overlapping or adjacent intervals.

    Input intervals must be sorted by start.
    Example: (10,20) and (18,25) -> merge to (10,25)
    """
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


def main():
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load the GFF
    gff = load_gff(GFF_PATH)

    # Filter to CDS only (coding DNA sequence regions)
    cds = gff[gff["type"] == "CDS"].dropna(subset=["start", "end"]).copy()

    # Convert coordinates and store
    cds["start"] = cds["start"].astype(int)
    cds["end"] = cds["end"].astype(int)

    cds["start0"], cds["end0"] = zip(*cds.apply(lambda r: gff_to_python_coords(r["start"], r["end"]), axis=1))

    # Keep only the interval columns
    cds_intervals = cds[["start0", "end0"]].sort_values(["start0", "end0"]).reset_index(drop=True)

    # Merge overlapping CDS intervals for a clean coding reference
    intervals_list = list(cds_intervals.itertuples(index=False, name=None))
    merged_list = merge_intervals(intervals_list)

    merged_df = pd.DataFrame(merged_list, columns=["start0", "end0"])
    merged_df.to_csv(OUT_CDS, index=False)

    print("Loaded GFF:", GFF_PATH)
    print("Raw CDS rows:", len(cds_intervals))
    print("Merged CDS intervals:", len(merged_df))
    print("Saved:", OUT_CDS)


if __name__ == "__main__":
    main()