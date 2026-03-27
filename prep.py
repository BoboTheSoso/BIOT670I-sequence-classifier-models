import extract_cds
import make_window_label
import balance_split
import kmer_features

def main():
    extract_cds.main()
    make_window_label.main()
    balance_split.main()
    kmer_features.main()