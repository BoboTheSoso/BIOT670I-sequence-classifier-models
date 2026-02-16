from Bio import SeqIO
import pandas as pd
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
#Remove NaNs
sequence_coding_regions = sequence_coding_regions.dropna(subset=["start", "end"])

print(sequence_coding_regions.head())
print(sequence_coding_regions.shape)
