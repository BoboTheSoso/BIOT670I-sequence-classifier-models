# BIOT670I-sequence-classifier-models
BIOT670I sequence classifier models - group repository


This DNA classifier focuses on classifying between coding and non-coding sequences. It uses a 250bp sliding window with 100bp step, and runs best on PCA=50 (see trained-models evaluations for PCA 64, 50, 25, 10, and 2).
To run this, simply clone the repository locally and run the script 'Run_scripts.py'. If you want to use a pre-trained model, pick one from the Trained-models folder and place it in a 'Models' folder in the main repository, and rename the file 'pca_svm_model.joblib'. Otherwise, the model will be trained on PCA 50, which can take up to 2 hours, then be saved within the repo folder in 'Models'.



CNN-MGP: Convolutional Neural Networks for Metagenomics Gene Prediction [NCBI Article] (https://pubmed.ncbi.nlm.nih.gov/30588558/)
Pytorch Tutorial for CNNs [Youtube Video] (https://www.youtube.com/watch?v=pDdP0TFzsoQ)

DOMAIN KNOWLEDGE
Domain knowledge Computational Biology textbook resource: https://ocw.mit.edu/ans7870/6/6.047/f15/MIT6_047F15_Compiled.pdf

Other domain knowledge resource: https://research-ebsco-com.ezproxy.umgc.edu/c/fbrdda/ebook-viewer/pdf/bxqiwspgxj/page/pp_ii
OR https://learning.oreilly.com/library/view/bioinformatics-and-functional/9780470085851/?sso_link=yes&sso_link_from=umgc OR https://users.fmf.uni-lj.si/podgornik/download/Calladine.pdf 

# GenomeLM - DNA Sequence Classifier

## Project Overview
This project trains machine learning models (CNN, RNN, SVM) to classify human DNA sequences as coding or non-coding. The models learn patterns from 250 bp windows extracted from human chromosome 22.

## Potential Annotated Sequence GRCh38 

  Folder Structure
- `Data/chr22/`: Contains extracted chromosome 22 files
  - `chr22.fna` : FASTA sequence
  - `chr22.gff` : Annotation
- `Scripts/preprocessing/`: Python scripts for preprocessing sequences
- `Notes/`: Notes / documentation

> Note: These extracted files are small enough for GitHub, unlike the full genome. One can clone the repo and use these directly.


