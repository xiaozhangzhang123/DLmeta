# DLmeta
DLmeta is a deep learning method for metagenomic identification.<br>
## Description<br>
Dlmeta uses a model combining Convolutional Neural Network and Transformer to perform metagenomic identification. DLmeta can achieve high accuracy and recall
with different datasets, exhibiting its strong flexibility. DLmeta will help
to solve various problems in natural community ecology and
play an active role in downstream analysis.

## Requirements
DLmeta is a Python script, thus, installation is not required. However, it has the following dependencies:<br>

* Python 3,<br>
* Prodigal (https://github.com/hyattpd/Prodigal),<br>
* Hmmer (http://hmmer.org/download.html),<br>
* Pfam-A database (ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/).<br>
To work properly, DLmeta require Prodigal and hmmsearch in your PATH environment variable.<br>
## Usage
The usage of DLmeta is:<br>
``` cd bin
./DLmeta  -f <input fasta file>  -o <output_directory>  --hmm <HMM database>
```
## Retraining model
The trained model may have different effects in different databases. You can use the provided model_ training. py retrains the model with custom data. To use the retrained classifier, replace the model path in DLmeta.
## Contack Information
If you have any questions or concerns, please feel free to contact zystudy2022@outlook.com
