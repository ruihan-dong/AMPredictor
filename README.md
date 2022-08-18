# AMPredictor

Antimicrobial peptide (AMP) activity prediction



## Dependencies

Python 3

PyTorch

PyG

RDKit

ESM

sklearn



## Usage for prediction

###### 01 Prepare input files (./input/)

* input.fasta  # the fasta file of input peptide sequences
* input.txt  # name/sequence/label, the label can be any number when predicting unknown MICs

Please carefully check the name of each sequence in input.fasta is consistent with that in input.txt, otherwise the predicted results will be wrong.

Two simple scripts can help you to transform these two input files when there is one of them:

* fasta2txt.py (Biopython required)
* txt2fasta.py

###### 02 Preprocess

The first step is to transform a peptide into ESM embedding and its contact map.

The input of this script is input.fasta

```python
python preprocess.py
```

The pretrained ESM-1b model would be downloaded automatically the first time operating, which may takes a long time. 

###### 03 Run predicting

```python
python predict.py
```

Then the predicted values will be printed, as well as saved in output.csv.

Note: If the input file name need to be changed, please edit AMPredictor.py.



## Usage for training

###### 01 Prepare input files and preprocess

This preparation step is same as mentioned above. 

The AMP datasets we used for training, validation and testing are in ./data folder.

###### 02 Run training

```python
python train.py
```

###### 03 Run testing

```python
python test.py
```

Metrics including RMSE, MSE, Pearson, Spearman and CI will be printed.



## References

[DGraphDTA](https://github.com/595693085/DGraphDTA), [SummaryDTA](https://github.com/PuYuQian/SummaryDTA), [PepVAE](https://www.frontiersin.org/articles/10.3389/fmicb.2021.725727/full), [Peptimizer](https://github.com/learningmatter-mit/peptimizer)

