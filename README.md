# AMPredictor

preprocess and inference
```shell
python predict.py sample.fasta sample.fasta.tsv
```

**environment**
- python 3.11.9
- requirements.txt

## Old readme

Antimicrobial peptide (AMP) activity prediction
* Online server: [click here](https://huggingface.co/spaces/ruihan-dong/AMPredictor)  (please submit one peptide sequence < 60 AAs at a time)
![image](https://github.com/ruihan-dong/AMPredictor/blob/main/AMPredictor_framework.png)


## Dependencies

* Python3
* PyTorch
* PyG
* RDKit
* ESM
* sklearn



## Usage for prediction

### 01 Prepare input files (./input/)

* input.fasta   (the fasta file of input peptide sequences)
* input.txt     (name/sequence/label, the label can be any number when predicting unknown MICs)

Please carefully check that the name of each sequence in input.fasta is consistent with which in input.txt, otherwise the predicted results will be wrong.

Two simple scripts can help you to transform these two input files when one of them is available:

* fasta2txt.py (Biopython required)
* txt2fasta.py

### 02 Preprocess

The first step is to transform a peptide sequence into ESM embedding and obtain its contact map.

The input of this script is input.fasta

```python
python preprocess.py
```

The pretrained ESM-1b model would be downloaded automatically the first time operating, which may take a long time. 

### 03 Run predicting

```python
python predict.py
```

Then the predicted values will be printed, as well as saved in output.csv.

Note: If the input file name need to be changed, please edit AMPredictor.py.



## Usage for training

### 01 Prepare input files and preprocess

This preparation step is same as mentioned above. 

The AMP datasets we used for training, validation and testing are in ./data folder.

### 02 Run training

```python
python train.py
```

### 03 Run testing

```python
python test.py
```

Metrics including RMSE, MSE, Pearson, Spearman and CI will be printed.



## References

[DGraphDTA](https://github.com/595693085/DGraphDTA), [SummaryDTA](https://github.com/PuYuQian/SummaryDTA), [PepVAE](https://www.frontiersin.org/articles/10.3389/fmicb.2021.725727/full), [Peptimizer](https://github.com/learningmatter-mit/peptimizer)


## To cite
```
@article{Dong2024.02.23.581845,
  author = {Ruihan Dong and Rongrong Liu and Ziyu Liu and Yangang Liu and Gaomei Zhao and Honglei Li and Shiyuan Hou and Xiaohan Ma and Huarui Kang and Jing Liu and Fei Guo and Ping Zhao and Junping Wang and Cheng Wang and Xingan Wu and Sheng Ye and Cheng Zhu},
  title = {Exploring the repository of de novo designed bifunctional antimicrobial peptides through deep learning},
  elocation-id = {2024.02.23.581845},
  year = {2024},
  doi = {10.1101/2024.02.23.581845},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2024/02/24/2024.02.23.581845},
  journal = {bioRxiv}
}
```
