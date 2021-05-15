# Principal Component Analysis

## Info

|Name|Student ID|Mail|
|---|---|---|
|Vũ Lê Thế Anh|20C13002|anh.vu2020@ict.jvn.edu.vn|


## How to run

Install the necessary libraries in `requirements.txt`

```
pip install -r requirements.txt
```

Use `main.py`

```
usage: main.py [-h] [--dataset DATASET] [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Dataset (iris|digits|wine|breast-cancer)
  --output OUTPUT, -o OUTPUT
                        Output directory (will be created if not existed)
```

## Folder & File structure

- [main.py](main.py) contains argument parsing and running scripts
- [pca.py](pca.py) contains the implementation of PCA
- [requirements.txt](requirements.txt) contains the required libraries
- [output](output) contains sample outputs