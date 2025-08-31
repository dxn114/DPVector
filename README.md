# DPVector
A scalable framework for top-K similarity queries in vector DBMS under differential privacy.

This repository contains Python codes for the paper:
> A Scalable Differentially Private Framework for Similarity Queries in Vector Databases

## Introduction

In this paper, 

## Datasets
We use 4 real-world datasets, and you can click on the `Name` to navigate to the download link.

| Name | 
| :----: |
| [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots/data) | 
| [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset/data) | 
| [ILSVRC 2012](https://www.image-net.org/download-images.php) | 
| [ImageNet-OOD](https://www.image-net.org/data/imagenetood.tar.gz) | 


## How to Run

Run the main program `pncs_difeps.py` from the command line with the required arguments:

```bash
python pncs_difeps.py \
  --n 1000 \
  --d 128 \
  --k 32 \
  --epsilon0 1.0 \
  --delta 1e-2 \
  --qnum 100 \
  --topk 10 \
  --k1 5 \
  --file_id 1
```

Argument descriptions:

- `--n`: Number of vectors
- `--d`: Vector dimension
- `--k`: Projected dimension
- `--epsilon0`: Privacy parameter 1
- `--delta`: Privacy parameter 2
- `--qnum`: Number of queries
- `--topk`: Number of nearest items to return per query
- `--k1`: Number of clusters
- `--file_id`: Dataset index (1~4, corresponding to `data1.csv`, etc.)

## Output

After running, the program will print clustering and retrieval metrics to the terminal.
