# Weight Space Representation Learning on Diverse NeRF Architectures (ICLR 2026)

[![paper](https://img.shields.io/badge/arxiv-paper-darkred?logo=arxiv)](https://arxiv.org/abs/2502.09623)
[![datasets](https://img.shields.io/badge/huggingface-datasets-teal?logo=huggingface)](https://huggingface.co/datasets/frallebini/gmnerf)
[![models](https://img.shields.io/badge/huggingface-models-plum?logo=huggingface)](https://huggingface.co/frallebini/gmnerf)

![teaser](https://cvlab-unibo.github.io/gmnerf/static/images/teaser.svg)

## Installation
### Option A
Create a `conda` virtual environment and install the required libraries:
```
$ conda env create -f environment.yml
```

### Option B
If option A fails, follow instead [this guide](https://github.com/CVLAB-Unibo/nf2vec?tab=readme-ov-file#nf2vec-1) and then run:
```
$ pip install torch_geometric torch_scatter scikit-learn
```
## Data

The datasets used troughout our paper can be found [here](https://huggingface.co/datasets/frallebini/gmnerf/tree/main). Only the NeRF directory ([`nerf`](https://huggingface.co/datasets/frallebini/gmnerf/tree/main/nerf)) is strictly required to run our code, whereas graphs ([`graph`](https://huggingface.co/datasets/frallebini/gmnerf/tree/main/graph)) and embeddings ([`emb`](https://huggingface.co/datasets/frallebini/gmnerf/tree/main/emb)) can be computed from scratch with some of the scripts provided in this repo, as detailed in the following sections.

By default, our scripts look for data in the `./data` directory. Otherwise, you can set their `--data-root` command-line argument to your desired directory. The directory structure of `--data-root` is assumed to be the same as the one described [here](https://huggingface.co/datasets/frallebini/gmnerf).

## Training

Model weights can be downloaded from [here](https://huggingface.co/frallebini/gmnerf/tree/main). Once downloaded, place them in a directory called `./ckpts` and keep the directory structure described [here](https://huggingface.co/frallebini/gmnerf#directory-structure).

To train one of our models yourself, run [`train.py`](train.py) with the required command-line arguments. For example, to train the $`\mathcal{L}_\text{R+C}`$ model, run:
```
$ python train.py --loss l_rec_con --wandb-user ... --wandb-project ... --data-root ...
```
The other choices for `--loss` are `l_rec` (aka $`\mathcal{L}_\text{R}`$) and `l_con` (aka $`\mathcal{L}_\text{C}`$). 

If graphs for training and validation NeRFs are not present in `--data-root`, `train.py` will compute them before training starts. Otherwise, it will skip the graph computation step and use the graphs found in `--data-root`.

## Graph computation

NeRF graphs can be downloaded from [here](https://huggingface.co/datasets/frallebini/gmnerf/tree/main/graph). To compute them yourself, run [`export_graphs.py`](export_graphs.py) with the required command-line arguments. For example, to compute the graphs of NeRFs belonging to the test set of $`\texttt{MLP}`$, run:
```
$ python export_graphs.py --data-root ... --dataset shapenet --arch mlp --split test
```

## Embedding computation

NeRF embeddings can be downloaded from [here](https://huggingface.co/datasets/frallebini/gmnerf/tree/main/emb). To compute them yourself, download/export the corresponding NeRF graphs first (see previous section) and then run [`export_embs.py`](export_embs.py) with the required command-line arguments. For example, to compute the embeddings produced by the trained $`\mathcal{L}_\text{R+C}`$ encoder when ingesting NeRFs belonging to the test set of $`\texttt{MLP}`$, run:
```
$ python export_embs.py --ckpt_name l_rec_con --data.root ... --dataset shapenet --arch mlp --split test
```

## Classification

To perform classification, download/export the desired NeRF embeddings first (see previous section) and then run [`classify.py`](classify.py) with the required command-line arguments. For example, to train a classifier on the embeddings produced by the trained $`\mathcal{L}_\text{R+C}`$ encoder when ingesting NeRFs belonging to $`\texttt{MLP}`$, run:
```
$ python classify.py --ckpt-name l_rec_con --wandb-user ... --wandb-project ... --data-root ... --arch mlp
```

## Retrieval

To perform retrieval, download/export the desired NeRF embeddings first (see previous section) and then run [`retrieve.py`](retrieve.py) with the required command-line arguments. For example, to perform the retrieval experiment on $`\mathcal{L}_\text{R+C}`$ embeddings where query NeRFs belong to $`\texttt{MLP}`$ and gallery NeRFs belong to $`\texttt{TRI}`$, run:
```
$ python retrieve.py --ckpt-name l_rec_con --wandb-user ... --wandb-project ... --data-root ... --query-arch mlp --gallery-arch triplane
```

## Language tasks

Our NeRF captioning and retrieval results can be reproduced by running the [official LLaNA code](https://github.com/CVLAB-Unibo/LLaNA) with our `(emb, text)` dataset, where `emb` is a NeRF embedding produced by our trained $`\mathcal{L}_\text{R+C}`$ encoder and `text` is a textual annotation from [ShapeNeRF-Text](https://huggingface.co/datasets/andreamaduzzi/ShapeNeRF-Text/tree/main). This dataset can be found [here](https://huggingface.co/datasets/frallebini/gmnerf/tree/main/language). 

## Cite us

If you find our work useful, please cite us:

```bibtex
@inproceedings{ballerini2026weight,
  title = {Weight Space Representation Learning on Diverse {NeRF} Architectures},
  author = {Ballerini, Francesco and Zama Ramirez, Pierluigi and Di Stefano, Luigi and Salti, Samuele},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year = {2026}
```
