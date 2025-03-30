# An extensive evaluation of single-cell RNA-Seq contrastive learning generative networks for intrinsic cell types distribution estimation
This is a python implementation of the **scRCL-G** framework that evaluates the performance of scRNA-Seq contrastive learning generative encoder networks in terms of their performance on learning the intrinsic distributions of different cell types.
```
@article{...,
  title={An extensive evaluation of single-cell RNA-Seq contrastive learning generative networks for intrinsic cell types distribution estimation},
  author={Alsaggaf, Ibrahim and Buchan, Daniel and Wan, Cen},
  journal={...},
  pages={...},
  year={...},
  publisher={...},
  note={In preparation}
}
```

# Usage
This repository contains the implementation of the **scRCL-G** framework. The implementation is built in Python3 (version 3.10.12) using Scikit-learn and the deep learning library Pytorch. 

## Requirements
- torch==2.1.1
- scikit-learn==1.4.0

## Tutorial
To run this implementation you need to do the following steps:
1. ...
2. ...
3. Execute the following command:

```
python3 main.py\
...
```

### Examples
To run ... execute:
```
python3 main.py\
...
```

### The code
Here we briefly describe each `.py` file in the **code** folder.

`main.py` Runs the selected method.

`data.py` Reads and preprocesses the given dataset.

`losses.py` Includes the self-supvised contrastive learning loss [(Chen et al., 2020)](http://proceedings.mlr.press/v119/chen20j.html) and the supervised contrastive learning loss [(Khosla et al., 2020)](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html).

`networks.py` Includes the encoder architecture.

`utils.py` and `h5.py` Includes some helper functions.

# Availability
The single-cell RNA-Seq datasets (i.e. genes experssion matrices) used in this work can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8087611.svg)](https://doi.org/10.5281/zenodo.8087611). The cell-type annotations for those datasets can be downloaded from [Cell-type annotations](https://github.com/ibrahimsaggaf/AFRCL/tree/main/Cell-type%20annotations). The pre-trained encoders can be downloaded from ?.

# Acknowledgements
The authors acknowledge the support by the School of Computing and Mathematical Sciences and the Birkbeck GTA programme.

