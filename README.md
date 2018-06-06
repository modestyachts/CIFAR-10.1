# CIFAR-10.1
This repository contains the release of the CIFAR-10.1 dataset, a new test set for CIFAR-10.
We describe the creation of the dataset in the paper ["Do CIFAR-10 classifiers generalize to CIFAR-10?"](https://arxiv.org/abs/1806.00451).

There are two versions of the CIFAR-10.1 dataset:
- `default` is the recommended dataset for future experiments and corresponds to the results in Appendix D of our paper.
- `v0` is the first version of our dataset. The numbers reported in the main section of our paper use the `v0` dataset.

The `datasets` directory contains the dataset files:
- The `default` files are `cifar10.1-data.npy` and `cifar10.1-labels.npy`.
- The `v0` files are `cifar10.1-v0-data.npy` and `cifar10.1-v0-labels.npy`.
The `notebooks` directory contains a short script to browse the CIFAR-10.1 dataset.
The `code` directory contains a `utils` file to help load the dataset.

To cite this dataset please use both references:
```
@article{recht2018cifar10.1,
  author = {Benjamin Recht and Rebecca Roelofs and Ludwig Schmidt and Vaishaal Shankar},
  title = {Do CIFAR-10 Classifiers Generalize to CIFAR-10?},
  year = {2018},
  note = {\url{https://arxiv.org/abs/1806.00451}},
}
@article{torralba2008tinyimages, 
  author = {A. Torralba and R. Fergus and W. T. Freeman}, 
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title = {80 Million Tiny Images: A Large Data Set for Nonparametric Object and Scene Recognition}, 
  year = {2008}, 
  volume = {30}, 
  number = {11}, 
  pages = {1958-1970}
}
```
