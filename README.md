# CIFAR-10.1
This repository contains the CIFAR-10.1 dataset, which is a new test set for [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

CIFAR-10.1 contains roughly 2,000 new test images that were sampled *after* multiple years of research on the original CIFAR-10 dataset.
The data collection for CIFAR-10.1 was designed to minimize distribution shift relative to the original dataset.
We describe the creation of CIFAR-10.1 in the paper ["Do CIFAR-10 Classifiers Generalize to CIFAR-10?"](https://arxiv.org/abs/1806.00451). 
The images in CIFAR-10.1 are a subset of the [TinyImages dataset](http://horatio.cs.nyu.edu/mit/tiny/data/index.html). 

# Dataset Releases

There are currently two versions of the CIFAR-10.1 dataset:

- `v4` is the first version of our dataset on which we tested any classifier (hence the `v4` dataset is independent of the classifiers we evaluate). The numbers reported in the main section of our paper use the `v4` dataset. The `v4` dataset was built from the top 25 TinyImage keywords for each class, which led to a slight class imbalance (the largest difference is that ships make up only 8% of the test set instead of 10%). `v4` contains 2,021 images.

- `v6` is derived from a slightly improved keyword allocation that is exactly class balanced. This version of the dataset corresponds to the results in Appendix D of our paper. `v6` contains 2,000 images.

The overlap between `v4` and `v6` is more than 90% of the respective datasets.
Moreover, the classification accuracies are very close (see Appendix D of our paper).
For future experiments, we recommend the `v6` version of our dataset.

The `datasets` directory contains the dataset files:
- The `v4` files are `cifar10.1_v4_data.npy` and `cifar10.1_v4_labels.npy`.
- The `v6` files are `cifar10.1_v6_data.npy` and `cifar10.1_v6_labels.npy`.

The `notebooks` directory contains a short script `inspect_dataset_simple.ipynb` to browse the CIFAR-10.1 dataset.
The notebook uses a utility function to load the dataset from `utils.py` in the the `code` directory.

# Dataset Creation Pipeline

WARNING: This is currently work in progress, some parts may be incomplete.


We include code to make it possible for others to replicate the creation of the new dataset. 
The dataset creation process has several stages:

1. **TinyImages**
2. **Keywords for CIFAR-10**
3. **Unique Keywords**

4. **Keyword counts for the new dataset.**  
* `generate_keyword_counts.ipynb` decides which keywords we want to include in the new dataset and determines the number of images we require for each of these keywords. 

5. **Labeling new images.**

6. **Double checking labeled images.** 
* `labeling_ui_subselect.ipynb` allows a second person to confirm the initial labelings and subselect a pool of labeled TinyImage indices.

7. **Sampling new images from the pool of labeled images.** 
* `sample_subselected_indices_v4.ipynb` samples the pool of labeled images and creates the new dataset for v4
* `sample_subselected_indices.ipynb` samples the pool of labeled images and creates the new dataset for v6 or v7

8. **Inspect the new dataset.**
* `inspect_dataset_simple.ipynb` is a simple notebook to browse the new dataset. 

9. **Inspect model predictions.**
* `inspect_model_predictions.ipynb` explores the model predictions made on the new test set and displays a dataframe including the original and new accuracy for each model. 


# Other Data

Metadata needed to create the new datasets can be downloaded from an s3 bucket using the `other_data/download.py` script.
The script requires Boto 3, which can be installed via pip: `pip install boto3`.

The following metadata files are used in the creation of the new datasets:

*  `cifar10_keywords_unique_v{}.json` contains the TinyImage index, asociated keyword, and CIFAR-10 label for every image in CIFAR-10.
*  `keyword_counts_v{}.json` contains the image counts for each keyword.


# License

Unless noted otherwise in individual files, the code in this repository is released under the MIT license (see the `LICENSE` file).
The `LICENSE` file does *not* apply to the actual image and label data in the `datasets` folder.
The image data is part of the Tiny Images dataset and can be used the same way as the Tiny Images dataset.


# Citing CIFAR-10.1

To cite the CIFAR-10.1 dataset, please use the following references:
```
@article{recht2018cifar10.1,
  author = {Benjamin Recht and Rebecca Roelofs and Ludwig Schmidt and Vaishaal Shankar},
  title = {Do CIFAR-10 Classifiers Generalize to CIFAR-10?},
  year = {2018},
  note = {\url{https://arxiv.org/abs/1806.00451}},
}

@article{torralba2008tinyimages, 
  author = {Antonio Torralba and Rob Fergus and William T. Freeman}, 
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title = {80 Million Tiny Images: A Large Data Set for Nonparametric Object and Scene Recognition}, 
  year = {2008}, 
  volume = {30}, 
  number = {11}, 
  pages = {1958-1970}
}
```
