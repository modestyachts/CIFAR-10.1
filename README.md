# CIFAR-10.1
This repository contains the CIFAR-10.1 dataset, which is a new test set for [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

CIFAR-10.1 contains roughly 2,000 new test images that were sampled *after* multiple years of research on the original CIFAR-10 dataset.
The data collection for CIFAR-10.1 was designed to minimize distribution shift relative to the original dataset.
We describe the creation of CIFAR-10.1 in the paper ["Do CIFAR-10 Classifiers Generalize to CIFAR-10?"](https://arxiv.org/abs/1806.00451). 
The images in CIFAR-10.1 are a subset of the [TinyImages dataset](http://horatio.cs.nyu.edu/mit/tiny/data/index.html). 

# Using the Dataset

## Dataset Releases

There are currently two versions of the CIFAR-10.1 dataset:

- `v4` is the first version of our dataset on which we tested any classifier. As mentioned above, this makes the `v4` dataset independent of the classifiers we evaluate. The numbers reported in the main sections of our paper use this version of the dataset. It was built from the top 25 TinyImage keywords for each class, which led to a slight class imbalance. The largest difference is that ships make up only 8% of the test set instead of 10%. `v4` contains 2,021 images.

- `v6` is derived from a slightly improved keyword allocation that is exactly class balanced. This version of the dataset corresponds to the results in Appendix D of our paper. `v6` contains 2,000 images.

The overlap between `v4` and `v6` is more than 90% of the respective datasets.
Moreover, the classification accuracies are very close (see Appendix D of our paper).
For future experiments, we recommend the `v6` version of our dataset.

Missing version numbers correspond to internal releases during our quality control process (e.g., near-duplicate removal) or potential variants of our dataset we did not pursue further.

## Loading the Dataset

The `datasets` directory contains the dataset files in the [NumPy](http://www.numpy.org/) binary format:
- The `v4` files are `cifar10.1_v4_data.npy` and `cifar10.1_v4_labels.npy`.
- The `v6` files are `cifar10.1_v6_data.npy` and `cifar10.1_v6_labels.npy`.

The `notebooks` directory contains a short script `inspect_dataset_simple.ipynb` to browse the CIFAR-10.1 dataset.
The notebook uses a utility function to load the dataset from `utils.py` in the the `code` directory.

# Dataset Creation Pipeline

WARNING: This is currently work in progress, some parts may be incomplete.

This repository contains code to replicate the creation process of CIFAR-10.1. 
The dataset creation process has several stages outlined below.
We describe the process here at a high level.
If you have questions about any individual steps, please do not hesitate to contact Rebecca Roelofs (roelofs@cs.berkeley.edu) and Ludwig Schmidt (ludwigschmidt2@gmail.com).

## 1. Extracting Data from TinyImages

Since the TinyImages dataset is quite large (around 280 GB), we first extract the relevant data for further processing.
In particular, we require the following information:

* The TinyImages keyword for each image in CIFAR-10.
* All images in TinyImages belonging to these keywords.

We have automated these two steps via two scripts in the `code` directory:

* `find_all_cifar10_keywords.sh`
* `build_tinyimage_subset.sh`

We recommend running these scripts on a machine with at least 1 TB of RAM, e.g., an `x1.16xlarge` instance on AWS.
After downloading the TinyImage dataset, running the scripts will take about 30h.

## 2. Collecting Candidate Images

After downloading the relevant subset of TinyImages (keywords and image data) to a local machine, we can now assemble a set of candidate images for the new dataset.
We proceed in two steps:

### 2.1 Keyword counts for the new dataset

The notebook `generate_keyword_counts.ipynb` decides which keywords we want to include in the new dataset and determines the number of images we require for each of these keywords. 

### 2.2 Labeling new images

Once we know the number of new images we require for each keyword, we can collect corresponding images from TinyImages.
We used two notebooks for this process:

* The first labeler (or set of labelers) use `labeling_ui.ipynb` in order to collect a set of candidate images.
* The second labeler (or set of labelers) verify this selection via the `labeling_ui_subselect.ipynb` notebook.

## 3. Assembling a New Dataset

Given a pool of new candidate images, we can now sample a new dataset from this pool.
We have the following notebooks for this step:

* `sample_subselected_indices_v4.ipynb` samples the pool of labeled images and creates the new dataset for v4
* `sample_subselected_indices.ipynb` samples the pool of labeled images and creates the new dataset for v6 or v7

After sampling a new dataset, it is necessary to run some final checks via the `check_dataset_ui.ipynb` notebook.
In particular, this notebook checks for near-duplicates both within the new test set and in CIFAR-10 (a new test set would not be interesting if it contains many near-duplicates of the original test set).
In our experience, the process involves a few round-trips of sampling a new test set, checking for near-duplicates, and adding the near-duplicates to the blacklist.
Sometimes it is necessary to collect a few additional images for keywords with many near-duplicates (using the notebooks from Step 2 above).

In order to avoid re-computing L2 distances to CIFAR-10, the notebook `compute_distances_to_cifar10.ipynb` computes all top-10 nearest neighbors between our TinyImages subset and CIFAR-10.
Running this notebook takes only a few minutes when executed on 100 `m5.4xlarge` instances via [PyWren](http://pywren.io/).

## 4. Inspecting Model Predictions (Extra Step)
After assembling a final dataset, we ran a broad range of classifiers on the new test set via our CIFAR-10 model test bed.
The notebook `inspect_model_predictions.ipynb` explores the resulting predictions and displays a [Pandas](https://pandas.pydata.org/) dataframe including the original and new accuracy for each model. 


## Intermediate Data Files

In order to run only individual steps of the process outlined above, we provide all intermediate data files.
They are stored in the S3 bucket `cifar-10-1` and can be downloaded with the script `other_data/download.py`.
The script requires Boto 3, which can be installed via pip: `pip install boto3`.

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
