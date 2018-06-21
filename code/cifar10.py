# Derived from https://github.com/jaberg/skdata/blob/master/skdata/cifar10/dataset.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np

class CIFAR10Data(object):
  def __init__(self, path):
    train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
    eval_filename = 'test_batch'
    metadata_filename = 'batches.meta'

    train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
    train_labels = np.zeros(50000, dtype='int32')
    for ii, fname in enumerate(train_filenames):
      cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
      train_images[ii * 10000 : (ii+1) * 10000, ...] = cur_images
      train_labels[ii * 10000 : (ii+1) * 10000, ...] = cur_labels
    eval_images, eval_labels = self._load_datafile(
      os.path.join(path, eval_filename))

    with open(os.path.join(path, metadata_filename), 'rb') as fo:
      data_dict = pickle.load(fo, encoding='bytes')
      self.label_names = data_dict[b'label_names']
    for ii in range(len(self.label_names)):
      self.label_names[ii] = self.label_names[ii].decode('utf-8')

    self.train_images = train_images
    self.train_labels = train_labels
    self.eval_images = eval_images
    self.eval_labels = eval_labels
    self.all_images = np.vstack([self.train_images, self.eval_images])
    self.all_labels = np.concatenate([self.train_labels, self.eval_labels])
    assert self.all_images.shape == (60000, 32, 32, 3)
    assert self.all_labels.shape == (60000,)
  
  def compute_l2_distances(self, x):
    return np.sqrt(np.sum(np.square(self.all_images.astype(np.float64) - x.astype(np.float64)), axis=(1,2,3)))
 
  @staticmethod
  def _load_datafile(filename):
    with open(filename, 'rb') as fo:
      data_dict = pickle.load(fo, encoding='bytes')
      assert data_dict[b'data'].dtype == np.uint8
      image_data = data_dict[b'data']
      image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
      return image_data, np.array(data_dict[b'labels'])
