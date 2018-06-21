import io
import os
import json

import numpy as np
import PIL.Image

cifar10_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def np_to_png(a, fmt='png', scale=1):
    a = np.uint8(a)
    f = io.BytesIO()
    tmp_img = PIL.Image.fromarray(a)
    tmp_img = tmp_img.resize((scale * 32, scale * 32), PIL.Image.NEAREST)
    tmp_img.save(f, fmt)
    return f.getvalue()


def load_new_test_data(version='default'):
    data_path = os.path.join(os.path.dirname(__file__), '../datasets/')
    filename = 'cifar10.1'
    if version == 'default':
        pass
    elif version == 'v0':
        filename += '-v0'
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version))
    label_filename = filename + '-labels.npy'
    imagedata_filename = filename + '-data.npy'
    label_filepath = os.path.join(data_path, label_filename)
    imagedata_filepath = os.path.join(data_path, imagedata_filename)
    labels = np.load(label_filepath)
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version == 'default':
        assert labels.shape[0] == 2000
    elif version == 'v0':
        assert labels.shape[0] == 2021
    return imagedata, labels

def load_v4_distances_to_cifar10(
        filename='../other_data/tinyimage_cifar10_distances_full.json'):
    with open(filename, 'r') as f:
        tmp = json.load(f)
    assert len(tmp) == 372131
    result = {}
    for k, v in tmp.items():
        result[int(k)] = v
    return result

def load_v6_distances_to_cifar10(
    filename='../other_data/tinyimage_large_dst_images_v6.1.json'):
    with open(filename, 'r') as f:
        tmp = json.load(f)
    result = {}
    for _, v in tmp.items():
        for obj in v:
            result[int(obj["tinyimage_index"])] = obj["cifar10_nn_dst"]
    return result


def load_cifar10_by_keyword():
    '''Returns a dictionary maping each keyword in CIFAR10 to a list of
       TinyImage indices.'''
    with open('../other_data/cifar10_keywords.json') as f:
        cifar10_keywords = json.load(f)
    cifar10_by_keyword = {}
    for ii, keyword_entries in enumerate(cifar10_keywords):
        for entry in keyword_entries:
            cur_keyword = entry['nn_keyword']
            if not cur_keyword in cifar10_by_keyword:
                cifar10_by_keyword[cur_keyword] = []
            cifar10_by_keyword[cur_keyword].append(ii)
    return cifar10_by_keyword