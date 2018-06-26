import json
import os
import pickle
import random

import numpy as np

import tinyimages

ti = tinyimages.TinyImages('/scratch/tinyimages')

new_dict = {}
data_dict = {}

with os.scandir('check_keyword_output/') as it:
    for entry in it:
        if entry.name.endswith('json') and entry.is_file():
            with open(entry.path, 'r') as f:
                new_imgs = json.load(f)
                keyword_name = new_imgs['tinyimage_keyword']
                assert keyword_name not in new_dict
                print('Adding keyword "{}" ...'.format(keyword_name))
                new_dict[keyword_name] = new_imgs['large_dst_images']
                for rec in new_dict[keyword_name]:
                    assert set(rec.keys()) == set(['tinyimage_index', 'cifar10_nn_dst', 'cifar10_nn'])
                    assert rec['cifar10_nn_dst'] >= 0.0
                    assert rec['cifar10_nn_dst'] <= 10000.0
                    assert rec['cifar10_nn'] >= 0
                    assert rec['cifar10_nn'] < 60000
                    ti_index = rec['tinyimage_index']
                    data_dict[ti_index] = ti.slice_to_numpy(ti_index)
                    assert data_dict[ti_index].shape == (32, 32, 3)
                    assert data_dict[ti_index].dtype == np.uint8

print('Loaded a total of {} keywords'.format(len(new_dict)))

all_ti_indices = []
for keyword_name in new_dict:
    for rec in new_dict[keyword_name]:
        all_ti_indices.append(rec['tinyimage_index'])
assert set(all_ti_indices) == set(data_dict.keys())

with open('tinyimage_large_dst_images.json', 'w') as f:
    json.dump(new_dict, f, indent=2)

with open('tinyimage_large_dst_image_data.pickle', 'wb') as f:
    pickle.dump(data_dict, f)
