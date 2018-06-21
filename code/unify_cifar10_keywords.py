import json
import numpy as np
import tqdm

import cifar10
import tinyimages

filenames = ['cifar10_keywords_offset_0_num_images_10000000.json',
             'cifar10_keywords_offset_10000000_num_images_10000000.json',
             'cifar10_keywords_offset_20000000_num_images_20000000.json',
             'cifar10_keywords_offset_40000000_num_images_10000000.json',
             'cifar10_keywords_offset_50000000_num_images_20000000.json',
             'cifar10_keywords_offset_70000000_num_images_9302017.json']
num_images = 60000
num_files = len(filenames)

ti = tinyimages.TinyImages('/scratch/tinyimages')

results = []
for fn in filenames:
    with open('../metadata/' + fn) as f:
        cur_result = json.load(f)
        assert len(cur_result) == num_images
        results.append(cur_result)

overall_result = []
for ii in range(num_images):
    cur_res = []
    for jj in range(num_files):
        cur_dst = results[jj][ii]['nn_l2_dst']
        if cur_dst < 1.0 and cur_dst > -0.1:
            cur_res.append(results[jj][ii])
    overall_result.append(cur_res)

print('CIFAR-10 indices without exact match in Tiny Images:')
for ii in range(num_images):
    if len(overall_result[ii]) == 0:
        print(ii)

json.dump(overall_result, open('../metadata/cifar10_keywords.json', 'w'), indent=2)
