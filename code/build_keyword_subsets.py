import json
from multiprocessing import Pool
import sys

import numba
import numpy as np
import tqdm

import cifar10
import tinyimages

keywords_filename = sys.argv[1]
num_processes = int(sys.argv[2])

keywords = []
with open(keywords_filename, 'r') as f:
    if keywords_filename.endswith('.json'):
        keywords = json.load(f)
    elif keywords_filename.endswith('.txt'):
        for line in f:
            parts = [p.strip() for p in line.split(',')]
            keywords.extend(parts)
    else:
        print('Unknown filename ending.')
        sys.exit(0)

unique_keywords = list(set(keywords))
print('Read {} keywords ({} unique):'.format(len(keywords), len(unique_keywords)))
print(unique_keywords)

if len(unique_keywords) > 1000:
    print('Warning: more than 1,000 keywords. This will lead to a large subset.')
    print('Remove this check in the code to proceed.')
    sys.exit(0)

@numba.jit(nopython=True)
def compute_dsts(imgs, other):
    return np.sqrt(np.sum(np.square(imgs - other), axis=1))

def process_keywords(keyword):
    print('Current keyword: {}'.format(keyword))
    ti = tinyimages.TinyImages('/scratch/tinyimages')
    cifar = cifar10.CIFAR10Data('/scratch/cifar10')
    imgs = cifar.all_images.astype(np.float32).reshape((60000, -1))
    assert imgs.shape == (60000, 32 * 32 * 3)
    idxs = ti.retrieve_by_term(keyword, 100000000)
    res = []
    range_obj = range(len(idxs))
    if num_processes == 1:
        range_obj = tqdm.tqdm(range_obj)
    cur_index_set = set()
    for ii in range_obj:
        assert idxs[ii] not in cur_index_set
        cur_index_set.add(idxs[ii])

        cur_img = ti.slice_to_numpy(idxs[ii]).astype(np.float32).reshape(-1)
        assert cur_img.shape == (32 * 32 * 3,)

        dsts = compute_dsts(imgs, cur_img)
        amin = int(np.argmin(dsts))

        cur_dst = float(dsts[amin])
        cur_res = {}
        cur_res['tinyimage_index'] = int(idxs[ii])
        cur_res['cifar10_nn_dst'] = cur_dst
        cur_res['cifar10_nn'] = amin
        res.append(cur_res)
        #print('ii = {} (ti index {})  min dst {}  (argmin {})'.format(ii, tmp[ii], dsts[amin], amin))
    
    final_res = {}
    final_res['tinyimage_keyword'] = keyword
    final_res['subset_indices'] = res
    with open('keyword_subsets/tinyimage_subset_{}.json'.format(keyword), 'w') as f:
        json.dump(final_res, f, indent=2)


if num_processes > 1:
    Pool(num_processes).map(process_keywords, unique_keywords)
else:
    for k in unique_keywords:
        process_keywords(k)
