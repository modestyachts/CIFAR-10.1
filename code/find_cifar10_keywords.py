import json
import sys

import falconn    # install via "pip install falconn"
import math
import numpy as np
from timeit import default_timer as mytimer
import tqdm

import cifar10
import tinyimages

offset = int(sys.argv[1])
num_images = int(sys.argv[2])

ti = tinyimages.TinyImages('/scratch/tinyimages')
cifar = cifar10.CIFAR10Data('/scratch/tinyimages/cifar10')

if num_images <= 0:
    num_images = ti.img_count

print('Reading {} images ...'.format(num_images))
start = mytimer()
imgs = ti.slice_to_numpy(offset, num_images)
imgs = imgs.reshape([num_images, 32 * 32 * 3])
stop = mytimer()
print('    done in {} seconds'.format(stop - start))

def normalize_data(data):
    n, dims = data.shape
    mean = np.sum(data, axis=0) / n
    data2 = data - mean.reshape((1, dims))
    norms = np.sqrt(np.sum(np.square(data2), axis=1))
    norms = np.maximum(norms, 0.1)
    data3 = data2 / norms.reshape((n, 1))
    data3 = data3.astype(np.float32)
    return data3, mean

print('Normalizing data for FALCONN ...')
start = mytimer()
imgs2, data_mean = normalize_data(imgs)
stop = mytimer()
print('    done in {} seconds'.format(stop - start))


print('Setting up FALCONN ...')
start = mytimer()
params_hp = falconn.LSHConstructionParameters()
params_hp.dimension = 32 * 32 * 3
params_hp.lsh_family = falconn.LSHFamily.Hyperplane
params_hp.distance_function = falconn.DistanceFunction.EuclideanSquared
params_hp.storage_hash_table = falconn.StorageHashTable.FlatHashTable
params_hp.k = num_images.bit_length() - 1
params_hp.l = 2
params_hp.num_setup_threads = 2
params_hp.seed = 833840234
hp_table = falconn.LSHIndex(params_hp)
hp_table.setup(imgs2)
qo = hp_table.construct_query_object()
qo.set_num_probes(2)
stop = mytimer()
print('    done in {} seconds'.format(stop - start))


n_cifar, = cifar.all_labels.shape
cifar_imgs_reshaped = cifar.all_images.reshape([n_cifar, 32 * 32 * 3]).astype(np.float32)
result = []
print('Going through the CIFAR10 dataset ...')
for ii in tqdm.tqdm(range(n_cifar)):
    query = cifar_imgs_reshaped[ii, :] - data_mean
    query /= np.linalg.norm(query)
    query_result = qo.find_nearest_neighbor(query.astype(np.float32))
    cur_res = {}
    cur_res['cifar10_label'] = cifar.label_names[cifar.all_labels[ii]]
    if query_result < 0:
        cur_res['nn_index'] = -1
        cur_res['nn_keyword'] = ''
        cur_res['nn_l2_dst'] = -1 
    else:
        cur_res['nn_index'] = query_result + offset
        cur_res['nn_keyword'] = ti.get_metadata(query_result + offset)[0]
        cur_res['nn_l2_dst'] = math.sqrt(np.sum(np.square(cifar_imgs_reshaped[ii, :] - imgs[query_result, :])))
    result.append(cur_res)

json.dump(result, open('cifar10_keywords_offset_{}_num_images_{}.json'.format(offset, num_images), 'w'), indent=2)
