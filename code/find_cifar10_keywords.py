import json
import sys

import falconn    # install via "pip install falconn"
import math
import numpy as np
from timeit import default_timer as mytimer
import tqdm

import cifar10
import tinyimages

if len(sys.argv) != 3:
    print('Need two cmd line arguments: offset and num_images.')
    sys.exit(0)

offset = int(sys.argv[1])
num_images = int(sys.argv[2])

print('offset {},  num_images {}'.format(offset, num_images))

ti = tinyimages.TinyImages('/scratch/tinyimages')
cifar = cifar10.CIFAR10Data('/scratch/cifar10')

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
#params_hp.k = num_images.bit_length() + 5
params_hp.k = 30
params_hp.l = 1
params_hp.num_setup_threads = 1
params_hp.seed = 833840234
hp_table = falconn.LSHIndex(params_hp)
hp_table.setup(imgs2)
qo = hp_table.construct_query_object()
qo.set_num_probes(1)
stop = mytimer()
print('    done in {} seconds'.format(stop - start))


n_cifar, = cifar.all_labels.shape
cifar_imgs_reshaped = cifar.all_images.reshape([n_cifar, 32 * 32 * 3]).astype(np.float32)
result = []
sum_candidates = 0
print('Going through the CIFAR10 dataset ...')
for ii in tqdm.tqdm(range(n_cifar)):
    query = cifar_imgs_reshaped[ii, :] - data_mean
    query /= np.linalg.norm(query)
    query_result = qo.get_unique_candidates(query.astype(np.float32))
    sum_candidates += len(query_result)
    cur_results = []
    for jj in query_result:
        tmp_dst = math.sqrt(np.sum(np.square(cifar_imgs_reshaped[ii, :] - imgs[jj, :])))
        #print('CIFAR-10 index {}  TI index {}  dst {}'.format(ii, jj + offset, tmp_dst))
        if tmp_dst > 1.0:
            continue
        cur_res = {}
        cur_res['nn_index'] = jj + offset
        cur_res['nn_keyword'] = ti.get_metadata(jj + offset)[0]
        cur_res['nn_l2_dst'] = tmp_dst
        cur_results.append(cur_res)
    result.append(cur_results)

print('Average number of candidates per point: {}'.format(sum_candidates / n_cifar))

json.dump(result, open('../other_data/cifar10_keywords_offset_{}_num_images_{}.json'.format(offset, num_images), 'w'), indent=2)
