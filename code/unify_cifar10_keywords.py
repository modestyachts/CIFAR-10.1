from collections import Counter
import json

filenames = ['cifar10_keywords_offset_0_num_images_10000000.json',
             'cifar10_keywords_offset_10000000_num_images_10000000.json',
             'cifar10_keywords_offset_20000000_num_images_10000000.json',
             'cifar10_keywords_offset_30000000_num_images_10000000.json',
             'cifar10_keywords_offset_40000000_num_images_10000000.json',
             'cifar10_keywords_offset_50000000_num_images_10000000.json',
             'cifar10_keywords_offset_60000000_num_images_10000000.json',
             'cifar10_keywords_offset_70000000_num_images_9302017.json']
num_images = 60000
num_files = len(filenames)

results = []
for fn in filenames:
    with open('../other_data/' + fn) as f:
        cur_result = json.load(f)
        assert len(cur_result) == num_images
        results.append(cur_result)

overall_result = []
for ii in range(num_images):
    cur_res = []
    for jj in range(num_files):
        cur_res.extend(results[jj][ii])
    overall_result.append(cur_res)

print('CIFAR-10 indices without exact match in Tiny Images:')
found_any = False
for ii in range(num_images):
    if len(overall_result[ii]) == 0:
        found_any = True
        print(ii)
if not found_any:
    print('  (all CIFAR-10 images have a match)')

print('CIFAR-10 indices with a non-zero distance match:')
found_any = False
for ii in range(num_images):
    for jj in range(len(overall_result[ii])):
        if overall_result[ii][jj]['nn_l2_dst'] != 0.0:
            print('  CIFAR-10 index: {}    match: {}'.format(ii, jj))
            found_any = True
if not found_any:
    print('  (all matches are distance 0)')

print('Count histogram')
counter = Counter([len(x) for x in overall_result])
for a, b in counter.most_common():
    print('    {} images with {} matches'.format(b, a))

json.dump(overall_result, open('../other_data/cifar10_keywords.json', 'w'), indent=2)
