from collections import Counter
import json
import sys

import cifar10

topk = int(sys.argv[1])

cifar = cifar10.CIFAR10Data('/scratch/cifar10')

num_images = 60000
num_classes = 10

keywords = json.load(open('../other_data/cifar10_keywords_unique.json', 'r'))
assert len(keywords) == num_images

class_counters = {}
for cl in range(num_classes):
    class_counters[cl] = Counter()
for ii, kw in enumerate(keywords):
    class_counters[cifar.all_labels[ii]].update([kw['nn_keyword']])

all_keywords = []
for _, counter in class_counters.items():
    cur_keywords = []
    for keyword, _ in counter.most_common(topk):
        cur_keywords.append(keyword)
    all_keywords.extend(cur_keywords)

print(json.dumps(all_keywords, indent=2))
