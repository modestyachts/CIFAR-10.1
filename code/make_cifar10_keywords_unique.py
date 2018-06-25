from collections import Counter
import json

import cifar10

cifar = cifar10.CIFAR10Data('/scratch/cifar10')

with open('../other_data/cifar10_keywords.json', 'r') as f:
    keywords = json.load(f)

assert len(keywords) == 60000

class_counters = {}
for ii in range(10):
    class_counters[ii] = Counter()
for ii, kws in enumerate(keywords):
    cur_keywords = []
    for kw in kws:
        cur_keywords.append(kw['nn_keyword'])
    class_counters[cifar.all_labels[ii]].update(cur_keywords)

new_keywords = []
for ii, kw_list in enumerate(keywords):
    cur_class = cifar.all_labels[ii]
    kws = []
    for tmp in kw_list:
        kws.append(tmp['nn_keyword'])
    assert len(kws) == len(set(kws))
    kws_with_frequencies = []
    for kw in kws:
        kws_with_frequencies.append((class_counters[cur_class][kw], kw))
    top_kw = sorted(kws_with_frequencies)[-1][1]
    if len(kws) > 1:
		    print('{}: {}'.format(ii, list(reversed(sorted(kws_with_frequencies)))))
    for tmp in kw_list:
        if tmp['nn_keyword'] == top_kw:
            new_keywords.append(tmp)

for ii, item in enumerate(new_keywords):
    assert item in keywords[ii]

with open('../other_data/cifar10_keywords_unique.json', 'w') as f:
    json.dump(new_keywords, f, indent=2)
