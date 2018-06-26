#!/bin/bash

rm -rf ../other_data/keyword_subsets
mkdir ../other_data/keyword_subsets

python3 assemble_topk_keywords.py 100 > ../other_data/relevant_keywords.json
python3 build_keyword_subsets.py ../other_data/relevant_keywords.json 62
python3 unify_keyword_subsets.py
