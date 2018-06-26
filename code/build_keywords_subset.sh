#!/bin/bash

python3 assemble_topk_keywords.py 100 > ../other_data/relevant_keywords.json
python3 build_keyword_subsets.py ../other_data/relevant_keywords.json 62
