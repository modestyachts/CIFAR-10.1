#!/bin/bash

python3 find_cifar10_keywords.py 0 10000000
python3 find_cifar10_keywords.py 10000000 10000000
python3 find_cifar10_keywords.py 20000000 10000000
python3 find_cifar10_keywords.py 30000000 10000000
python3 find_cifar10_keywords.py 40000000 10000000
python3 find_cifar10_keywords.py 50000000 10000000
python3 find_cifar10_keywords.py 60000000 10000000
python3 find_cifar10_keywords.py 70000000 9302017

python3 unify_cifar10_keywords.py

python3 make_cifar10_keywords_unique.py
