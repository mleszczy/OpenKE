import argparse
import numpy as np
import random
import sys

seed = int(sys.argv[1])
np.random.seed(seed)
random.seed(seed)

# load all train tuples
file = '/dfs/scratch1/mleszczy/OpenKE_custom/benchmarks/FB15K/train2id.txt'
out = f'/dfs/scratch1/mleszczy/OpenKE_custom/benchmarks/FB15K/train2id_95_seed_{seed}.txt'
percentage = 0.95

with open(file, 'r') as f:
	# first line is number of tuples
	all_train_tuples = f.readlines()[1:]
print(f'Loaded {len(all_train_tuples)} training tuples')

ntuples = int(len(all_train_tuples) * percentage)
print(ntuples)

# shuffle indices
random.shuffle(all_train_tuples)

with open(out, 'w') as f:
	f.write(f'{ntuples}\n')
	for i in range(ntuples):
		f.write(all_train_tuples[i])