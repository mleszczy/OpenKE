import os
lrs = [0.00001, 0.0001, 0.01, 0.1]
seeds = range(1234, 1237)
dim = 50

# for lr in lrs:
# 	for seed in seeds:
# 		dirname = f'sweep_results/dim_{dim}_lr_{lr}_seed_{seed}'
# 		os.makedirs(dirname, exist_ok=True)
# 		print(f'python train_transe.py --lr {lr} --seed {seed} --dim {dim}  &> {dirname}/training.log')

# for lr in lrs:
# 	for seed in seeds:
# 		dirname = f'sweep_results_95/dim_{dim}_lr_{lr}_seed_{seed}'
# 		os.makedirs(dirname, exist_ok=True)
# 		print(f'python train_transe_partial.py --lr {lr} --seed {seed} --dim {dim} &> {dirname}/training.log')


for lr in lrs:
	for seed in seeds:
		dirname = f'sweep_results_95_complete/dim_{dim}_lr_{lr}_seed_{seed}'
		os.makedirs(dirname, exist_ok=True)
		print(f'python train_transe_partial.py --lr {lr} --seed {seed} --dim {dim} --resultdir sweep_results_95_complete --datadir FB15K_95_complete &> {dirname}/training.log')

