import os
lr = 0.001
seeds = range(1234, 1237)
dims = [10, 20, 50, 100, 200, 400]

# for seed in seeds:
# 	for dim in dims:
# 		dirname = f'sweep_results/dim_{dim}_lr_{lr}_seed_{seed}'
# 		os.makedirs(dirname, exist_ok=True)
# 		print(f'python train_transe.py --lr {lr} --seed {seed} --dim {dim}  &> {dirname}/training.log')

# for seed in seeds:
# 	for dim in dims:
# 		dirname = f'sweep_results_95/dim_{dim}_lr_{lr}_seed_{seed}'
# 		os.makedirs(dirname, exist_ok=True)
# 		print(f'python train_transe_partial.py --lr {lr} --seed {seed} --dim {dim} &> {dirname}/training.log')


for seed in seeds:
	for dim in dims:
		dirname = f'sweep_results_95_complete/dim_{dim}_lr_{lr}_seed_{seed}'
		os.makedirs(dirname, exist_ok=True)
		print(f'python train_transe_partial.py --lr {lr} --seed {seed} --dim {dim} --resultdir sweep_results_95_complete --datadir FB15K_95_complete &> {dirname}/training.log')


