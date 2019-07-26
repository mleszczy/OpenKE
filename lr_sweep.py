# lrs = [0.00001, 0.0001, 0.001, 0.01, 0.1]
# seeds = range(1234, 1237)
# dim = 50 

# for lr in lrs: 
# 	for seed in seeds: 
# 		print(f'python train_transe.py --lr {lr} --seed {seed} --dim {dim}')


lrs = [0.00001, 0.0001, 0.001, 0.01, 0.1]
seeds = range(1234, 1237)
dim = 50 

for lr in lrs: 
	for seed in seeds: 
		print(f'python train_transe_partial.py --lr {lr} --seed {seed} --dim {dim}')