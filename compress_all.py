import os
seeds = range(1234, 1237)
dims = [10,20,50,100,200,400]
lr = 0.001
bitrates = [1,2,4,8,16,32]
for dim in dims:
	for seed in seeds:
		for br in bitrates:
			# print(f'python compress_models.py --seed {seed} --dim {dim} --bitrate {br} --resultdir sweep_results')
			print(f'python compress_models.py --seed {seed} --dim {dim} --br {br}')