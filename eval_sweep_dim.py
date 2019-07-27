import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()

args = parse_args()

lr = 0.001
seeds = range(1234, 1237)
dims = [10, 20, 50, 100, 200, 400]
bitrates = [1,2,4,8,16,32]

# override if needed
if args.dim:
	dims = [args.dim]

if args.seed:
	seeds = [args.seed]

for seed in seeds:
	for dim in dims:
		for br in bitrates:
			dirname = f'sweep_results/dim_{dim}_lr_{lr}_seed_{seed}'
			print(f'python eval_transe.py --lr {lr} --seed {seed} --dim {dim} --resultdir sweep_results --bitrate {br} &> {dirname}/TransE_br_{br}_test.log')

			dirname = f'sweep_results_95/dim_{dim}_lr_{lr}_seed_{seed}'
			print(f'python eval_transe.py --lr {lr} --seed {seed} --dim {dim} --resultdir sweep_results_95 --bitrate {br} &> {dirname}/TransE_br_{br}_test.log')