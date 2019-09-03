import config
from  models import *
import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resultdir", type=str, required=True)
    parser.add_argument("--bitrate", type=int, required=True)
    parser.add_argument("--complete", action='store_true', help='Use variant of sampled dataset where all tuples occur.')
    return parser.parse_args()

args = parse_args()
lr = args.lr
dim = args.dim
seed = args.seed
bitrate = args.bitrate
tag = f'TransE_br_{bitrate}'
if args.complete:
	tag += '_complete'

os.environ['CUDA_VISIBLE_DEVICES']='0'
con = config.Config()
con.set_use_gpu(True)
con.set_in_path("./benchmarks/FB15K/")
con.set_dimension(dim)
con.set_seed(seed)
con.set_result_dir(f"./{args.resultdir}/dim_{dim}_lr_{lr}_seed_{seed}")
con.set_val_link(False)
con.set_test_link(True)
if '95' in args.resultdir:
	# con.set_test_triple(True,  f'sweep_results_95/dim_{dim}_lr_{lr}_seed_{seed}/{tag}_relThresh.pkl')
	con.set_test_triple(True)
else:
	if args.complete:
		con.set_test_triple(True, f'sweep_results_95_complete/dim_{dim}_lr_{lr}_seed_{seed}/{tag}_relThresh.pkl')
	else:
		con.set_test_triple(True, f'sweep_results_95/dim_{dim}_lr_{lr}_seed_{seed}/{tag}_relThresh.pkl')
# con.set_test_triple(True)
con.init()
con.set_test_model(TransE, path=f'{args.resultdir}/dim_{dim}_lr_{lr}_seed_{seed}/{tag}.ckpt')
con.test()
