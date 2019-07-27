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
    return parser.parse_args()

args = parse_args()
lr = args.lr
dim = args.dim
seed = args.seed
bitrate = args.bitrate

os.environ['CUDA_VISIBLE_DEVICES']='0'
con = config.Config()
con.set_use_gpu(True)
con.set_in_path("./benchmarks/FB15K/")
con.set_dimension(dim)
con.set_seed(seed)
con.set_result_dir(f"./{args.resultdir}/dim_{dim}_lr_{lr}_seed_{seed}")
con.set_val_link(False)
con.set_test_link(True)
con.set_test_triple(False)
con.init()
con.set_test_model(TransE, path=f'{args.resultdir}/dim_{dim}_lr_{lr}_seed_{seed}/TransE_br_{bitrate}.ckpt')
con.test()
