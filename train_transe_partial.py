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
    return parser.parse_args()

args = parse_args()
lr = args.lr
dim = args.dim
seed = args.seed

os.environ['CUDA_VISIBLE_DEVICES']='0'
con = config.Config()
con.set_use_gpu(True)
con.set_in_path("./benchmarks/FB15K_95/")
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(lr)
con.set_bern(0)
con.set_dimension(dim)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")
con.set_save_steps(100)
con.set_valid_steps(100)
con.set_early_stopping_patience(10)
con.set_seed(seed)
con.set_checkpoint_dir(f"./sweep_results_95/dim_{dim}_lr_{lr}_seed_{seed}")
con.set_result_dir(f"./sweep_results_95/dim_{dim}_lr_{lr}_seed_{seed}")
con.set_val_link(True)
con.set_test_link(True)
con.set_test_triple(False)
con.init()
con.set_train_model(TransE)
con.train()
