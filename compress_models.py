import config
from  models import *
import json
import os
import argparse
from smallfry.compress import compress_uniform, compress_kmeans
import torch
from scipy.linalg import orthogonal_procrustes
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--br", type=int, required=True)
    return parser.parse_args()

def load_model(resultdir):
	con = config.Config()
	con.set_in_path("./benchmarks/FB15K/")
	con.set_dimension(args.dim)
	con.set_seed(seed)
	con.set_result_dir(f"./{resultdir}/dim_{dim}_lr_{lr}_seed_{seed}")
	con.init()
	con.set_test_model(TransE)
	return con

def get_embs(con):
	ent_embs1 = con.testModel.ent_embeddings.weight.cpu().data.numpy()
	rel_embs1 = con.testModel.rel_embeddings.weight.cpu().data.numpy()
	return ent_embs1, rel_embs1

def align(emb, emb_target):
	assert emb.shape == emb_target.shape, 'Shapes must match between the two embeddings'
	R, _ = orthogonal_procrustes(emb, emb_target)
	emb = np.dot(emb, R)
	return emb

args = parse_args()
lr = 0.001
dim = args.dim
seed = args.seed
br = args.br

# load model, necessary config params to set and get embeddings
con1 = load_model('sweep_results_95')
ent_embs1, rel_embs1 = get_embs(con1)

# compress entity and relation embeddings
compressed_ent_embs1, _, _ = compress_uniform(X=ent_embs1, bit_rate=br,
            adaptive_range=True)
compressed_rel_embs1, _, _ = compress_uniform(X=rel_embs1, bit_rate=br,
            adaptive_range=True)
# copy both compressed embeddings back to model
con1.testModel.ent_embeddings.weight.data.copy_(torch.from_numpy(compressed_ent_embs1))
con1.testModel.rel_embeddings.weight.data.copy_(torch.from_numpy(compressed_rel_embs1))
# save new ckpt with compressed embs
con1.save_compressed_checkpoint(br)

# load second model and get embs
con2 = load_model('sweep_results')
ent_embs2, rel_embs2 = get_embs(con2)
# align embeddings to original embeddings using procrustes
# vocab is fixed by dataset so we don't have to worry about vocab matching
# stacked1 = np.concatenate((ent_embs1, rel_embs1))
# stacked2 = np.concatenate((ent_embs2, rel_embs2))

# stacked2 = align(stacked2, stacked1)
# ent_embs2 = stacked2[:len(ent_embs2)]
# rel_embs2 = stacked2[len(ent_embs2):]

assert ent_embs2.shape == ent_embs1.shape
assert rel_embs2.shape == rel_embs1.shape

# ent_embs2 = align(ent_embs2, ent_embs1)
# rel_embs2 = align(rel_embs2, rel_embs1)

# use the same range to compress
compressed_ent_embs2, _, _ = compress_uniform(X=ent_embs2, bit_rate=br, adaptive_range=True, X_0=ent_embs1)
compressed_rel_embs2, _, _ = compress_uniform(X=rel_embs2, bit_rate=br, adaptive_range=True, X_0=rel_embs1)

# copy both compressed embeddings back to model
con2.testModel.ent_embeddings.weight.data.copy_(torch.from_numpy(compressed_ent_embs2))
con2.testModel.rel_embeddings.weight.data.copy_(torch.from_numpy(compressed_rel_embs2))
# save new ckpt with compressed embs
con2.save_compressed_checkpoint(br)