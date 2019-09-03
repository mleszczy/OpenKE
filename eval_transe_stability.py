import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import spearmanr
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--complete", action='store_true', help='Use variant of sampled dataset where all tuples occur.')
    return parser.parse_args()

def read_results(file):
	with open(file, 'r') as f:
		results = f.readlines()

	ids = []
	dists = []
	ranks = []
	filter_ranks = []
	top10 = []
	top10_filter = []
	# skip over head
	for r in results[1:]:
		terms = r.split(',')

		id_ = int(terms[0])
		if id_ in [14946, 14885, 14731, 14703, 14896, 14751]:
			continue
		ids.append(int(terms[0]))
		ranks.append(float(terms[1]))
		filter_ranks.append(float(terms[2]))
		dists.append(float(terms[3]))
		top10.append(int(terms[4]))
		top10_filter.append(int(terms[5]))
	print(len(ranks), file)
	return ids, ranks, filter_ranks, dists, top10, top10_filter

def calc_top10_stability(l1, l2):
	preds1 = np.array(l1)
	preds2 = np.array(l2)
	return np.sum(preds1 == preds2) / float(len(preds1))

def calc_top10_stability_correct(l1, l2):
	preds1 = np.array(l1)
	preds2 = np.array(l2)
	count = 0.
	for i, j in zip(preds1, preds2):
		if i == j and i == 1:
			count += 1
	return (count / np.sum(preds1) + count / np.sum(preds2)) / 2

def calc_rank_stability(l1, l2):
	preds1 = np.array(l1)
	preds2 = np.array(l2)
	diff = np.abs(preds1-preds2)
	return np.mean(diff)
	# return np.linalg.norm(preds1 - preds2, ord=1)

def calc_dist_stability(l1, l2, dim):
	dist1 = np.array(l1)
	dist2 = np.array(l2)
	diff = np.linalg.norm(dist1 - dist2, ord=1)/len(dist1)
	return diff

def calc_rank_diff(l1, l2, diff=10):
	ranks1 = np.array(l1)
	ranks2 = np.array(l2)
	rank_diff = np.abs(ranks1 - ranks2)
	return len(np.where(rank_diff > diff)[0]) / float(len(l1))

def get_results(entity_type, dim, seed, bitrate, tag='', resultdir1='sweep_results_95'):
	file1 = f'{resultdir1}/dim_{dim}_lr_0.001_seed_{seed}/{tag}_test_{entity_type}_results.txt'
	file2 =f'sweep_results/dim_{dim}_lr_0.001_seed_{seed}/{tag}_test_{entity_type}_results.txt'
	ids_1, ranks_1, filter_ranks_1, dists_1, top10_1, top10_filter_1 = read_results(file1)
	ids_2, ranks_2, filter_ranks_2, dists_2, top10_2, top10_filter_2 = read_results(file2)
	assert ids_1[:] == ids_2[:], 'entity ids must match'
	top10_overlap = calc_top10_stability(top10_1, top10_2)
	filter_top10_overlap = calc_top10_stability(top10_filter_1, top10_filter_2)
	top10_overlap_correct = calc_top10_stability_correct(top10_1, top10_2)
	filter_top10_overlap_correct = calc_top10_stability_correct(top10_filter_1, top10_filter_2)

	rank_diff = calc_rank_stability(ranks_1, ranks_2)
	filter_rank_diff = calc_rank_stability(filter_ranks_1, filter_ranks_2)
	dist = calc_dist_stability(dists_1, dists_2, dim)
	rank_diff_thresh = calc_rank_diff(filter_ranks_1, filter_ranks_2)
	return rank_diff, filter_rank_diff, dist, top10_overlap, filter_top10_overlap, top10_overlap_correct, filter_top10_overlap_correct, filter_ranks_2, top10_filter_2, rank_diff_thresh

def get_triple_results(dim, seed, bitrate, tag='', resultdir1='sweep_results_95'):
	file1 = f'{resultdir1}/dim_{dim}_lr_0.001_seed_{seed}/{tag}_triple_preds.txt'
	file2 =f'sweep_results/dim_{dim}_lr_0.001_seed_{seed}/{tag}_triple_preds.txt'
	with open(file1, 'r') as f:
		preds1 = np.array([int(line.strip()) for line in f.readlines()])
	with open(file2, 'r') as f:
		preds2 = np.array([int(line.strip()) for line in f.readlines()])
	# print(preds1)
	# print(preds1, preds2, np.sum(preds1 == preds2), len(preds1))
	print(len(preds1))
	print(np.sum(preds1)/len(preds1), np.sum(preds2)/len(preds2))
	return 1 - np.sum(preds1 == preds2) / float(len(preds1))

# TODO(mleszczy): sanity check and make more efficient
def get_rank_correlations(dim, seed, bitrate, sample_prop=.10, tag='', resultdir1='sweep_results_95'):
	file1 = f'/dfs/scratch1/mleszczy/OpenKE_custom/{resultdir1}/dim_{dim}_lr_0.001_seed_{seed}/{tag}_dist.pkl'
	file2 = f'/dfs/scratch1/mleszczy/OpenKE_custom/sweep_results/dim_{dim}_lr_0.001_seed_{seed}/{tag}_dist.pkl'
	with open(file1, 'rb') as f:
		dist1 = pickle.load(f)
	with open(file2, 'rb') as f:
	    dist2 = pickle.load(f)
	print(type(dist1), type(dist2))
	print('Loaded distances.')
	print(len(dist1))
	total = 0.
	for i in range(len(dist1)):
		val, _ = spearmanr(dist1[i], dist2[i])
		total += val
	return total / len(dist1)

args = parse_args()

resultdir1 = 'sweep_results_95'
if args.complete:
	resultdir1 += '_complete'

# dims = [400]
dims = [10, 20, 50, 100, 200, 400]
seeds = [1234, 1235, 1236]
# seeds = [1234]
bitrates = [1, 2,4,8,16,32]
# bitrates = [32]
results = []
for dim in dims:
	for seed in seeds:
		for bitrate in bitrates:
			print(dim, seed)
			tag = f'TransE_br_{bitrate}'
			if args.complete:
				tag += '_complete'

			rank_diff_tail, filter_rank_diff_tail, dist_tail, top10_overlap_tail, filter_top10_overlap_tail, top10_overlap_correct_tail, filter_top10_overlap_correct_tail,rank_tail, top10_tail, rank_diff_tail = get_results('tail', dim=dim, seed=seed, bitrate=bitrate, tag=tag, resultdir1=resultdir1)
			rank_diff_head, filter_rank_diff_head, dist_head, top10_overlap_head, filter_top10_overlap_head, top10_overlap_correct_head,  filter_top10_overlap_correct_head, rank_head, top10_head, rank_diff_head = get_results('head', dim=dim, seed=seed, bitrate=bitrate, tag=tag, resultdir1=resultdir1)
			triple = get_triple_results(dim, seed, bitrate, tag=tag, resultdir1=resultdir1)
			# rank_corr = get_rank_correlations(dim, seed, bitrate, tag=tag)
			# print(rank_corr)
			row = {}
			row['space'] = bitrate * dim
			row['bitrate'] = bitrate
			row['seed'] = seed
			row['dim'] = dim
			row['dist'] = (dist_head + dist_tail)/2.
			row['avg_rank_diff'] = (rank_diff_head + rank_diff_tail)/2.
			row['top10_overlap'] = (top10_overlap_tail + top10_overlap_head)/2.
			row['top10_overlap_tail'] = top10_overlap_tail
			row['top10_overlap_head'] = top10_overlap_head
			row['top10_overlap_correct'] = (top10_overlap_correct_tail + top10_overlap_correct_head)/2.
			# row['top10_overlap_tail'] = top10_overlap_tail
			# row['top10_overlap_head'] = top10_overlap_head
			row['filter_top10_overlap_correct'] = (filter_top10_overlap_correct_tail + filter_top10_overlap_correct_head)/2.
			row['filter_top10_overlap'] = (filter_top10_overlap_tail + filter_top10_overlap_head)/2.
			row['avg_filter_rank_diff'] = (filter_rank_diff_tail + filter_rank_diff_head)/2.
			row['filter_rank_diff_thresh'] = (rank_diff_tail + rank_diff_head) / 2.
			row['filter_rank_diff_head'] = rank_diff_head
			row['filter_rank_diff_tail'] = rank_diff_tail
			row['top10'] = np.mean(top10_tail + top10_head)
			row['rank'] = np.mean(rank_tail + rank_head)
			row['triple'] = triple
			# row['rank_corr']  = rank_corr
			results.append(row)

df_results = pd.DataFrame(results)
df_sum = df_results.groupby(['dim', 'bitrate', 'space']).aggregate(['mean', 'std']).reset_index()
df_sum = df_sum.sort_values('space')
print(df_sum)
if args.complete:
	df_results_path = f"/dfs/scratch1/mleszczy/sigmod/analysis/transe/results_complete.pkl"
else:
	df_results_path = f"/dfs/scratch1/mleszczy/sigmod/analysis/transe/results.pkl"
df_results.to_pickle(df_results_path)