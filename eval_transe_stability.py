import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp

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
		ids.append(int(terms[0]))
		ranks.append(float(terms[1]))
		filter_ranks.append(float(terms[2]))
		dists.append(float(terms[3]))
		top10.append(int(terms[4]))
		top10_filter.append(int(terms[5]))
	return ids, ranks, filter_ranks, dists, top10, top10_filter

def calc_top10_stability(l1, l2):
	preds1 = np.array(l1)
	preds2 = np.array(l2)
	count = 0.
	for i, j in zip(preds1, preds2):
		if i == j and i == 1:
			count += 1
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
	diff = np.linalg.norm(dist1 - dist2, ord=1)
	return diff

def get_results(entity_type, dim, seed, bitrate):
	file1 = f'sweep_results_95/dim_{dim}_lr_0.001_seed_{seed}/TransE_br_{bitrate}_test_{entity_type}_results.txt'
	file2 =f'sweep_results/dim_{dim}_lr_0.001_seed_{seed}/TransE_br_{bitrate}_test_{entity_type}_results.txt'
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
	return rank_diff, filter_rank_diff, dist, top10_overlap, filter_top10_overlap, top10_overlap_correct, filter_top10_overlap_correct, filter_ranks_2, top10_filter_2

# dims = [100]
dims = [10, 20, 50, 100, 200, 400]
seeds = [1234, 1235, 1236]
# seeds = [1234]
bitrates = [1,2,4,8,16,32]
# bitrates = [32]
results = []
for dim in dims:
	for seed in seeds:
		for bitrate in bitrates:
			print(dim, seed)

			rank_diff_tail, filter_rank_diff_tail, dist_tail, top10_overlap_tail, filter_top10_overlap_tail, top10_overlap_correct_tail, filter_top10_overlap_correct_tail,rank_tail, top10_tail = get_results('tail', dim=dim, seed=seed, bitrate=bitrate)
			rank_diff_head, filter_rank_diff_head, dist_head, top10_overlap_head, filter_top10_overlap_head, top10_overlap_correct_head,  filter_top10_overlap_correct_head, rank_head, top10_head = get_results('head', dim=dim, seed=seed, bitrate=bitrate)
			row = {}
			row['space'] = bitrate * dim
			row['bitrate'] = bitrate
			row['seed'] = seed
			row['dim'] = dim
			row['dist'] = (dist_head + dist_tail)/2.
			row['avg_rank_diff'] = (rank_diff_head + rank_diff_tail)/2.
			row['top10_overlap'] = (top10_overlap_tail + top10_overlap_head)/2.
			# row['top10_overlap_tail'] = top10_overlap_tail
			# row['top10_overlap_head'] = top10_overlap_head
			row['top10_overlap_correct'] = (top10_overlap_correct_tail + top10_overlap_correct_head)/2.
			# row['top10_overlap_tail'] = top10_overlap_tail
			# row['top10_overlap_head'] = top10_overlap_head
			row['filter_top10_overlap_correct'] = (filter_top10_overlap_correct_tail + filter_top10_overlap_correct_head)/2.
			row['filter_top10_overlap'] = (filter_top10_overlap_tail + filter_top10_overlap_head)/2.
			row['avg_filter_rank_diff'] = (filter_rank_diff_tail + filter_rank_diff_head)/2.
			row['top10'] = np.mean(top10_tail + top10_head)
			row['rank'] = np.mean(rank_tail + rank_head)
			results.append(row)

df_results = pd.DataFrame(results)
df_sum = df_results.groupby(['dim', 'bitrate', 'space']).aggregate(['mean', 'std']).reset_index()
df_sum = df_sum.sort_values('space')

df_results_path = f"/dfs/scratch1/mleszczy/sigmod/analysis/transe/results.pkl"
df_results.to_pickle(df_results_path)