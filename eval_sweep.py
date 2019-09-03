import glob
import os
import argparse
import sys
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--resultdir",
        type=str,
        default='/dfs/scratch1/mleszczy/OpenKE_custom/sweep_results')
    parser.add_argument(
        "--nseeds",
        type=int,
        default=3,
        help='Number of seeds expected')
    parser.add_argument(
        "--dim",
        type=int,
        default=50,
        help='Dimension')
    return parser.parse_args()

args = parse_args()
resultdir = args.resultdir

os.makedirs(resultdir, exist_ok=True)

results = {}
embs = glob.glob(f'{resultdir}/*/training.log')
print(embs)
for emb in embs:
    terms = os.path.basename(os.path.dirname(emb)).split("_")
    seed = terms[terms.index("seed") + 1].split("/")[0]
    dim = int(terms[terms.index("dim") + 1])
    lr = terms[terms.index("lr") + 1]

    if dim == args.dim:
        ff = open(emb, 'r')
        dat = [_.strip() for _ in ff]
        for line in dat:
            if "MR rank on valid set is" in line:
                mean_rank = float(line.split(' ')[-1])
                print(emb, mean_rank)
                if lr not in results:
                    results[lr] = [mean_rank]
                else:
                    results[lr] += [mean_rank]
                continue
avg = []
for lr in results:
    # skip over ones that don't have three seeds
    print(results[lr], lr)
    assert len(results[lr]) == args.nseeds, f"Seed number incorrect"
    avg.append((lr, np.mean(results[lr])))

for lr in results:
    print(f'{lr} ' +  ' '.join([str(i) for i in results[lr]]) + f' {np.mean(results[lr])}')

print(sorted(
            avg,
            key=lambda x: x[1] if not math.isnan(x[1]) else 0,
            reverse=False))
