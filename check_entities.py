import sys

with open(f'{sys.argv[1]}', 'r') as f:
	lines = f.readlines()

idx = set()
for line in lines[1:]:
	line = line.strip()
	e1, e2, _ = line.split(' ')
	idx.add(int(e1))
	idx.add(int(e2))

# with open('/dfs/scratch1/mleszczy/OpenKE_custom/benchmarks/FB15K/valid2id.txt', 'r') as f:
# 	lines = f.readlines()
# for line in lines[1:]:
# 	line = line.strip()
# 	e1, _, e2 = line.split(' ')
# 	idx.add(int(e1))
# 	idx.add(int(e2))


# with open('/dfs/scratch1/mleszczy/OpenKE_custom/benchmarks/FB15K/test2id.txt', 'r') as f:
# 	lines = f.readlines()
# for line in lines[1:]:
# 	line = line.strip()
# 	e1, _, e2 = line.split(' ')
# 	idx.add(int(e1))
# 	idx.add(int(e2))

print(len(set(idx)))
print(sys.argv[1])
print(set(range(14951)) - idx)