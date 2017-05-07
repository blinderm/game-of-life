import pandas as pd
from scipy import stats

threads_per_block = [2**6]
region_dim = [0, 10]
sizes = ['big', 'small']

data = [[[pd.read_csv('../data/{}TPB_{}RD_{}.csv'.format(i, j, size), sep=',')['time'] \
        for i in threads_per_block] for j in region_dim] for size in sizes]

for (n, size) in enumerate(data):
    for (j, regions) in enumerate(size):
        for (i, d) in enumerate(regions):
            d.name = '{} threads, {} regions {}'.format(threads_per_block[i], region_dim[j], sizes[n])

flattened = [item for row in data for items in row for item in items]
results = [[stats.ttest_ind(d_i, d_j) for d_i in flattened] for d_j in flattened]

threshold = 0.05

print("Insignificant results:")
for (i, row) in enumerate(results):
    for (j, col) in enumerate(row):
        if i < j and col[1] >= threshold:
            print(("({}) vs ({}).\n"
                "t-statistic: {:+1.3f},\n"
                "p-val: {:1.4e}.\n").format(flattened[i].name,
                    flattened[j].name,
                    float(col[0]),
                    col[1])) 

