import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

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

for (i, row) in enumerate(results):
    for (j, col) in enumerate(row):
        print(
                ("({0:}) vs ({1:}).\n"
                    "t-statistic: {2:+1.3f},\n"
                    "p-val: {3:1.4e},\n"
                    "mu_{4:}: {5:1.2f}, mu_{6:}: {7:1.2f}.\n"
                    ).format(flattened[i].name,
                        flattened[j].name,
                        float(col[0]),
                        col[1],
                        i, flattened[i].mean(),
                        j, flattened[j].mean()
                        )
                    ) 

shape = (len(sizes), len(region_dim))

fig, axes = plt.subplots(*shape, )

for i in range(len(sizes)):
    for j in range(len(region_dim)):
        axes[i,j].hist(data[i][j][0], bins=50, normed=True, 
                label=data[i][j][0].name, align='mid')
        axes[i,j].legend()
plt.show()






