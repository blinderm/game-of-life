import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

  
threads_per_block = [2**i for i in range(4,9)]                                   
region_dim = [0, 5, 50]    
samples = 1000
fs = 16

directory = '../data'

sns.set_context('paper', rc={"font.size":fs,"axes.titlesize":fs,"axes.labelsize":fs})
sns.set_style('whitegrid', {'font.family': 'Times New Roman'})

dfs = [pd.read_csv('{0:}/{1:}TPB_{2:}RD.csv'.format(directory, 
    th, rg))[['threads_per_block', 'region_dim', 'time']].ix[1:samples-1] for \
    th in threads_per_block for rg in region_dim]

dfs += [pd.read_csv('{0:}/0TPB_0RD.csv'.format(directory))[['threads_per_block', 
    'region_dim', 'time']].ix[1:samples-1]]

dfs[-1]['region_dim'] = -1

df = pd.concat(dfs)

grouped = df.groupby(['region_dim', 'threads_per_block']).mean()
g = sns.FacetGrid(df, col="region_dim", size=4, aspect=1.5, sharex=False,
        despine=True, col_wrap=2)
(g.map(sns.violinplot, "threads_per_block", "time").despine())

g.axes[0].set_title("Serial")
g.axes[0].set_xticks([0])
g.axes[1].set_title("No optimizations")
g.axes[2].set_title(r"$(5 \times 5)$-cell regions")
g.axes[3].set_title(r"$(50 \times 50)$-cell regions")

for i in range(2,4):
    g.axes[i].set_xlabel("Threads per block")

for i in [0,2]:
    g.axes[i].set_ylabel("Time to compute update (ms)")

plt.savefig('../images/boxplot.eps', bbox_inches='tight')


results = [[stats.ttest_ind(d_i['time'], d_j['time']) for d_i in dfs] for d_j in dfs]

d = [] 

threshold = 0.01
 
for (i, row) in enumerate(results):
    for (j, col) in enumerate(row):
        if i < j and results[i][j].pvalue >= threshold:
            dct = {
                    'Region dimension 1': dfs[i]['region_dim'][1],
                    'Threads per block 1': dfs[i]['threads_per_block'][1],
                    'Region dimension 2': dfs[j]['region_dim'][1],
                    'Threads per block 2': dfs[j]['threads_per_block'][1],
                    'Statistic:' : results[i][j].statistic,
                    'p-value' : results[i][j].pvalue
                    }
            d.append(dct)

table_str = pd.DataFrame(d).to_latex()

with open('../report/statistics_table.tex', 'w+') as f:
    f.write(table_str)
