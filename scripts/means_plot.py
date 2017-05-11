import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


blues = sns.color_palette('GnBu_r')
reds = sns.color_palette("Reds")
   
threads_per_block = [2**i for i in range(4,9)]                                   
region_dim = [0, 5, 50]    
samples = 1000

fs = 8
sns.set_context('paper', rc={"font.size":fs,"axes.titlesize":fs,"axes.labelsize":fs})
sns.set_style('whitegrid', {'font.family': 'Times New Roman'})

directory = '../data'

dfs = [pd.read_csv('{0:}/{1:}TPB_{2:}RD.csv'.format(directory, 
    th, rg))[['threads_per_block', 'region_dim', 'time']].ix[1:samples-1] \
    for th in threads_per_block for rg in region_dim]

dfs += [pd.read_csv('{0:}/0TPB_0RD.csv'.format(directory))[['threads_per_block', 
    'region_dim', 'time']].ix[1:samples-1]]

dfs[-1]['region_dim'] = -1

df = pd.concat(dfs)

grouped = df.groupby(['region_dim', 'threads_per_block']).mean()

fig, ax = plt.subplots(1,1)

baseline = np.arange(0,257,64)
yvalues = float(grouped.ix[-1]['time']) * np.ones(baseline.size)

lw = 1
ms = 5

ax.plot(baseline, yvalues, linewidth=lw, color='black')
grouped.ix[0].plot(kind='line',  color=reds[-1],
        linewidth=lw, marker='s', markersize=ms, ax=ax)
grouped.ix[5].plot(kind='line',  color=blues[0], 
        linewidth=lw, marker='o', markersize=ms, ax=ax)
grouped.ix[50].plot(kind='line', color=blues[2], 
        linewidth=lw, marker='^', markersize=ms, ax=ax)

ax.set_xlim([-10, 280])
ax.set_xlabel('Threads per block')
ax.set_xticks(threads_per_block)

ax.set_ylabel('Mean update time (ms)')
ax.legend(['Serial', 'No optimizations', 
    'Optimization (small regions)', 'Optimization (large regions)'], 
        numpoints=1,
        loc='lower right')

sns.despine()

plt.savefig('../images/plot.eps', bbox_inches='tight')
