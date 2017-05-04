import numpy as np

probs_dead = 0.8
fname = "random_{}.board".format(probs_dead)

rows = 60
cols = 80

i = 0
with open(fname, 'w+') as f:
    while i < rows * cols:
        if probs_dead  < np.random.rand():
            f.write('X')
        else:
            f.write(' ')
        i += 1
        if (i % cols == 0):
            f.write('\n')
