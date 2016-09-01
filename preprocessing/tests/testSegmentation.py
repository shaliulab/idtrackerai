import numpy as np
import itertools

def isplit(iterable,splitters):
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]

a = np.array([1,5,5,5,5,1,1,5,5,5,5,1,5,1,1,5,5,5,1,1,5,5,5,1,1])
frames = np.arange(len(a))
frames[a!=5] = -1
fragments = isplit(frames,[-1])
print fragments



fragments2 = ssplit2(frames,[-1])
print fragments2
