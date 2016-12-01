import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import numpy as np
import pandas as pd
import cPickle as pkl

df = pd.read_hdf('/home/lab/Desktop/TF_models/IdTracker/Medaka/20161129174648_/statistics.hdf5' )
df = df.to_dict()[0]
P2 = df['P2FragAllVideo']
P2good = np.max(P2,axis=1).T


fig, ax = plt.subplots(1, 1)
cmap = plt.cm.get_cmap("jet")
cmap.set_under("black")
im = plt.imshow(P2good,cmap=cmap,interpolation='none')
fig.colorbar(im, ax=ax)
plt.axis('tight')
plt.show()
