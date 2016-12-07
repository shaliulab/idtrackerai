import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')

sys.path.append('../utils')
sys.path.append('../CNN')

from py_utils import *
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import cPickle as pickle

df = pd.read_hdf('/home/lab/Desktop/TF_models/IdTracker/Medaka/20161129174648_/statistics.hdf5' )
df = df.to_dict()[0]
portraits = pd.read_hdf('/home/lab/Desktop/TF_models/IdTracker/Medaka/20161202142742_/portraits.hdf5')
fragmentsDict = pickle.load(open('/home/lab/Desktop/TF_models/IdTracker/Medaka/20161202142742_/fragments.pkl','rb'))
fragments = fragmentsDict['fragments']

permutations = np.asarray(portraits.loc[:,'permutations'].tolist())
P2 = df['P2FragAllVideo'] # (numFrames,maxNumBlobs,numAnimals)
identities = df['fragmentIds']
numAnimals = 20
maxNumBlobs = len(permutations[0])


def orderVideo(matrixToOrder,permutations,maxNumBlobs):
    matrixOrdered = np.zeros_like(matrixToOrder)

    for frame in range(len(permutations)):
        for i in range(maxNumBlobs):
            index = list(np.where(permutations[frame]==i)[0])
            # print index
            if len(index) == 1:
                matrixOrdered[frame,i] = matrixToOrder[frame,index]
            else:
                matrixOrdered[frame,i] = -1

    return matrixOrdered

# permOrdered =  orderVideo(permutations,permutations,maxNumBlobs)
#
# P2Ordered =  orderVideo(P2,permutations,maxNumBlobs)
#
#
# # for frame in range(len(permutations)):
# #     for i in range(maxNumBlobs):
# #         index = list(np.where(permutations[frame]==i)[0])
# #         # print index
# #         if len(index) == 1:
# #             permOrdered[frame,i] = permutations[frame,index]
# #             # P2Ordered[frame,i,:] = P2[frame,index,:]
# #             P2Ordered[frame,i] = P2[frame,index]
# #             idsOrdered[frame,i] = identities[frame,index]
# #         else:
# #             permOrdered[frame,i] = -1
# #             # P2Ordered[frame,i,:] = -1.
# #             P2Ordered[frame,i] = -1.
# #             idsOrdered[frame,i] = -1
#
#
# P2good = np.max(P2Ordered,axis=2).T


''' fragmentation '''
permOrdered =  orderVideo(permutations,permutations,maxNumBlobs)
permOrdered = permOrdered.T.astype('float32')
plt.ion()
fig, ax = plt.subplots(figsize=(25, 5))
permOrdered[permOrdered >= 0] = .5
im = plt.imshow(permOrdered,cmap=plt.cm.gray,interpolation='none',vmin=0.,vmax=1.)
im.cmap.set_under('r')
# im.set_clim(0, 1.)
# cb = plt.colorbar(im)

# for i in range(len(fragments)):
for i in range(10):
    ax.add_patch(
        patches.Rectangle(
            (fragments[i,0], -0.5),   # (x,y)
            fragments[i,1]-fragments[i,0],  # width
            20,          # height
            fill=True,
            edgecolor=None,
            facecolor='b',
            alpha = 0.5
        )
    )

plt.axis('tight')
plt.xlabel('Frame number')
plt.ylabel('Blob index')
plt.gca().set_yticks(range(0,maxNumBlobs,4))
plt.gca().set_yticklabels(range(1,maxNumBlobs+1,4))
plt.gca().invert_yaxis()
plt.tight_layout()

''' P2 '''
P2Ordered =  orderVideo(P2,permutations,maxNumBlobs)
P2good = np.max(P2Ordered,axis=2).T
plt.ion()
fig, ax = plt.subplots(figsize=(25, 5))
im2 = plt.imshow(P2good,cmap=plt.cm.gray,interpolation='none')
im2.cmap.set_under('r')
im2.set_clim(0, 1)
cb = plt.colorbar(im2)
# fig.colorbar(im, ax=ax)
plt.axis('tight')
plt.xlabel('Frame number')
plt.ylabel('Blob index')
plt.gca().set_yticks(range(0,maxNumBlobs,4))
plt.gca().set_yticklabels(range(1,maxNumBlobs+1,4))
plt.gca().invert_yaxis()
plt.tight_layout()

''' Identities '''
idsOrdered =  orderVideo(identities,permutations,maxNumBlobs)
idsOrdered = idsOrdered.T
# plt.ion()
# fig, ax = plt.subplots(figsize=(25, 5))
# # cmap = plt.cm.get_cmap("jet")
# # cmap.set_under("black")
# im3 = plt.imshow(idsOrdered,cmap=plt.cm.jet,interpolation='none')
# im3.cmap.set_under('k')
# im3.set_clim(0, 19)
# cb = plt.colorbar(im3)
# # fig.colorbar(im, ax=ax)
# plt.axis('tight')
# plt.xlabel('Frame number')
# plt.ylabel('Individual index')
# plt.tight_layout()


plt.show()
