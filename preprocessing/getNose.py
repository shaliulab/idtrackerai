import cv2
import sys
sys.path.append('../utils')
from py_utils import *
from video_utils import *
import time
import numpy as np
from matplotlib import pyplot as plt
from Tkinter import *
import tkMessageBox
import argparse
import os
import glob
import pandas as pd
import re
from joblib import Parallel, delayed
import multiprocessing
import cPickle as pickle


paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_10.avi')
df, numSegment = loadFile(paths[0], 'segmentation', time=0)

contour = df.loc[100,'contours'][2]
contour = np.squeeze(contour)
X = contour[:,0]
Y = contour[:,1]

approxC = cv2.approxPolyDP(contour,.5,True)
approxC = cv2.arcLength(contour,.5,True)
print approxC
# approxC = np.squeeze(approxC)
# X = approxC[:,0]
# Y = approxC[:,1]
#
# plt.ion()
# plt.plot(X,Y,'r')
# plt.show()
