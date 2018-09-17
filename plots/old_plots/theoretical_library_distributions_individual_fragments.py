from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys

import glob
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

# Import third party libraries
import cv2
from pprint import pprint
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
import seaborn as sns
from scipy.stats import truncnorm

# Import application/library specifics
sys.path.append('./utils')
from idtrackerai.utils.py_utils import  get_spaced_colors_util

if __name__ == '__main__':

    max = 3000
    min = 1
    means = [50, 100, 250, 500, 1000]
    std = 150
    RGB_tuples = get_spaced_colors_util(len(means), norm=True, black=False)

    sns.set_style('ticks')
    plt.ion()
    fig, ax_arr = plt.subplots(1,2)
    for i, mean in enumerate(means):
        X = truncnorm((min - mean) / std, (max - mean) / std, loc=mean, scale=std)
        x = np.linspace(min,max, 100)
        ax_arr[0].plot(x,X.pdf(x),'-',color = RGB_tuples[i])
        ax_arr[1].semilogx(x,X.pdf(x),'-',color = RGB_tuples[i],label = str(mean))

    ax_arr[0].set_xlabel('number of frames')
    ax_arr[0].set_ylabel('PDF')
    ax_arr[1].set_xlabel('number of frames')
    ax_arr[1].legend(title = 'mean number of \nframes in individual \nfragments', fancybox = True)

    plt.show()
