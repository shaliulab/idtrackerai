import cv2
import sys
sys.path.append('../utils')
from py_utils import *
import time
import numpy as np
from matplotlib import pyplot as plt
from Tkinter import *
import tkMessageBox
import argparse
import os
import glob
import pandas as pd
import time
import re
from joblib import Parallel, delayed
import multiprocessing
import itertools
import cPickle as pickle

numSegment = 0
paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi')
frameIndices = pd.read_pickle('../Conflict8/conflict3and4_frameIndices.pkl')
allIdentities = pd.read_pickle('../Conflict8/conflict3and4_identities.pkl')
path = paths[numSegment]

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    rgbcolorslist = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
    return rgbcolorslist

def IdPlayer(path,allIdentities,frameIndices):

    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    sNumber = int(filename.split('_')[-1])
    folder = os.path.dirname(path)
    df = pd.read_pickle(folder +'/'+ filename + '.pkl')
    cap = cv2.VideoCapture(path)
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    numAnimals = 8
    colors = get_spaced_colors(8)
    print colors
    def onChange(trackbarValue):
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        centroids = df.loc[trackbarValue,'centroids']
        pixels = df.loc[trackbarValue,'pixels']
        permutation = df.loc[trackbarValue,'permutation']
        # print 'previous frame, ', str(trackbarValue-1), ', permutation, ', df.loc[trackbarValue-1,'permutation']
        # print 'current frame, ', str(trackbarValue), ', permutation, ', permutation
        # if sNumber == 1 and trackbarValue > 100:
        #     trueFragment, s = computeFrameIntersection(df.loc[trackbarValue-1,'pixels'],df.loc[trackbarValue,'pixels'],5)
        #     print trueFragment, s
        #     result = df.loc[trackbarValue-1,'permutation'][s]
        #     print 'result, ', result
        #Get frame from video file
        ret, frame = cap.read()
        #Color to gray scale
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Plot segmentated blobs
        # for i, pixel in enumerate(pixels):
        #     px = np.unravel_index(pixel,(height,width))
        #     frame[px[0],px[1]] = 255


        # plot numbers if not crossing
        globalFrame = frameIndices[frameIndices['frame']== trackbarValue][frameIndices['segment']==sNumber].index[0]
        if not isinstance(permutation,float):
            # print 'pass'
            for i, centroid in enumerate(centroids):
                print centroid
                cur_id = allIdentities.loc[globalFrame,i]
                cv2.putText(frame,str(cur_id),centroid, font, 1,0)
                cv2.circle(frame, centroid,2, colors[cur_id],2)

        cv2.putText(frame,str(trackbarValue),(50,50), font, 3,(255,0,0))

        # Visualization of the process
        cv2.imshow('IdPlayer',frame)
        pass

    cv2.namedWindow('IdPlayer')
    print '*************************************************'
    cv2.createTrackbar( 'start', 'IdPlayer', 0, numFrame-1, onChange )
    # cv2.createTrackbar( 'end'  , 'IdPlayer', numFrame-1, numFrame, onChange )

    onChange(1)
    cv2.waitKey()

    start = cv2.getTrackbarPos('start','IdPlayer')

    return raw_input('Which segment do you want to inspect?')

finish = False
while not finish:
    print 'I am here', numSegment
    numSegment = IdPlayer(paths[int(numSegment)],allIdentities,frameIndices)
    if numSegment == 'q':
        finish = True
