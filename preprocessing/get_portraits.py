# Import standard libraries
import sys
import numpy as np
import multiprocessing
import math

# Import third party libraries
from matplotlib import pyplot as plt
import cv2
import pandas as pd
from joblib import Parallel, delayed
import scipy.ndimage
from scipy.signal import argrelmax

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
from py_utils import *
from video_utils import *

def smooth(x,window_len=20,window='hanning'):
   """smooth the data using a window with requested size."""
   if x.ndim != 1:
       raise ValueError, "smooth only accepts 1 dimension arrays."

   if x.size < window_len:
       raise ValueError, "Input vector needs to be bigger than window size."


   if window_len<3:
       return x

   if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
       raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

   s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
   # s = x
   #print(len(s))
   if window == 'flat': #moving average
       w=np.ones(window_len,'d')
   else:
       w=eval('np.'+window+'(window_len)')

   y=np.convolve(s,w/w.sum(),mode='valid')
   return y

def smoother(contour):
    """
    smooth contour by convolution
    """
    window = 11
    X = contour[:,0]
    X = np.append(X,X[:window])
    Y = contour[:,1]
    Y = np.append(Y,Y[:window])

    bkg = np.ones((np.max(Y),np.max(X)))

    G = cv2.transpose(cv2.getGaussianKernel(window, 32, cv2.CV_64FC1))

    X_smooth = np.convolve(X,G[0],mode='same')
    Y_smooth = np.convolve(Y,G[0],mode='same')
    X_smooth = X_smooth[int(np.floor(window/2)):-int(np.ceil(window/2))]
    Y_smooth = Y_smooth[int(np.floor(window/2)):-int(np.ceil(window/2))]
    return X_smooth, Y_smooth

def smooth_resample(counter, contour,smoothFlag = False):
    if smoothFlag:
        x,y  = smoother(contour)
    else:
        x = contour[:,0]
        y = contour[:,1]

    x_new = x
    y_new = y


    M = 1000
    while 2*len(x_new) >= M: ### FIXME When the contour is too big we need to increase the number of points for the resampling.
    # Mainly this happens because the contour does not belong to a single animal. We need to check whether there is a better way of discarding blobs belonging to crossings.
        M += 500

    t = np.linspace(0, len(x_new), M)
    x = np.interp(t, np.arange(len(x_new)), x_new)
    y = np.interp(t, np.arange(len(y_new)), y_new)
    tol = .1
    i, idx = 0, [0]
    i_new = i
    while i < len(x):
        # print 'i, ', i
        total_dist = 0
        for j in range(i+1, len(x)):
            total_dist += np.sqrt((x[j]-x[j-1])**2 + (y[j]-y[j-1])**2)
            if total_dist > tol:
                idx.append(j)
                break

        i = j+1
        if i == i_new:
            print 'i, ', i
            print 'tol, ', tol
            print 'total_dist, ', total_dist
            print 'M, ', M
            print 'len contour, ', len(x_new)
            print 'perimiter, ', np.sum(np.sqrt(np.diff(x_new)**2 + np.diff(y_new)**2))
            plt.ion()
            plt.plot(x,y,'-b')
            plt.pause(3)
            plt.show()
            raise ValueError('It got stuck in the while loop of the resampling of the contour to compute the curvature')

        i_new = i


    xn = x[idx]
    yn = y[idx]
    sContour = [[x,y] for (x,y) in zip(xn,yn)]

    return sContour

def phi(ind, cnt): # FIXME we need to understand the arctan2 and return the correct angle to compute the curvature and do not have artifacts
    cnt = np.squeeze(cnt)
    points = [cnt[(ind-1) % (len(cnt)-1)], cnt[(ind+1) % (len(cnt)-1)]]
    return math.atan2((points[1][1] - points[0][1]), (points[1][0] - points[0][0]))
    # atan = np.arctan2((points[1][1] - points[0][1]), (points[1][0] - points[0][0]))
    # if atan > 0:
    #     atan = np.pi*2 - atan
    # # else:
    # #     atan = -atan
    # return atan

def curv(cnt,i,n,orientation):
    left = (i + n) % (len(cnt)-1)
    right = (i - n) % (len(cnt)-1)
    phi_l = phi(left, cnt) #% 360
    phi_r = phi(right, cnt) #% 360

    if cnt[left+1][0] <= cnt[right+1][0]:
        if phi_l <= 0:
            phi_l += 2*np.pi
        if phi_r <= 0:
            phi_r += 2*np.pi

    return orientation*(phi_l - phi_r) /(2*n+1) # good one

def get_extrema(curvature):
    gradients = np.diff(curvature)
    maxima_num=0
    minima_num=0
    max_locations=[]
    min_locations=[]
    count=0
    for i in gradients[:-1]:
        count+=1

        if ((cmp(i,0)>0) & (cmp(gradients[count],0)<0) & (i != gradients[count])):
            maxima_num+=1
            max_locations.append(count)

        if ((cmp(i,0)<0) & (cmp(gradients[count],0)>0) & (i != gradients[count])):
            minima_num+=1
            min_locations.append(count)

    return np.asarray(max_locations), np.asarray(min_locations)

def getMiddle(points):
    l = points.shape[0]
    b = np.sum(points, axis=0)
    b = np.divide(b,l)
    return b

def full2miniframe(point, boundingBox):
    """
    Push a point in the fullframe to miniframe coordinate system.
    Here it is use for centroids
    """
    return tuple(np.asarray(point) - np.asarray([boundingBox[0][0],boundingBox[0][1]]))

def getEncompassingIndices(frameIndices, num_segmnent, goodIndices):
    """
    frameIndices = dataframe containing the list of frame per segment
    num_segment
    goodIndices = indices in which the permutation is defined (non-crossing and overlapping)
    """
    frameSegment = frameIndices.loc[frameIndices.loc[:,'segment']==num_segmnent]
    goodFrameIndices = frameSegment.iloc[goodIndices].index.tolist()

    return goodFrameIndices, frameSegment.index.tolist()

def getMFandC(path, frameIndices):
    """
    path: path to dataframe
    generate a list of arrays containing miniframes and centroids detected in
    path at this point we can already discard miniframes that does not belong
    to a specific fragments
    """
    # get number of segment
    df, numSegment = loadFile(path, 'segmentation')
    # check if permutations are NaN (i.e. the frame is not included in a fragment)
    permutationsBool = np.asarray(df['permutation'].notnull())

    #generate a lists of "admissible" miniframes and centroids
    permutations = np.asarray(df['permutation'])
    boundingBoxes = np.asarray(df.loc[:, 'boundingBoxes'])
    miniframes = np.asarray(df.loc[:, 'miniFrames'])
    centroids = np.asarray(df.loc[:, 'centroids'])
    bkgSamples = np.asarray(df.loc[:,'bkgSamples'])

    goodIndices = np.where(permutationsBool==True)[0]
    goodFrameIndices, segmentIndices = getEncompassingIndices(frameIndices, int(numSegment), goodIndices)
    goodFrameIndices = segmentIndices

    return boundingBoxes.tolist(), miniframes.tolist(), centroids.tolist(), bkgSamples.tolist(), goodFrameIndices, segmentIndices, permutations.tolist()

def fillSquareFrame(square_frame,bkgSamps):
    numSamples = 0
    threshold = 150
    while numSamples <= 10:

        bkgSampsNew = bkgSamps[bkgSamps > threshold]
        threshold -= 10
        if threshold == 0:
            plt.imshow(square_frame,cmap='gray',interpolation='none')
            plt.show()
            break
        numSamples = len(bkgSampsNew)
    bkgSamps = bkgSampsNew
    if numSamples == 0:
        raise ValueError('I do not have enough samples for the background')

    numSamplesRequested = np.sum(square_frame == 0)
    indicesSamples = np.random.randint(0,numSamples,numSamplesRequested)
    square_frame[square_frame == 0] = bkgSamps[indicesSamples]
    return square_frame

def getPortrait(miniframe,cnt,bb,bkgSamp,counter = None):
    height, width = miniframe.shape
    orientation = np.sign(cv2.contourArea(cnt,oriented=True)) ### TODO this can probably be optimized

    # Pass contour to bb coord, resample, smooth, and duplicate
    cnt = full2miniframe(cnt, bb)
    cnt = np.asarray(cnt)
    cnt = np.squeeze(cnt)
    cnt = smooth_resample(counter, cnt,smoothFlag=True)
    cnt = np.vstack([cnt,cnt])

    # Compute curvature
    curvature = [curv(cnt,i,3,orientation) for i in range(len(cnt))]
    curvature = np.asarray(curvature)


    # Smooth curvature
    window = 51
    curvature = smooth(curvature, window)
    index = (window-1)/2
    curvature = curvature[index:-index]

    # Crop contour and curvature from the first zero curvature point
    zero = np.argmin(np.abs(curvature[:len(curvature)/2]))
    i1 = zero
    i2 = zero+len(cnt)/2
    cnt = cnt[i1:i2]
    curvature = curvature[i1:i2]

    ### Uncomment to plot
    # plt.close("all")
    # plt.ion()
    # plt.figure()
    # plt.plot(curvature)
    # plt.figure()
    # plt.plot(cnt[:,0],cnt[:,1],'o')
    # plt.show()
    # plt.pause(.5)

    # Find first two maxima (tail and nose), get middle point, and angle of rotation
    max_locations, min_locations = get_extrema(curvature)
    sorted_locations = max_locations[np.argsort(curvature[max_locations])][::-1]
    maxCoord = [cnt[max_loc,:] for max_loc in sorted_locations]
    nose = maxCoord[1]
    cntL = cnt.shape[0]
    steps = 50
    support = np.asarray([cnt[(sorted_locations[1] + steps) % cntL,:], cnt[(sorted_locations[1] - steps) % cntL,:]])
    m = getMiddle(support)
    rot_ang = -(-np.arctan2(nose[1]-m[1],nose[0]-m[0])-np.pi/2)*180/np.pi

    # Copy miniframe in a bigger frame to rotate
    rowsMin, colsMin = miniframe.shape

    diag = np.round(np.sqrt(rowsMin**2 + colsMin**2)).astype('int')
    new_frame = np.zeros((diag,diag)).astype('uint8')
    x_offset = np.ceil((diag-colsMin)/2).astype('int')
    y_offset = np.ceil((diag-rowsMin)/2).astype('int')
    new_frame[y_offset:y_offset + rowsMin, x_offset:x_offset+colsMin] = miniframe
    new_frame = fillSquareFrame(new_frame,bkgSamp)
    # print 'shape of the new miniframe, ', new_frame.shape

    # Translate and rotate nose and middle point to the new frame
    new_nose = tuple(np.asarray([nose[0]+x_offset, nose[1]+y_offset]).astype('int'))
    new_m = tuple(np.asarray([m[0]+x_offset, m[1]+y_offset]).astype('int'))

    # Get roto-translation matrix and rotate nose and miniframe
    M = cv2.getRotationMatrix2D(new_m, rot_ang,1)
    R = np.matrix([M[0][0:2], M[1][0:2]])
    T = np.matrix([M[0][2],M[1][2]])
    nose_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R, np.asmatrix(new_nose).T))),T).astype('int'))[0]
    minif_rot = cv2.warpAffine(new_frame, M, (diag,diag),flags = cv2.INTER_NEAREST)

    # Crop the image in 32x32 frame around the nose
    nose_pixels = [int(nose_rt[0]),int(nose_rt[1])]
    # print counter
    # if counter == 251:
    #     plt.close("all")
    #     plt.ion()
    #     plt.figure()
    #     plt.imshow(minif_rot,interpolation='none',cmap='gray')
    #     # plt.figure()
    #     # plt.plot(cnt[:,0],cnt[:,1],'o')
    #     # plt.show()
    #     plt.pause(.5)
    # print 'nose pixels, ', nose_pixels
    if nose_pixels[1]<7:
        nose_pixels[1] = 7
    portrait = minif_rot[nose_pixels[1]-7:nose_pixels[1]+25,nose_pixels[0]-16:nose_pixels[0]+16]
    if portrait.shape[0] != 32 or portrait.shape[1] != 32:
        print portrait.shape
        raise ValueError('This portrait do not have 32x32 pixels. Changes in light during the video could deteriorate the blobs: try and rais the threshold in the preprocessing parametersm, and run segmentation and fragmentation again.')

    noseFull = cntBB2Full(nose,bb)
    return portrait, tuple(noseFull.astype('int'))

def reaper(videoPath, frameIndices):
    # only function called from idTrackerDeepGUI
    print 'reaping', videoPath
    df, numSegment = loadFile(videoPath, 'segmentation')

    boundingboxes = np.asarray(df.loc[:, 'boundingBoxes']) #coordinate in the frame
    miniframes = np.asarray(df.loc[:, 'miniFrames']) #image containing the blob, same size
    miniframes = np.asarray(miniframes)
    contours = np.asarray(df.loc[:, 'contours'])
    bkgSamples = np.asarray(df.loc[:,'bkgSamples'])
    centroidsSegment = np.asarray(df.loc[:,'centroids'])

    segmentIndices = frameIndices.loc[frameIndices.loc[:,'segment']==int(numSegment)]
    segmentIndices = segmentIndices.index.tolist()


    """ Visualise """
    AllPortraits = pd.DataFrame(index = segmentIndices, columns= ['images'])
    AllNoses = pd.DataFrame(index = segmentIndices, columns= ['noses'])
    AllCentroids= pd.DataFrame(index = segmentIndices, columns= ['centroids'])

    counter = 0
    while counter < len(miniframes):
        portraits = []
        noses = []
        bbs = boundingboxes[counter]
        minif = miniframes[counter]
        cnts = contours[counter]
        bkgSamps = bkgSamples[counter]
        centroids = centroidsSegment[counter]
        for j, miniframe in enumerate(minif):
            ### Uncomment to plot
            # cv2.imshow('frame', miniframe)
            # cv2.waitKey()
            portrait, nose_pixels = getPortrait(miniframe,cnts[j],bbs[j],bkgSamps[j],j)

            # get all the heads in a single list
            portraits.append(portrait)
            noses.append(nose_pixels)
        ### UNCOMMENT TO PLOT ##################################################
        #     cv2.imshow(str(j),portrait)
        #
        # k = cv2.waitKey(100) & 0xFF
        # if k == 27: #pres esc to quit
        #     break
        ########################################################################

        AllPortraits.set_value(segmentIndices[counter], 'images', portraits)
        AllNoses.set_value(segmentIndices[counter], 'noses', noses)
        AllCentroids.set_value(segmentIndices[counter], 'centroids', centroids)
        counter += 1
    print 'you just reaped', videoPath
    return AllPortraits, AllNoses, AllCentroids

def portrait(videoPaths, dfGlobal):
    frameIndices = loadFile(videoPaths[0], 'frameIndices')
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    allPortraitsAndNoses = Parallel(n_jobs=num_cores)(delayed(reaper)(videoPath,frameIndices) for videoPath in videoPaths)
    allPortraits = [t[0] for t in allPortraitsAndNoses]
    allNoses = [t[1] for t in allPortraitsAndNoses]
    allCentroids = [t[2] for t in allPortraitsAndNoses]
    allPortraits = pd.concat(allPortraits)
    allPortraits = allPortraits.sort_index(axis=0,ascending=True)
    allNoses = pd.concat(allNoses)
    allNoses = allNoses.sort_index(axis=0, ascending=True)
    allCentroids = pd.concat(allCentroids)
    allCentroids = allCentroids.sort_index(axis=0, ascending=True)


    if list(allPortraits.index) != list(dfGlobal.index):
        raise ValueError('The list of indexes in allPortraits and dfGlobal should be the same')
    dfGlobal['images'] = allPortraits
    dfGlobal['identities'] = dfGlobal['permutations']
    dfGlobal['noses'] = allNoses
    dfGlobal['centroids'] = allCentroids

    saveFile(videoPaths[0], dfGlobal, 'portraits')
    return dfGlobal
