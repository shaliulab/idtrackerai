import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.pathces import Ellipse


df = pd.read_hdf('/home/lab/Desktop/TF_models/IdTracker/Cafeina5pecesLarge/20161210194527_/segmentation/segm_1.hdf5')
miniframes = df.miniFrames.tolist()

ex = np.asarray(miniframes[100])


def getSkeleton(cnt, shape):
    width,height = shape
    toile = np.uint8(np.zeros((width, height)))
    cv2.drawContours(toile, [cnt], -1, color=155, thickness = -1)
    toile_el = toile.copy()
    ellipse = cv2.fitEllipse(cnt)
    # print ellipse
    # cv2.ellipse(toile_el,ellipse,255,2)
    # cv2.imshow('fitEll', toile_el)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # print cnt
    # print np.asarray(cnt).shape
    cnt_ext = sorted(np.squeeze(np.asarray(cnt))[:,0])

    minCnt = cnt_ext[0]
    maxCnt = cnt_ext[-1]
    # print maxCnt
    # print minCnt
    ## 1. COMPUTE DISTANCE TRANSFORM
    # compute the distance d(p_1,p_0) between each white pixel and its black nearest neighbour.
    # 3 is the size of the mask, maybe a bit coarse
    distance = cv2.distanceTransform(toile, cv2.cv.CV_DIST_L2,5)
    # ## Uncomment to plot
    plt.ion()
    plt.subplot(2,2,1)
    plt.imshow(distance, interpolation='none')
    center = ellipse[0]
    minAxis = ellipse[1][0]
    maxAxis = ellipse[1][1]
    angle = ellipse[2]
    # ellipse_plot = Ellipse()
    # plt.subplot(2,2,2)
    # th_dist = distance > 0.666*np.max(distance)
    # plt.imshow(th_dist)
    ## 2. COMPUTE LAPLACIAN OF DISTANCE TRANSFORM
    kernel_size = 3
    scale = 1
    delta = 0
    ddepth = cv2.CV_32F
    distance = (distance - np.min(distance))/(np.max(distance) - np.min(distance))
    y_head, x_head = np.unravel_index(distance.argmax(), distance.shape)

    gray_lap = cv2.Laplacian(distance,ddepth, ksize = kernel_size,scale = scale,delta = delta)
    # gray_lap = np.max(gray_lap) - gray_lap
    # ## 3. Threshold the Laplacian
    gray_lap = - gray_lap
    gray_lap2 = gray_lap > np.max(gray_lap)*0.66

    ## Uncomment to plot
    plt.subplot(2,2,2)
    plt.imshow(gray_lap, interpolation='none')
    plt.colorbar()
    plt.show()

    min_lap = np.argmin(gray_lap)
    print 'min Laplacian', min_lap


    plt.subplot(2,2,3)
    plt.imshow(gray_lap2*gray_lap, interpolation='none')
    plt.colorbar()
    plt.show()

    max_lap = np.max(gray_lap)
    max_gray = gray_lap == max_lap
    plt.subplot(2,2,4)
    plt.imshow(max_gray, interpolation='none')
    plt.colorbar()
    plt.show()




for minif in ex:
    plt.figure()
    ret, thEx = cv2.threshold(minif,150,255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thEx,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = contours[np.argmax([len(c) for c in contours])]
    getSkeleton(cnt, minif.shape)
