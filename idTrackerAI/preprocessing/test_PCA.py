from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys
sys.path.append('./')

import numpy as np
import numpy.linalg as npl
from matplotlib import pyplot as plt
import seaborn as sns
import cv2

import time

from video import Video
from blob import compute_fragment_identifier_and_blob_index,\
                connect_blob_list,\
                apply_model_area_to_video
from GUI_utils import selectFile,\
                    getInput,\
                    selectOptions,\
                    ROISelectorPreview,\
                    selectPreprocParams,\
                    fragmentation_inspector,\
                    frame_by_frame_identity_inspector,\
                    selectDir

if __name__ == '__main__':

    video_path = '/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_7/video_object.npy'
    video = np.load(video_path).item(0)
    blobs_path = '/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_7/preprocessing/blobs_collection.npy'
    list_of_blobs = ListOfBlobs.load(blobs_path)
    blobs = list_of_blobs.blobs_in_video

    blob = blobs[2675][0]

    from sklearn.decomposition import PCA
    # PCA 1
    print("PCA 1 numpy")
    start = time.time()
    pxs = np.unravel_index(blob.pixels,(video.height,video.width))
    pxs1 = np.asarray(zip(pxs[0],pxs[1]))
    center1 = (np.mean(pxs[0]),np.mean(pxs[1]))
    cov_mat = np.cov(pxs1.T)
    w,v = npl.eig(cov_mat)
    v_major = v[:,np.argmax(w)]
    a1 = 180 - np.arctan(v_major[1]/v_major[0])*180/np.pi
    print(time.time()-start)
    print(a1)
    print(center1)
    # PCA 2
    print("PCA 2 sklearn")
    start = time.time()
    pca = PCA()
    pca.fit(pxs1)
    a2 = 180 - np.arctan(pca.components_[0][1]/pca.components_[0][0])*180/np.pi
    center2 = pca.mean_
    print(time.time()-start)
    print(a2)
    print(center2)
    # fit ellipse
    print("ellipse cv2")
    start = time.time()
    ellipse = cv2.fitEllipse(blob.contour)
    center = ellipse[0]
    axes = ellipse[1]
    rot_ang = ellipse[2]
    print(time.time()-start)
    print(rot_ang)
    print(center)
