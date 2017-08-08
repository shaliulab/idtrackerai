from __future__ import absolute_import, division, print_function
import os
from itertools import count
import sys
sys.path.append('../utils')
import numpy as np
import cv2
try:
    import cPickle as pickle
except:
    import pickle

from network_nose_detector import ConvNetwork
from video_utils import getSegmPaths
from py_utils import scanFolder
from tqdm import tqdm

PREDICTION_THRESHOLD = .9 #threshold use to filter false positive

class TestNoseDetector(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.frame_indices, self.video_paths = getSegmPaths(scanFolder(self.video_path))
        self.net = ConvNetwork(from_video_path = self.video_path)
        self.global_counter = count()
        #check if a ROI exists
        self.video_folder = os.path.dirname(self.video_path)
        self.roi_path = os.path.join(self.video_folder, 'preprocessing/ROI.pkl')
        print(self.roi_path)
        if os.path.isfile(self.roi_path):
            self.roi = pickle.load(open(self.roi_path, 'rb'))
            self.roi = self.roi.values
        else:
            tempCap = cv2.VideoCapture(self.video_path)
            ret, frame = tempCap.read()
            self.roi = (np.ones_like(frame)*255).astype('uint8')
            tempCap.release()


        if '.plh' in self.video_paths[0]:
            self.feed_images(self.video_path)
        else:
            for video_path in self.video_paths:
                self.feed_images(video_path)

    def feed_images(self, video_path):
        cap = cv2.VideoCapture(video_path)
        local_counter = count()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.bitwise_and(gray,gray, mask=self.roi)
                gray[gray == 0] = 255 ###FIXME ugly trick, gives artifacts on the borders of the ROI
                gray_normalised = gray[None,:,:, None] / 255.0 - 0.5
                prediction = np.squeeze(self.net.prediction(gray_normalised))
                noses_y, noses_x = np.where(prediction > PREDICTION_THRESHOLD)
                local_frame = next(local_counter)
                global_frame = next(self.global_counter)
                for x,y in zip(noses_x,noses_y):
                    cv2.circle(frame, (int(x),int(y)), 2, [255,0,0])
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
            else: break
        cap.release()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    a = TestNoseDetector('../videos/Cafeina5pecesShort/Caffeine5fish_20140206T122428_1.avi')
