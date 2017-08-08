from __future__ import absolute_import, division, print_function

from itertools import count
import os
import sys
sys.path.append('../utils')
try: import cPickle as pickle
except: import pickle

import skimage.draw
import numpy as np
import cv2
from tqdm import tqdm

from py_utils import saveFile, loadFile, scanFolder
from video_utils import getSegmPaths

"""After segmentation uses the permutations stored in
video_folder/preprocessing/segmentation to extract the coordinates of both
noses and head centroids, and the non-crossing frames. It cuts square images
around the coordinate of the nose of each fish and organizes them in a dataset
of images, together with the labels. It ouputs a dataset organised as follows:
nose_detector_dataset = {
'images': the images obtained by cutting a square around the nose of each
          fish in the original square (only renormalised to [-1,1] from uint8)
          shape: [portrait_library_length, PORTRAIT_SIDE, PORTRAIT_SIDE]

'labels': array of boolean matrix of the same size of images with True where
       the nose has been detected during portraying

'line_labels': the same as labels, but True correspond to a line-segment
       connecting the nose to the centroid of the head of the fish
}

"""

PORTRAIT_SIDE = 80

class GetTrainData(object):
    """Given a path to a video finds, after segmentation and portraying are computed
    """
    def __init__(self, video_path, save_flag = True):
        self.video_path = video_path
        self.save_flag = save_flag
        self.get_video_info()
        self.images = np.zeros((self.portrait_library_length,PORTRAIT_SIDE,PORTRAIT_SIDE,1),dtype=np.float32)
        self.labels = np.zeros((self.portrait_library_length,PORTRAIT_SIDE,PORTRAIT_SIDE,1),dtype=np.bool)
        self.line_labels = np.zeros((self.portrait_library_length,PORTRAIT_SIDE,PORTRAIT_SIDE,1),dtype=np.bool)
        self.portraits_counter = 0
        self.get_miniframes()
        self.save_training_data()

    def get_video_info(self):
        #load video_info. It is a dictionary created during segmentation and it
        #contains:
        # ['numCores', 'numFrames', 'stdIndivArea', 'minThreshold', 'maxThreshold'
        # , 'meanIndivArea', 'maxNumBlobs', 'numAnimals', 'path', 'height',
        # 'width', 'maxArea']
        self.video_info = loadFile(self.video_path, 'videoInfo', hdfpkl='pkl')
        self.num_animals = self.video_info['numAnimals']
        self.num_frames = self.video_info['numFrames']
        self.max_number_blobs = self.video_info['maxNumBlobs']

        self.portrait_library_length = self.num_animals * self.num_frames
        self.frame_indices, self.video_paths = getSegmPaths(scanFolder(self.video_path),framesPerSegment=500)
        portraits_dict = loadFile(self.video_path, 'portraits', hdfpkl='pkl')
        self.noses = portraits_dict['noses'].values
        self.head_centroids = portraits_dict['head_centroids'].values

    def get_miniframes(self):
        diff = self.max_number_blobs - self.num_animals

        for video_path in tqdm(self.video_paths, desc='percentage of processed video'):
            #segmentation_dataFrame contains
            #[u'avIntensity', u'boundingBoxes', u'miniFrames', u'contours',
            # u'centroids', u'areas', u'pixels', u'numberOfBlobs', u'bkgSamples',
            # u'permutation']
            segmentation_dataFrame, seg_num = loadFile(video_path, 'segmentation')
            #hierarchies is a np array containing the hierarchy of each blob
            #detected during the segmentation
            hierarchies = segmentation_dataFrame.loc[: , 'permutation'].values
            no_crossing_indices = [i for i in range(len(hierarchies)) if list(hierarchies[i]).count(-1) <= diff]###TODO consider adding a flag directly in fragmentation
            self.cap = cv2.VideoCapture(self.video_path)
            self.get_images_from_frames(video_path, no_crossing_indices, seg_num)


    def get_images_from_frames(self, video_path, no_crossing_indices, seg_num):
        """We capture the video again, since we need to cut square around the
        nose of each fish, if no crossing occurs"""
        if '.avi' in video_path:
            self.cap = cv2.VideoCapture(video_path)

        for frame_num in no_crossing_indices:
            query_string = 'frame==' + str(frame_num) + '&' + 'segment==' + seg_num
            global_frame_num = self.frame_indices.query(query_string).index.values[0]

            if '.avi' in video_path:
                self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_num)
            elif '.plh' in video_path:
                self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,global_frame_num)

            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #get coordinates of each nose detected in a non-crossing frame
            nose_coordinates = self.noses[global_frame_num]
            head_centroid_coordinates = self.head_centroids[global_frame_num]
            frame_line_bool = np.zeros_like(frame, dtype = np.bool)
            frame_bool = np.zeros_like(frame, dtype = np.bool)
            #Adding a point in frame_bool, drawing a line in frame_line_bool
            frame_bool[np.asarray(nose_coordinates)[:,1], np.asarray(nose_coordinates)[:,0]] = True
            self.mark_lines(frame_line_bool, np.asarray(nose_coordinates), np.asarray(head_centroid_coordinates))

            for nose in nose_coordinates:
                self.portraits_counter += 1
                self.images[self.portraits_counter,:,:,0] = self.preprocess_portrait(self.cut_a_square(frame, nose, PORTRAIT_SIDE))
                self.labels[self.portraits_counter,:,:,0] = self.cut_a_square(frame_bool,nose, PORTRAIT_SIDE)
                self.line_labels[self.portraits_counter,:,:,0] = self.cut_a_square(frame_line_bool,nose, PORTRAIT_SIDE)
                ##uncomment to visualise images and labels
                # cv2.imshow('image', self.images[self.portraits_counter,:,:,0])
                # cv2.imshow( 'label nose ', (self.labels[self.portraits_counter,:,:,0]*255).astype('uint8'))
                # cv2.imshow('label line ', (self.line_labels[self.portraits_counter,:,:,0] * 255).astype('uint8'))
                # cv2.waitKey(100)

    def preprocess_portrait(self, portrait):
        return (portrait/255 - 0.5).astype(np.float32)

    def cut_a_square(self, frame, center, side):
        """Cuts and returns a square in frame
        """
        h,w = frame.shape
        [x,y] = center #Center of the square
        s = int(side/2)
        x = max(x-s,0) + s #Maybe we need to shift the square to fit the frame
        x = int(min(x+s,w) - s)
        y = max(y-s,0) + s
        y = int(min(y+s,h) - s)
        return frame[(y-s):(y+s),(x-s):(x+s)]

    def mark_lines(self,frame,start,end):
        """Draws straight lines of True values on a boolean array.
        Just the simplest thing, no blurring or any anti-aliasing shit.
        DANGER: Frame is modified in place, without any copy!

        :param frame: Boolean 2D frame
        :param start: 2D array of starting points for lines in (x,y)
        :param end: 2D array of ending points for lines in (x,y)
        """
        assert start.shape == end.shape #Same number of starting and ending points
        for i in range(start.shape[0]):
            rr,cc = skimage.draw.line(start[i,1],start[i,0],end[i,1],end[i,0])
            frame[rr,cc] = True
        return frame

    def save_training_data(self):
        self.data_dict = {}
        self.data_dict['images'] = self.images[:self.portraits_counter]
        self.data_dict['labels'] = self.labels[:self.portraits_counter]
        self.data_dict['line_labels'] = self.line_labels[:self.portraits_counter]
        if self.save_flag:
            print("saving data ...")
            saveFile(self.video_path, self.data_dict, 'nose_detector_dataset', hdfpkl = 'pkl')


if __name__ == "__main__":

    a = GetTrainData('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/test/1.avi', save_flag = False )
    print(len(a.data_dict['labels']))
