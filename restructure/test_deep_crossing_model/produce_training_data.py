from __future__ import absolute_import, division, print_function

from itertools import count
import os

try: import cPickle as pickle
except: import pickle

import skimage.draw
import numpy as np
import cv2

VIDEO_FOLDER = 'Videos/'
VIDEO_FILE = 'conflict3and4_20120316T155032_'
SEGMENTS = np.arange(16) #segments of video to be processed
EXTENSION = '.avi'
FRAME_FOLDER = 'portraits/'
FRAME_PREFIX = 'fish'
TRAJECTORIES_FILE = 'trajectories.pkl'
PORTRAITS_FILE = 'portraits.pkl'
#BACKGROUND = 'background.png'

PORTRAIT_SIDE = 80
PORTRAIT_LIBRARY_LENGTH = 50000 #Maximum number of portraits, to non-dynamically book memory

def give_me_nose_coordinates(trajectories_filename):
    """Opens a trajectories file, as produced by iDTrackerDeep,
    and returns the coordinates of nose and head centroid.
    """
    with open(trajectories_filename, "rb") as f:
        try: 
            a = pickle.load(f,encoding='latin1')
        except:
            a = pickle.load(f)
    return np.array(a['noses']), np.array(a['head_centroids'])

def valid_coordinates(coordinates):
    """Returns True iff none of the coordinates contains a nan.
    As idTrackerDeep returns nan when the fish is not identified
    in that frame, this function will return True iff all fish were
    identified in the current frame
    """
    return not (np.isnan(coordinates)).any()  

def preprocess_portrait(portrait):
    return (portrait/255 - 0.5).astype(np.float32)

def cut_a_square(frame, center, side):
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

def mark_lines(frame,start,end):
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
 
if __name__ == "__main__":
    """Picks coordinates from idTrackerDeep output file COORDINATES_FILE,
    uses frames from the VIDEO_FOLDER/VIDEO_FILE
    and uses them to output a pickle file with square portraits and
    boolean arrays showing the position of noses,
    or a line between a nose and the head centroid.
    """
    nose_coordinates, head_centroid_coordinates = give_me_nose_coordinates(TRAJECTORIES_FILE)
    number_of_fish = nose_coordinates.shape[1]

    portraits = np.zeros((PORTRAIT_LIBRARY_LENGTH,PORTRAIT_SIDE,PORTRAIT_SIDE,1),dtype=np.float32)
    labels = np.zeros((PORTRAIT_LIBRARY_LENGTH,PORTRAIT_SIDE,PORTRAIT_SIDE,1),dtype=np.bool) 
    line_labels = np.zeros((PORTRAIT_LIBRARY_LENGTH,PORTRAIT_SIDE,PORTRAIT_SIDE,1),dtype=np.bool) 

    global_counter = count() #Frame counter
    valid_counter = count() #Valid portraits counter
    for segment in SEGMENTS:
        video_file = VIDEO_FILE+str(segment)+EXTENSION
        print("Loading ", video_file) 
        cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER,video_file))

        try: #OpenCV 3
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        except: #OpenCV 2
            length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))

        local_counter = count()
        while(cap.isOpened()):
            ret, image = cap.read()
            if ret:                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                local_frame = next(local_counter)
                global_frame = next(global_counter)
                print("Processing frame ", global_frame)

                if valid_coordinates(nose_coordinates[global_frame,:,:]):
                    frame_line_bool = np.zeros((height,width),dtype = np.bool)
                    frame_bool = np.zeros((height,width),dtype = np.bool)
                    #Adding a point in frame_bool, drawing a line in frame_line_bool
                    frame_bool[nose_coordinates[global_frame,:,1].astype(np.uint),nose_coordinates[global_frame,:,0].astype(np.uint)] = True
                    mark_lines(frame_line_bool, nose_coordinates[global_frame].astype(np.uint), head_centroid_coordinates[global_frame].astype(np.uint))

                    for i in range(number_of_fish):
                        valid_index = next(valid_counter)
                        portraits[valid_index,:,:,0] = preprocess_portrait(cut_a_square(gray,nose_coordinates[global_frame,i], PORTRAIT_SIDE))
                        labels[valid_index,:,:,0] = cut_a_square(frame_bool,nose_coordinates[global_frame,i], PORTRAIT_SIDE)
                        line_labels[valid_index,:,:,0] = cut_a_square(frame_line_bool,nose_coordinates[global_frame,i], PORTRAIT_SIDE)
            else: break
        cap.release()

    number_portraits = next(valid_counter)
    print("Number of valid portraits: ", number_portraits)
    cv2.destroyAllWindows()

    # Pickling everything
    a = {}
    a['portraits'] = portraits[:number_portraits]
    a['labels'] = labels[:number_portraits]
    a['line_labels'] = line_labels[:number_portraits]
    pickle.dump( a, open( PORTRAITS_FILE, "wb" ) )

   

