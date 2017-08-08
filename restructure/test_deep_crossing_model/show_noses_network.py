from __future__ import absolute_import, division, print_function
import os
from itertools import count

import numpy as np
import cv2

from network import ConvNetwork 

VIDEO_FOLDER = 'Videos/'
VIDEO_FILE = 'conflict3and4_20120316T155032_'
SEGMENTS = np.arange(0,19)
EXTENSION = '.avi'
FRAME_FOLDER = 'portraits/'
FRAME_PREFIX = 'fish'

if __name__ == "__main__":
    global_counter = count()
    net = ConvNetwork(from_file='/tmp/checkpoint-384')
    for segment in SEGMENTS:
        video_file = VIDEO_FILE+str(segment)+EXTENSION
        print("Loading ", video_file) 
        cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER,video_file))
        local_counter = count()
        while(cap.isOpened()):
            ret, image = cap.read()
            if ret:                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_ = gray[None,:,:,None]/255.0 - 0.5
                prediction = np.squeeze(net.prediction(gray_))
                noses_y, noses_x = np.where(prediction>0.9)
                local_frame = next(local_counter)
                global_frame = next(global_counter)
                print("Frame: ", global_frame)
                for x,y in zip(noses_x,noses_y):
                    cv2.circle(image, (int(x),int(y)), 2, [255,0,0])
                cv2.imshow('frame', image)
                cv2.waitKey(1)
            else: break
        cap.release()


    cv2.destroyAllWindows()
