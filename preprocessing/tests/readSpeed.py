import cv2
import sys
import time

cap = cv2.VideoCapture('../test2.avi')
start = time.time()

counter = 0
while True:
    counter += 1;
    image = cap.read()[1]
    if counter %1 == 0:
        print "time", time.time() - start
