import cv2
cap = cv2.VideoCapture('../Cafeina5peces/Caffeine5fish_20140206T122428_2.avi')
numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

def onChange(trackbarValue):
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
    err,img = cap.read()
    cv2.imshow("mywindow", img)
    pass

cv2.namedWindow('mywindow')
print numFrame
cv2.createTrackbar( 'start', 'mywindow', 0, numFrame, onChange )
cv2.createTrackbar( 'end'  , 'mywindow', 0, numFrame, onChange )

onChange(0)
cv2.waitKey()

start = cv2.getTrackbarPos('start','mywindow')
end   = cv2.getTrackbarPos('end','mywindow')
if start >= end:
    raise Exception("start must be less than end")

cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,start)
while cap.isOpened():
    err,img = cap.read()
    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) > end:
        break
    cv2.imshow("mywindow", img)
    k = cv2.waitKey(10) & 0xff
    if k==27:
        break

# #import the necessary packages
# import cv2
# import numpy as np
# #'optional' argument is required for trackbar creation parameters
# def nothing():
#     pass
#
# #Capture video from the stream
# cap = cv2.VideoCapture(0)
# cv2.namedWindow('Colorbars') #Create a window named 'Colorbars'
#
# #assign strings for ease of coding
# hh='Hue High'
# hl='Hue Low'
# sh='Saturation High'
# sl='Saturation Low'
# vh='Value High'
# vl='Value Low'
# wnd = 'Colorbars'
# #Begin Creating trackbars for each
# cv2.createTrackbar(hl, wnd,0,179,nothing)
# cv2.createTrackbar(hh, wnd,0,179,nothing)
# cv2.createTrackbar(sl, wnd,0,255,nothing)
# cv2.createTrackbar(sh, wnd,0,255,nothing)
# cv2.createTrackbar(vl, wnd,0,255,nothing)
# cv2.createTrackbar(vh, wnd,0,255,nothing)
#
# #begin our 'infinite' while loop
# cap = cv2.VideoCapture('./Cafeina5peces/Caffeine5fish_20140206T122428_2.avi')
# while(1):
#     #read the streamed frames (we previously named this cap)
#     _,frame=cap.read()
#
#     cv2.imshow('image',frame)
