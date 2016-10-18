import cv2
import numpy as np

cap = cv2.VideoCapture('./data/library/group_2_camera_2/group_2_camera_2_20160508T094727_13.avi')
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('image',frame)
circles = cv2.HoughCircles(frame,cv2.cv.CV_HOUGH_GRADIENT,1,300,param1=50,param2=45,minRadius=200,maxRadius=250)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
