from __future__ import division

# Import third party libraries
import cv2
import numpy as np

if __name__ == '__main__':
    video_path = '/home/themis/Desktop/IdTrackerDeep/videos/thermites_dani_calovi/thermites_0.avi'
    cap = cv2.VideoCapture(video_path)
    # image = np.zeros((1000,1000,3))
    # image[250:750,250:750,0] = 255


    #optional argument
    def nothing(x):
        pass
    cv2.namedWindow('image')

    #easy assigments
    hb='High Blue'
    lb='Low Blue'
    hg='High Green'
    lg='Low Green'
    hr='High Red'
    lr='Low Red'

    cv2.createTrackbar(lb, 'image',0,255,nothing)
    cv2.createTrackbar(hb, 'image',0,255,nothing)
    cv2.createTrackbar(lg, 'image',0,255,nothing)
    cv2.createTrackbar(hg, 'image',0,255,nothing)
    cv2.createTrackbar(lr, 'image',0,255,nothing)
    cv2.createTrackbar(hr, 'image',0,255,nothing)

    while(1):
        _,frame=cap.read()
        frame=cv2.GaussianBlur(frame,(5,5),0)
        #convert to HSV from BGR



        #read trackbar positions for all
        lowB=cv2.getTrackbarPos(lb, 'image')
        highB=cv2.getTrackbarPos(hb, 'image')
        lowG=cv2.getTrackbarPos(lg, 'image')
        highG=cv2.getTrackbarPos(hg, 'image')
        lowR=cv2.getTrackbarPos(lr, 'image')
        highR=cv2.getTrackbarPos(hr, 'image')
        #make array for final values
        RGBLOW=np.array([lowB,lowG,lowR])
        RGBHIGH=np.array([highB,highG,highR])

        #apply the range on a mask
        mask = cv2.inRange(frame,RGBLOW, RGBHIGH)
        res = cv2.bitwise_and(frame,frame, mask =mask)

        cv2.imshow('image', res)
        cv2.imshow('yay', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break


    cv2.destroyAllWindows()
