import cv2
import os
import sys
sys.path.append('./utils')
from py_utils import *
from video_utils import *

path = './Conflict8/conflict3and4_20120316T155032_1.avi'
paths = [path]
# width, height = getVideoInfo(paths)
print 'Segmenting video %s' % path
cap = cv2.VideoCapture(path)
video = os.path.basename(path)
filename, extension = os.path.splitext(video)
numSegment = int(filename.split('_')[-1])
numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
counter = 0
# df = pd.DataFrame(columns=('avIntensity', 'boundingBoxes','miniFrames', 'contours', 'centroids', 'areas', 'pixels', 'numberOfBlobs', 'bkgSamples'))
maxNumBlobs = 0
while counter < numFrames:

    #Get frame from video file
    ret, frame = cap.read()
    # frameToPlot = masker(frame, maskFrame, ROI, selectROI)
    #Color to gray scale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Equalize image to enhance contrast
    # frame = checkEq(EQ, frame)
    # frameToPlot = masker(frame, maskFrame, ROI, selectROI)
    # # mask or crop the image
    # frame = cropper(frame, ROI)
    # avFrame = frame
    # #Normalize each frame by its mean intensity
    # frame, avIntensity = frameAverager(frame)

    # perform background subtraction if needed
    # segmentedFrame, frameBkg = segmentVideo(frame, minThreshold, maxThreshold, bkg, bkgSubstraction)
    # segmentedFrame = segmentVideo(frame, minThreshold, maxThreshold, bkg, bkgSubstraction)
    # frameBkg = 255 - frameBkg
    # Find contours in the segmented image
    # boundingBoxes, miniFrames, centroids, areas, pixels, goodContoursFull, bkgSamples = blobExtractor(segmentedFrame, avFrame, minArea, maxArea, ROI, height, width)

    # if len(centroids) > maxNumBlobs:
    #     maxNumBlobs = len(centroids)

    ### UNCOMMENT TO PLOT ##################################################
    # cv2.drawContours(frameToPlot,goodContoursFull,-1,color=(255,0,0),thickness=-1)
    cv2.imshow('checkcoord', frame)
    k = cv2.waitKey(30) & 0xFF
    if k == 27: #pres esc to quit
        break
    ########################################################################

    # Add frame imformation to DataFrame
    # df.loc[counter] = [avIntensity, boundingBoxes, miniFrames, goodContoursFull, centroids, areas, pixels, len(centroids), bkgSamples]
    counter += 1
cap.release()
cv2.destroyAllWindows()
