import cv2
"""
Get general information from video
"""
def getVideoInfo(paths):
    if len(paths) == 1:
        path = paths
    elif len(paths) > 1:
        cap = cv2.VideoCapture(paths[0])
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    else:
        raise ValueError('the path (or list of path) seems to be empty')
    return width, height

def getNumFrame(path):
    cap = cv2.VideoCapture(path)
    return int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    # """
    # *** Visualization ***
    # """
    # paths = scanFolder('./Cafeina5peces/Caffeine5fish_20140206T122428_1.avi')
    # for path in paths:
    #     video = os.path.basename(path)
    #     filename, extension = os.path.splitext(video)
    #     folder = os.path.dirname(path)
    #     df = pd.read_pickle(folder +'/'+ filename + '.pkl')
    #     print 'Segmenting video %s' % path
    #     cap = cv2.VideoCapture(path)
    #     numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    #     counter = 0
    #     time.sleep(2)
    #     while counter < numFrame:
    #         print counter
    #         centroids = df.loc[counter,'centroids']
    #         permutation = df.loc[counter,'permutation']
    #         #Get frame from video file
    #         ret, frame = cap.read()
    #         #Color to gray scale
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         # print permutation
    #         if not isinstance(permutation,float):
    #             # print 'pass'
    #             for i, centroid in enumerate(centroids):
    #                 font = cv2.FONT_HERSHEY_SIMPLEX
    #                 cv2.putText(frame,str(permutation[i]),centroid, font, 1,0)
    #         cv2.putText(frame,str(counter),(50,50), font, 3,(255,0,0))
    #
    #         # Visualization of the process
    #         cv2.imshow('ROIFrameContours',frame)
    #         #
    #         # ## Plot miniframes
    #         # for i, miniFrame in enumerate(miniFrames):
    #         #     cv2.imshow('miniFrame' + str(i), miniFrame)
    #         #
    #         k = cv2.waitKey(30) & 0xFF
    #         if k == 27: #pres esc to quit
    #             break
    #
    #         time.sleep(.1)
    #         if isinstance(df.loc[counter,'permutation'],float):
    #             print 'cross or something...'
    #             time.sleep(1)
    #         counter += 1
