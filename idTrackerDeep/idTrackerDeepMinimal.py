# This file is a reduced version of idTrackerDeepGUI, to work in preprocessing without the need of using tensorflow


# Import standard libraries

import sys

# Import third party libraries
import cv2

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/preprocessing')
sys.path.append('IdTrackerDeep/tracker')

from segmentation import *
from fragmentation import *
from get_portraits import *
from video_utils import *
from py_utils import *
from GUI_utils import *
#from idAssigner import *
#from fragmentFinder import *
#from fineTuner import *
#from tracker import *
#from plotters import *


if __name__ == '__main__':
    cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...

    print '\n********************************************************************'
    print 'Selecting the path to the videos...'
    print '********************************************************************\n'

    initialDir = ''
    videoPath = selectFile() ### NOTE The video to be tracked need to be splited in to small segments of video with a suffix '_(numSegment)'. The video selected has to be the one with extension suffix '_1'
    print 'The video selected is, ', videoPath
    videoPaths = scanFolder(videoPath) ### FIXME if the video selected does not finish with '_1' the scanFolder function won't select all of them. This can be improved
    print 'The list of videos is ', videoPaths

    print '\n********************************************************************'
    print 'Asking user whether to reuse preprocessing steps...'
    print '********************************************************************\n'
    reUseAll = getInput('Reuse all preprocessing, ', 'Do you wanna reuse all previos preprocessing? ([y]/n)')

    if reUseAll == 'n':
        print '\n********************************************************************'
        print 'Selecting preprocessing parameters...'
        print '********************************************************************\n'

        prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
        useBkg = int(prepOpts['bkg'])
        useROI =  int(prepOpts['ROI'])
        print 'useBkg set to ', useBkg
        print 'useROI set to ', useROI

        print '\nLooking for finished steps in previous session...'
        processesList = ['ROI', 'bkg', 'preprocparams', 'segmentation','fragments','portraits']

        existentFiles, srcSubFolder = getExistentFiles(videoPath, processesList)
        print 'List of processes finished, ', existentFiles
        print '\nSelecting files to load from previous session...'
        loadPreviousDict = selectOptions(processesList, existentFiles, text='Already processed steps in this video \n (check to load from ' + srcSubFolder + ')')

        usePreviousROI = loadPreviousDict['ROI']
        usePreviousBkg = loadPreviousDict['bkg']
        usePreviousPrecParams = loadPreviousDict['preprocparams']
        print 'usePreviousROI set to ', usePreviousROI
        print 'usePreviousBkg set to ', usePreviousBkg
        print 'usePreviousPrecParams set to ', usePreviousPrecParams

        ''' ROI selection/loading '''
        width, height, mask, centers = ROISelectorPreview(videoPaths, useROI, usePreviousROI, numSegment=0)
        ''' BKG computation/loading '''
        bkg = checkBkg(videoPaths, useBkg, usePreviousBkg, 0, width, height)

        ''' Selection/loading preprocessing parameters '''
        preprocParams = selectPreprocParams(videoPaths, usePreviousPrecParams, width, height, bkg, mask, useBkg)
        print 'The video will be preprocessed according to the following parameters: ', preprocParams

        ''' Loading preprocessing image '''
        img = cv2.imread('IdTrackerDeep/utils/loadingIdDeep.png')
        cv2.imshow('Bars',img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    elif reUseAll == '' or reUseAll.lower() == 'y' :
        print '\n********************************************************************'
        print 'The preprocessing paramemters will be loaded from last time they were computed.'
        print '********************************************************************\n'
        loadPreviousDict = {'ROI': 1, 'bkg': 1, 'preprocparams': 1, 'segmentation': 1, 'fragments': 1, 'portraits': 1}

    else:
        raise ValueError('The input introduced do not match the possible options')