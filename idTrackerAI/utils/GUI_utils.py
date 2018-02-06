from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from shutil import move
import sys
# Import application/library specifics
sys.path.append('../preprocessing')
import numpy as np
import logging

# Import third party libraries
import cv2
from pylab import *
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set(style="white", context="talk")
from scipy import ndimage
import pyautogui
import Tkinter, tkSimpleDialog, tkFileDialog,tkMessageBox
from Tkinter import Tk, Label, W, IntVar, Button, Checkbutton, Entry, mainloop
from tqdm import tqdm
from segmentation import segmentVideo
from video_utils import checkBkg, blobExtractor
from py_utils import get_existent_preprocessing_steps
from blob import Blob

if sys.argv[0] == 'idtrackerdeepApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.GUI_utils")

"""
Display messages and errors
"""
def load_previous_dict_check(processes, loadPreviousDict):
    zero_key_indices = [ind for ind, key in enumerate(processes)
                        if loadPreviousDict[key] == 0]
    if len(zero_key_indices) > 0:
        for i, p in enumerate(processes):
            if i > zero_key_indices[0]: loadPreviousDict[p] = 0

    for key in loadPreviousDict:
        if loadPreviousDict[key] == -1: loadPreviousDict[key] = 0
    return loadPreviousDict

def selectOptions(optionsList, loadPreviousDict=None, text="Select preprocessing options:  ", is_processes_list = True):
    master = Tk()
    if loadPreviousDict==None:
        loadPreviousDict = {el:'1' for el in optionsList}

    def createCheckBox(name,i):
        var = IntVar()
        Checkbutton(master, text=name, variable=var).grid(row=i+1, sticky=W)
        return var

    Label(master, text=text).grid(row=0, sticky=W)
    variables = []

    for i, opt in enumerate(optionsList):
        if loadPreviousDict[opt] == '1':
            var = createCheckBox(opt,i)
            variables.append(var)
            var.set(loadPreviousDict[opt])
        elif loadPreviousDict[opt] == '0':
            var = createCheckBox(opt,i)
            variables.append(var)
            var.set(loadPreviousDict[opt])
        else:
            Label(master, text= '     ' + opt).grid(row=i+1, sticky=W)
            var = IntVar()
            var.set('-1')
            variables.append(var)

    Button(master, text='Ok', command=master.quit).grid(row=i+2, sticky=W, pady=4)
    mainloop()
    varValues = []
    for var in variables:
        varValues.append(var.get())
    if is_processes_list:
        loadPreviousDict = load_previous_dict_check(optionsList, dict((key, value) for (key, value) in zip(optionsList, varValues)))
    else:
        loadPreviousDict = dict((key, value) for (key, value) in zip(optionsList, varValues))

    master.destroy()
    return loadPreviousDict

def selectFile():
    root = Tk()
    root.withdraw()
    filename = tkFileDialog.askopenfilename()
    root.destroy()
    return filename

def selectDir(initialDir, text = "Select folder"):
    root = Tk()
    root.withdraw()
    Label(root, text=text).grid(row=0, sticky=W)
    dirName = tkFileDialog.askdirectory(initialdir = initialDir)
    root.destroy()
    return dirName

def getInput(name,text):
    root = Tk() # dialog needs a root window, or will create an "ugly" one for you
    root.withdraw() # hide the root window
    inputString = tkSimpleDialog.askstring(name, text, parent=root)
    root.destroy() # clean up after yourself!
    return inputString.lower()

def displayMessage(title,message):
    window = Tk()
    window.wm_withdraw()
    window.geometry("1x1+"+str(window.winfo_screenwidth()/2)+"+"+str(window.winfo_screenheight()/2))
    tkMessageBox.showinfo(title=title, message=message)

def displayError(title, message):
    window = Tk()
    window.wm_withdraw()
    window.geometry("1x1+200+200")
    tkMessageBox.showerror(title=title,message=message,parent=window)

def getMultipleInputs(winTitle, inputTexts):
    #Gui Things
    def retrieve_inputs():
        global inputs
        inputs = [var.get() for var in variables]
        window.destroy()
        return inputs
    window = Tk()
    window.title(winTitle)
    variables = []

    for inputText in inputTexts:
        text = Label(window, text =inputText)
        guess = Entry(window)
        variables.append(guess)
        text.pack()
        guess.pack()
    finished = Button(text="ok", command=retrieve_inputs)
    finished.pack()
    window.mainloop()
    return inputs

'''****************************************************************************
Resolution reduction
****************************************************************************'''
def check_resolution_reduction(video, old_video, usePreviousRR):
    if video.reduce_resolution:
        if usePreviousRR and old_video.resolution_reduction is not None:
            if hasattr(old_video, 'resolution_reduction'):
                return old_video.resolution_reduction
            else:
                return float(getInput('Resolution reduction', 'Resolution reduction parameter not found in previous video. \nInput the resolution reduction factor (.5 would reduce by half): '))
        else:
            return float(getInput('Resolution reduction', 'Input the resolution reduction factor (.5 would reduce by half): '))
    else:
        return 1

''' ****************************************************************************
ROI selector GUI
*****************************************************************************'''
def getMask(im):
    """Returns a uint8 mask following openCV convention
    as used in segmentation (0 invalid, 255 valid)
    adapted from: matplotlib.org/examples/widgets/rectangle_selector.py
    """
    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        global coord
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        coord = (x1, y1, x2, y2)

    def toggle_selector(event):
        global coord
        if event.key == 'r':
            coordinates.append(coord)
            c = coordinates[-1]
            p = patches.Rectangle((c[0], c[1]), c[2]-c[0], c[3]-c[1],alpha=0.4)
            p1 = patches.Rectangle((c[0], c[1]), c[2]-c[0], c[3]-c[1],facecolor="white")
            current_ax.add_patch(p)
            plt.draw()
            coord = np.asarray(c).astype('int')
            cv2.rectangle(maskout,(coord[0],coord[1]),(coord[2],coord[3]),255,-1)
            centers.append(None)
        if event.key == 'c':
            coordinates.append(coord)
            c = coordinates[-1]
            w = c[2]-c[0]
            h = c[3]-c[1]
            p = patches.Ellipse((c[0]+w/2, c[1]+h/2), w, h, angle=0.0, alpha=0.4)
            p1 = patches.Ellipse((c[0]+w/2, c[1]+h/2), w, h, angle=0.0,facecolor="white")
            current_ax.add_patch(p)
            plt.draw()
            coord = np.asarray(c).astype('int')
            center = ((coord[2] + coord[0]) // 2,(coord[3] + coord[1]) // 2)
            angle = 90
            axes = tuple(sorted(((coord[2] - coord[0]) // 2,(coord[3] - coord[1]) //2)))
            cv2.ellipse(maskout,center,axes,angle,0,360,255,-1)
            centers.append(center)

    coordinates = []
    centers = []
    #visualise frame in full screen
    w, h = pyautogui.size()
    fig, ax_arr = plt.subplots(1,1, figsize=(w/96,h/96))
    fig.suptitle('Select mask')
    current_ax = ax_arr
    current_ax.set_title('Drag on the image, adjust,\n press r, or c to get a rectangular or a circular ROI')
    sns.despine(fig=fig, top=True, right=True, left=True, bottom=True)
    current_ax.imshow(im, cmap = 'gray')
    mask = np.zeros_like(im)
    maskout = np.zeros_like(im,dtype='uint8')
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

    return maskout, centers

def checkROI(video, old_video, usePreviousROI, frame):
    ''' Select ROI '''
    if video.apply_ROI:
        if usePreviousROI and old_video.ROI is not None:
            logger.debug('Getting ROI from previous session')
            mask = old_video.ROI
        else:
            logger.debug('Selecting ROI...')
            mask, _ = getMask(frame)
            if np.count_nonzero(mask) == 0:
                np.ones_like(frame, dtype = np.uint8)*255
    else:
        logger.debug('No ROI selected...')
        mask = np.ones_like(frame, dtype = np.uint8)*255
    return mask

def ROISelectorPreview(video, old_video, usePreviousROI):
    """
    loads a preview of the video for ROI selection
    """
    cap = cv2.VideoCapture(video.video_path)
    ret, frame = cap.read()
    cap.release()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = checkROI(video, old_video, usePreviousROI, frameGray)
    return mask

''' ****************************************************************************
First preview numAnimals, inspect parameters for segmentation and portraying
**************************************************************************** '''
def SegmentationPreview(video):
    global cap, currentSegment
    currentSegment = 0
    cap = cv2.VideoCapture(video.video_path)
    ret, frame = cap.read()
    if frame.shape[2] == 1 or (np.any(frame[:,:,1] == frame[:,:,2] ) and np.any(frame[:,:, 0] == frame[:,:,1])):
        video._number_of_channels = 1
    else:
        raise NotImplementedError("Colour videos has still to be integrated")
    numFrames = video.number_of_frames
    subtract_bkg = video.subtract_bkg

    def thresholder(minTh, maxTh):
        toile = np.zeros_like(frameGray, dtype='uint8')
        segmentedFrame = segmentVideo(avFrame, minTh, maxTh, video.bkg, video.ROI, subtract_bkg)
        #contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        maxArea = cv2.getTrackbarPos('maxArea', 'Bars')
        minArea = cv2.getTrackbarPos('minArea', 'Bars')
        segmentedFrame = ndimage.binary_fill_holes(segmentedFrame).astype('uint8')
        bbs, miniFrames, _, areas, pixels, goodContours, estimated_body_lengths = blobExtractor(segmentedFrame, frameGray, minArea, maxArea)
        cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
        shower = cv2.addWeighted(frameGray,1,toile,.5,0)
        showerCopy = shower.copy()
        resUp = cv2.getTrackbarPos('ResUp', 'Bars')
        resDown = cv2.getTrackbarPos('ResDown', 'Bars')
        showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
        showerCopy = cv2.resize(showerCopy,None, fx = np.true_divide(1,resDown), fy = np.true_divide(1,resDown))
        numColumns = 5
        numGoodContours = len(goodContours)
        numBlackImages = numColumns - numGoodContours % numColumns
        numImages = numGoodContours + numBlackImages
        j = 0
        maximum_body_length = 70
        if estimated_body_lengths:
            maximum_body_length = np.max(estimated_body_lengths)
        imagesMat = []
        rowImage = []

        logger.debug("num blobs detected: %i" %numGoodContours)
        logger.debug("maximum_body_length %i " %maximum_body_length)
        logger.debug("areas: %s" %str(areas))

        identificationImageSize = int(np.sqrt(maximum_body_length ** 2 / 2))
        identificationImageSize = identificationImageSize + identificationImageSize%2  #this is to make the identificationImageSize even

        while j < numImages:
            if j < numGoodContours:
                _ , _, _, image_for_identification = Blob._get_image_for_identification(video.height, video.width, miniFrames[j], pixels[j], bbs[j], identificationImageSize)
            else:
                image_for_identification = np.zeros((identificationImageSize,identificationImageSize),dtype='uint8')
            rowImage.append(image_for_identification)
            if (j+1) % numColumns == 0:
                imagesMat.append(np.hstack(rowImage))
                rowImage = []
            j += 1

        imagesMat = np.vstack(imagesMat)
        cv2.imshow('Bars',np.squeeze(imagesMat))
        cv2.imshow('IdPlayer', showerCopy)
        cv2.moveWindow('Bars', 10,10 )
        cv2.moveWindow('IdPlayer', 200, 10 )

    def scroll(trackbarValue):
        global frame, avFrame, frameGray, cap, currentSegment
        # Select segment dataframe and change cap if needed
        sNumber = video.in_which_episode(trackbarValue)
        sFrame = trackbarValue
        if sNumber != currentSegment: # we are changing segment
            currentSegment = sNumber
            if video.paths_to_video_segments:
                cap = cv2.VideoCapture(video.paths_to_video_segments[sNumber])
        #Get frame from video file
        if video.paths_to_video_segments:
            start = video.episodes_start_end[sNumber][0]
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame - start)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        ret, frame = cap.read()
        if video.resolution_reduction != 1:
            frame = cv2.resize(frame,None, fx = video.resolution_reduction, fy = video.resolution_reduction, interpolation = cv2.INTER_CUBIC)
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avIntensity = np.float32(np.mean(frameGray))
        avFrame = frameGray/avIntensity
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def changeMinTh(minTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def changeMaxTh(maxTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def changeMinArea(x):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def changeMaxArea(maxArea):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def resizeImageUp(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def resizeImageDown(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def resizeImageDown(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    cv2.createTrackbar('start', 'Bars', 0, numFrames-1, scroll )
    cv2.createTrackbar('minTh', 'Bars', 0, 255, changeMinTh)
    cv2.createTrackbar('maxTh', 'Bars', 0, 255, changeMaxTh)
    cv2.createTrackbar('minArea', 'Bars', 0, 5000, changeMinArea)
    cv2.createTrackbar('maxArea', 'Bars', 0, 60000, changeMaxArea)
    cv2.createTrackbar('ResUp', 'Bars', 1, 20, resizeImageUp)
    cv2.createTrackbar('ResDown', 'Bars', 1, 20, resizeImageDown)
    defFrame = 1
    defMinTh = video._min_threshold
    defMaxTh = video._max_threshold
    defMinA = video._min_area
    defMaxA = video._max_area
    defRes = video._resize
    scroll(defFrame)
    cv2.setTrackbarPos('start', 'Bars', defFrame)
    changeMaxArea(defMaxA)
    cv2.setTrackbarPos('maxArea', 'Bars', defMaxA)
    changeMinArea(defMinA)
    cv2.setTrackbarPos('minArea', 'Bars', defMinA)
    changeMinTh(defMinTh)
    cv2.setTrackbarPos('minTh', 'Bars', defMinTh)
    changeMaxTh(defMaxTh)
    cv2.setTrackbarPos('maxTh', 'Bars', defMaxTh)

    if defRes > 0:
        resizeImageUp(defRes)
        cv2.setTrackbarPos('ResUp', 'Bars', defRes)
        resizeImageDown(1)
        cv2.setTrackbarPos('ResDown', 'Bars', 1)
    elif defRes < 0:
        resizeImageUp(1)
        cv2.setTrackbarPos('ResUp', 'Bars', 1)
        resizeImageDown(abs(defRes))
        cv2.setTrackbarPos('ResDown', 'Bars', abs(defRes))

    cv2.waitKey(0)
    #update values in video
    video._min_threshold =  cv2.getTrackbarPos('minTh', 'Bars')
    video._max_threshold = cv2.getTrackbarPos('maxTh', 'Bars')
    video._min_area = cv2.getTrackbarPos('minArea', 'Bars')
    video._max_area = cv2.getTrackbarPos('maxArea', 'Bars')
    video._resize =  - cv2.getTrackbarPos('ResDown', 'Bars') + cv2.getTrackbarPos('ResUp', 'Bars')
    video._has_preprocessing_parameters = True
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def resegmentation_preview(video, frame_number, new_preprocessing_parameters):
    global cap, currentSegment
    currentSegment = 0
    cap = cv2.VideoCapture(video.video_path)
    ret, frame = cap.read()
    numFrames = video.number_of_frames

    def thresholder(minTh, maxTh):
        toile = np.zeros_like(frameGray, dtype='uint8')
        segmentedFrame = segmentVideo(avFrame, minTh, maxTh, video.bkg, video.ROI, video.subtract_bkg)
        maxArea = cv2.getTrackbarPos('maxArea', 'Bars')
        minArea = cv2.getTrackbarPos('minArea', 'Bars')
        segmentedFrame = ndimage.binary_fill_holes(segmentedFrame).astype('uint8')
        bbs, miniFrames, _, areas, pixels, goodContours, estimated_body_lengths = blobExtractor(segmentedFrame, frameGray, minArea, maxArea)
        cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
        shower = cv2.addWeighted(frameGray,1,toile,.5,0)
        showerCopy = shower.copy()
        resUp = cv2.getTrackbarPos('ResUp', 'Bars') if cv2.getTrackbarPos('ResUp', 'Bars') > 0 else 1
        resDown = cv2.getTrackbarPos('ResDown', 'Bars') if cv2.getTrackbarPos('ResDown', 'Bars') > 0 else 1
        showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
        showerCopy = cv2.resize(showerCopy,None, fx = 1/resDown, fy = 1/resDown)
        numColumns = 5
        numGoodContours = len(goodContours)
        numBlackImages = numColumns - numGoodContours % numColumns
        numImages = numGoodContours + numBlackImages
        j = 0
        maximum_body_length = 70
        if estimated_body_lengths:
            maximum_body_length = np.max(estimated_body_lengths)
        imagesMat = []
        rowImage = []

        logger.debug("num blobs detected: %i" %numGoodContours)
        logger.debug("maximum_body_length %i " %maximum_body_length)
        logger.debug("areas: %s" %str(areas))

        identificationImageSize = int(np.sqrt(maximum_body_length ** 2 / 2))
        identificationImageSize = identificationImageSize + identificationImageSize%2  #this is to make the identificationImageSize even

        while j < numImages:
            if j < numGoodContours:
                _ , _, _, image_for_identification = Blob._get_image_for_identification(video.height, video.width, miniFrames[j], pixels[j], bbs[j], identificationImageSize)
            else:
                image_for_identification = np.zeros((identificationImageSize,identificationImageSize),dtype='uint8')
            rowImage.append(image_for_identification)
            if (j+1) % numColumns == 0:
                imagesMat.append(np.hstack(rowImage))
                rowImage = []
            j += 1

        imagesMat = np.vstack(imagesMat)
        cv2.imshow('Bars',np.squeeze(imagesMat))
        cv2.imshow('IdPlayer', showerCopy)
        cv2.moveWindow('Bars', 10,10 )
        cv2.moveWindow('IdPlayer', 200, 10 )

    def visualise_frame(frame_number):
        global frame, avFrame, frameGray, cap, currentSegment
        # Select segment dataframe and change cap if needed
        sNumber = video.in_which_episode(frame_number)
        if sNumber != currentSegment: # we are changing segment
            currentSegment = sNumber
            if video.paths_to_video_segments:
                cap = cv2.VideoCapture(video.paths_to_video_segments[sNumber])
        #Get frame from video file
        if video.paths_to_video_segments:
            start = video.episodes_start_end[sNumber][0]
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_number - start)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if video.resolution_reduction != 1:
            frame = cv2.resize(frame,None, fx = video.resolution_reduction, fy = video.resolution_reduction, interpolation = cv2.INTER_CUBIC)
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avIntensity = np.float32(np.mean(frameGray))
        avFrame = frameGray/avIntensity
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def changeMinTh(minTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def changeMaxTh(maxTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def changeMinArea(x):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def changeMaxArea(maxArea):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def resizeImageUp(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def resizeImageDown(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    def resizeImageDown(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    cv2.createTrackbar('minTh', 'Bars', 0, 255, changeMinTh)
    cv2.createTrackbar('maxTh', 'Bars', 0, 255, changeMaxTh)
    cv2.createTrackbar('minArea', 'Bars', 0, 2000, changeMinArea)
    cv2.createTrackbar('maxArea', 'Bars', 0, 60000, changeMaxArea)
    cv2.createTrackbar('ResUp', 'Bars', 1, 20, resizeImageUp)
    cv2.createTrackbar('ResDown', 'Bars', 1, 20, resizeImageDown)
    defMinTh = new_preprocessing_parameters['min_threshold']
    defMaxTh = new_preprocessing_parameters['max_threshold']
    defMinA = new_preprocessing_parameters['min_area']
    defMaxA = new_preprocessing_parameters['max_area']
    defRes = video.resize
    visualise_frame(frame_number)
    changeMaxArea(defMaxA)
    cv2.setTrackbarPos('maxArea', 'Bars', defMaxA)
    changeMinArea(defMinA)
    cv2.setTrackbarPos('minArea', 'Bars', defMinA)
    changeMinTh(defMinTh)
    cv2.setTrackbarPos('minTh', 'Bars', defMinTh)
    changeMaxTh(defMaxTh)
    cv2.setTrackbarPos('maxTh', 'Bars', defMaxTh)

    if defRes > 0:
        resizeImageUp(defRes)
        cv2.setTrackbarPos('ResUp', 'Bars', defRes)
        resizeImageDown(1)
        cv2.setTrackbarPos('ResDown', 'Bars', 1)
    elif defRes < 0:
        resizeImageUp(1)
        cv2.setTrackbarPos('ResUp', 'Bars', 1)
        resizeImageDown(abs(defRes))
        cv2.setTrackbarPos('ResDown', 'Bars', abs(defRes))
    else:
        cv2.setTrackbarPos('ResDown', 'Bars', 1)
        cv2.setTrackbarPos('ResUp', 'Bars', 1)


    cv2.waitKey(0)
    #update values in video
    new_preprocessing_parameters = {}
    new_preprocessing_parameters['min_threshold'] =  cv2.getTrackbarPos('minTh', 'Bars')
    new_preprocessing_parameters['max_threshold'] = cv2.getTrackbarPos('maxTh', 'Bars')
    new_preprocessing_parameters['min_area'] = cv2.getTrackbarPos('minArea', 'Bars')
    new_preprocessing_parameters['max_area'] = cv2.getTrackbarPos('maxArea', 'Bars')
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return new_preprocessing_parameters

def selectPreprocParams(video, old_video, usePreviousPrecParams):
    restore_segmentation = False
    if not usePreviousPrecParams:
        if old_video and old_video.has_been_segmented:
            restore_segmentation = getInput("Load segmentation", "Load the previous segmentation? Y/n")
            if restore_segmentation == 'y' or restore_segmentation == '':
                restore_segmentation = True
            else:
                restore_segmentation = False

    if not usePreviousPrecParams and not restore_segmentation:
        prepOpts = selectOptions(['bkg', 'ROI', 'resolution_reduction'], None, text = 'Do you want to do BKG or select a ROI or reduce the resolution?', is_processes_list = False)
        video._subtract_bkg = bool(prepOpts['bkg'])
        video._apply_ROI =  bool(prepOpts['ROI'])
        video.reduce_resolution = bool(prepOpts['resolution_reduction'])
        if old_video is not None:
            preprocessing_steps = ['bkg', 'ROI', 'resolution_reduction']
            existentFiles = get_existent_preprocessing_steps(old_video, preprocessing_steps)
            load_previous_preprocessing_steps = selectOptions(preprocessing_steps, existentFiles, text='Restore existing preprocessing steps?', is_processes_list = False)
            if old_video.number_of_animals == None:
                video._number_of_animals = int(getInput('Number of animals','Type the number of animals'))
            else:
                video._number_of_animals = old_video.number_of_animals
            usePreviousROI = bool(load_previous_preprocessing_steps['ROI'])
            usePreviousBkg = bool(load_previous_preprocessing_steps['bkg'])
            usePreviousRR = bool(load_previous_preprocessing_steps['resolution_reduction'])
        else:
            usePreviousROI, usePreviousBkg, usePreviousRR = False, False, False
            video._number_of_animals = int(getInput('Number of animals','Type the number of animals'))
        video._original_ROI = ROISelectorPreview(video, old_video, usePreviousROI)
        video._original_bkg = checkBkg(video, old_video, usePreviousBkg)
        video.resolution_reduction = check_resolution_reduction(video, old_video, usePreviousRR)
        # Preprocessing
        if old_video is not None:
            preprocessing_attributes = ['min_threshold', 'max_threshold',
                                        'min_area','max_area', 'resize',]
            video.copy_attributes_between_two_video_objects(old_video, preprocessing_attributes)
        preprocParams = SegmentationPreview(video)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    elif not usePreviousPrecParams and restore_segmentation:
        preprocessing_attributes = ['apply_ROI','subtract_bkg',
                                    'resolution_reduction',
                                    'maximum_number_of_blobs',
                                    'number_of_channels',
                                    'blobs_path_segmented', 'min_threshold',
                                    'max_threshold',
                                    'min_area','max_area', 'resize',
                                    'number_of_animals', 'original_ROI',
                                    'original_bkg', 'ROI' ,'bkg',
                                    'width', 'height',
                                    'preprocessing_folder']
        video.copy_attributes_between_two_video_objects(old_video, preprocessing_attributes)
        video._has_been_segmented = True
    else:
        preprocessing_attributes = ['apply_ROI','subtract_bkg',
                                    'resolution_reduction',
                                    'maximum_number_of_blobs',
                                    'number_of_channels',
                                    'median_body_length',
                                    'model_area',
                                    'identification_image_size',
                                    'blobs_path_segmented',
                                    'min_threshold','max_threshold',
                                    'min_area','max_area',
                                    'resize', 'number_of_animals',
                                    'original_ROI', 'original_bkg', 'ROI',
                                    'width', 'height',
                                    'bkg', 'preprocessing_folder',
                                    'fragment_identifier_to_index',
                                    'number_of_unique_images_in_global_fragments',
                                    'maximum_number_of_images_in_global_fragments',
                                    'gamma_fit_parameters']
        video.copy_attributes_between_two_video_objects(old_video, preprocessing_attributes)
        video._has_preprocessing_parameters = True
    return restore_segmentation
