from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
import sys
# Import application/library specifics
sys.path.append('../preprocessing')
import numpy as np

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
from segmentation import segmentVideo, blobExtractor
from get_portraits import get_portrait, get_body
# from video_utils import *
from py_utils import get_spaced_colors_util, saveFile, loadFile

"""Change session folder name
"""
def rename_session_folder(video_object, new_session_name):
    assert new_session_name != ''
    new_session_name = 'session_' + new_session_name
    current_session_name = os.path.split(video_object._session_folder)[1]
    print("Modifying folder name: ", current_session_name, " --> ", new_session_name)
    os.rename(video_object._session_folder,
            os.path.join(video_object._video_folder, new_session_name))
    print("Done")

    attributes_to_modify = {key: getattr(video_object, key) for key in video_object.__dict__
    if isinstance(getattr(video_object, key), basestring)
    and current_session_name in getattr(video_object, key) }

    print("Updating video object")

    for key in attributes_to_modify:
        new_value = attributes_to_modify[key].replace(current_session_name, new_session_name)
        setattr(video_object, key, new_value)

    video_object.save()
    print("Done")

"""
Display messages and errors
"""
def selectOptions(optionsList, optionsDict=None, text="Select preprocessing options:  "):
    master = Tk()
    if optionsDict==None:
        optionsDict = {el:'1' for el in optionsList}

    def createCheckBox(name,i):
        var = IntVar()
        Checkbutton(master, text=name, variable=var).grid(row=i+1, sticky=W)
        return var

    Label(master, text=text).grid(row=0, sticky=W)
    variables = []

    for i, opt in enumerate(optionsList):
        if optionsDict[opt] == '1':
            var = createCheckBox(opt,i)
            variables.append(var)
            var.set(optionsDict[opt])
        else:
            Label(master, text= '     ' + opt).grid(row=i+1, sticky=W)
            var = IntVar()
            var.set(0)
            variables.append(var)

    Button(master, text='Ok', command=master.quit).grid(row=i+2, sticky=W, pady=4)
    mainloop()
    varValues = []
    for var in variables:
        varValues.append(var.get())
    optionsDict = dict((key, value) for (key, value) in zip(optionsList, varValues))
    master.destroy()
    return optionsDict

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
            print('\n Getting ROI from previous session')
            mask = old_video.ROI
        else:
            print('\n Selecting ROI ...')
            mask, _ = getMask(frame)
            if np.count_nonzero(mask) == 0:
                np.ones_like(frame, dtype = np.uint8)*255
    else:
        print('\n No ROI selected ...')
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


def checkROI_library(useROI, usePreviousROI, frame, videoPath):
    ''' Select ROI '''
    if useROI:
        if usePreviousROI:
            mask = loadFile(videoPath, 'ROI')
            mask = np.asarray(mask)
            centers= loadFile(videoPath, 'centers')
            centers = np.asarray(centers) ### TODO maybe we need to pass to a list of tuples
        else:
            print('\n Selecting ROI ...')
            mask, centers = getMask(frame)
            # mask = adaptROI(mask)
    else:
        print('\n No ROI selected ...')
        mask = np.ones_like(frame)*255
        centers = []
    return mask, centers

def ROISelectorPreview_library(videoPaths, useROI, usePreviousROI, numSegment=0):
    """
    loads a preview of the video for manual fine-tuning
    """
    cap2 = cv2.VideoCapture(videoPaths[0])
    flag, frame = cap2.read()
    cap2.release()
    height = frame.shape[0]
    width = frame.shape[1]
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask, centers = checkROI_library(useROI, usePreviousROI, frameGray, videoPaths[0])
    saveFile(videoPaths[0], mask, 'ROI')
    saveFile(videoPaths[0], centers, 'centers')
    return width, height, mask, centers

''' ****************************************************************************
First preview numAnimals, inspect parameters for segmentation and portraying
**************************************************************************** '''
def SegmentationPreview(video):
    if video.number_of_animals == None:
        video._number_of_animals = int(getInput('Number of animals','Type the number of animals'))
    if not video.preprocessing_type:
        video.preprocessing_type = getInput('Preprocessing type','What preprocessing do you want to apply? portrait, body or body_blob?')
        #exception for unsupported animal is managed in the class Video
    global cap, currentSegment
    currentSegment = 0
    cap = cv2.VideoCapture(video.video_path)
    numFrames = video._num_frames
    bkg = video.bkg
    mask = video.ROI
    if video.resolution_reduction != 1:
        if bkg is not None:
            bkg = cv2.resize(bkg,None, fx = video.resolution_reduction, fy = video.resolution_reduction, interpolation = cv2.INTER_CUBIC)
            video.bkg = bkg
        mask = cv2.resize(mask,None, fx = video.resolution_reduction, fy = video.resolution_reduction, interpolation = cv2.INTER_CUBIC)
        video.ROI = mask
    subtract_bkg = video.subtract_bkg
    height = video._height
    width = video._width

    def thresholder(minTh, maxTh):
        toile = np.zeros_like(frameGray, dtype='uint8')
        segmentedFrame = segmentVideo(avFrame, minTh, maxTh, bkg, mask, subtract_bkg)
        #contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        maxArea = cv2.getTrackbarPos('maxArea', 'Bars')
        minArea = cv2.getTrackbarPos('minArea', 'Bars')
        segmentedFrame = ndimage.binary_fill_holes(segmentedFrame).astype('uint8')
        bbs, miniFrames, _, areas, pixels, goodContours, estimated_body_lengths = blobExtractor(segmentedFrame, frameGray, minArea, maxArea, height, width)
        cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
        shower = cv2.addWeighted(frameGray,1,toile,.5,0)
        showerCopy = shower.copy()
        # print showerCopy.shape
        resUp = cv2.getTrackbarPos('ResUp', 'Bars')
        resDown = cv2.getTrackbarPos('ResDown', 'Bars')
        showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
        showerCopy = cv2.resize(showerCopy,None, fx = np.true_divide(1,resDown), fy = np.true_divide(1,resDown))
        numColumns = 5
        numGoodContours = len(goodContours)
        numBlackPortraits = numColumns - numGoodContours % numColumns
        numPortraits = numGoodContours + numBlackPortraits
        j = 0
        maximum_body_length = 70
        if estimated_body_lengths:
            maximum_body_length = np.max(estimated_body_lengths)
        portraitsMat = []
        rowPortrait = []

        print("num blobs detected: ", numGoodContours)
        print("maximum_body_length: ", maximum_body_length)
        print("areas: ", areas)

        if video.preprocessing_type == 'portrait':
            portraitSize = int(maximum_body_length/2)
            portraitSize =  portraitSize + portraitSize%2 #this is to make the portraitSize even
        elif video.preprocessing_type == 'body' or  video.preprocessing_type == 'body_blob':
            portraitSize = int(np.sqrt(maximum_body_length ** 2 / 2))
            portraitSize = portraitSize + portraitSize%2  #this is to make the portraitSize even

        while j < numPortraits:
            if j < numGoodContours:
                if video.preprocessing_type == 'portrait':
                    portrait, _, _= get_portrait(miniFrames[j],goodContours[j],bbs[j],portraitSize)
                elif video.preprocessing_type == 'body':
                    portrait, _, _ = get_body(height, width, miniFrames[j], pixels[j], bbs[j], portraitSize)
                elif video.preprocessing_type == 'body_blob':
                    portrait, _, _ = get_body(height, width, miniFrames[j], pixels[j], bbs[j], portraitSize, only_blob = True)
            else:
                portrait = np.zeros((portraitSize,portraitSize),dtype='uint8')
            rowPortrait.append(portrait)
            if (j+1) % numColumns == 0:
                portraitsMat.append(np.hstack(rowPortrait))
                rowPortrait = []
            j += 1

        portraitsMat = np.vstack(portraitsMat)
        cv2.imshow('Bars',np.squeeze(portraitsMat))
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
            if video._paths_to_video_segments:
                cap = cv2.VideoCapture(video._paths_to_video_segments[sNumber])
        #Get frame from video file
        if video._paths_to_video_segments:
            start = video._episodes_start_end[sNumber][0]
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame - start)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        ret, frame = cap.read()
        if video.resolution_reduction != 1:
            frame = cv2.resize(frame,None, fx = video.resolution_reduction, fy = video.resolution_reduction, interpolation = cv2.INTER_CUBIC)
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avIntensity = np.float32(np.mean(frameGray))
        avFrame = np.divide(frameGray,avIntensity)
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

    def resizePortraitDown(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)

    cv2.createTrackbar('start', 'Bars', 0, numFrames-1, scroll )
    cv2.createTrackbar('minTh', 'Bars', 0, 255, changeMinTh)
    cv2.createTrackbar('maxTh', 'Bars', 0, 255, changeMaxTh)
    cv2.createTrackbar('minArea', 'Bars', 0, 1000, changeMinArea)
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

def SegmentationPreview_library(videoPaths, width, height, bkg, mask, useBkg, preprocParams, frameIndices, size = 1):

    ### FIXME Currently the scale factor of the image is not passed everytime we change the segment. It need to be changed so that we do not need to resize everytime we open a new segmen.
    minArea = preprocParams['minArea']
    maxArea = preprocParams['maxArea']
    minThreshold = preprocParams['minThreshold']
    maxThreshold = preprocParams['maxThreshold']
    numAnimals = preprocParams['numAnimals']
    preprocessing_type = preprocParams['preprocessing_type']

    if numAnimals == None:
        numAnimals = getInput('Number of animals','Type the number of animals')
        numAnimals = int(numAnimals)

    if preprocessing_type is None:
        preprocessing_type = getInput('Preprocessing type','What kind of preprocessing do you want? portrait, body or body_blob?')

    if preprocessing_type != 'portrait' and preprocessing_type != 'body' and preprocessing_type != 'body_blob':
        # print(len(preprocessing_type))
        raise ValueError('The animal you selected is not trackable yet :)')

    global cap, currentSegment
    currentSegment = 0
    cap = cv2.VideoCapture(videoPaths[0])
    numFrames = len(frameIndices)

    def thresholder(minTh, maxTh):
        toile = np.zeros_like(frameGray, dtype='uint8')
        segmentedFrame = segmentVideo(avFrame, minTh, maxTh, bkg, mask, useBkg)
        #contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        maxArea = cv2.getTrackbarPos('maxArea', 'Bars')
        minArea = cv2.getTrackbarPos('minArea', 'Bars')
        bbs, miniFrames, _, areas, pixels, goodContours, estimated_body_lengths = blobExtractor(segmentedFrame, frameGray, minArea, maxArea, height, width)
        cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
        shower = cv2.addWeighted(frameGray,1,toile,.5,0)
        showerCopy = shower.copy()
        # print showerCopy.shape
        resUp = cv2.getTrackbarPos('ResUp', 'Bars')
        resDown = cv2.getTrackbarPos('ResDown', 'Bars')
        showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
        showerCopy = cv2.resize(showerCopy,None, fx = np.true_divide(1,resDown), fy = np.true_divide(1,resDown))
        numColumns = 5
        numGoodContours = len(goodContours)
        numBlackPortraits = numColumns - numGoodContours % numColumns
        numPortraits = numGoodContours + numBlackPortraits
        j = 0
        maximum_body_length = 70
        if estimated_body_lengths:
            maximum_body_length = np.max(estimated_body_lengths)
        portraitsMat = []
        rowPortrait = []

        print("num blobs detected: ", numGoodContours)
        print("maximum_body_length: ", maximum_body_length)
        print("areas: ", areas)

        if preprocessing_type == 'portrait':
            portraitSize = int(maximum_body_length/2)
            portraitSize =  portraitSize + portraitSize%2 #this is to make the portraitSize even
        elif preprocessing_type == 'body' or preprocessing_type == 'body_blob':
            portraitSize = int(np.sqrt(maximum_body_length ** 2 / 2))
            portraitSize = portraitSize + portraitSize%2  #this is to make the portraitSize even

        while j < numPortraits:
            if j < numGoodContours:
                if preprocessing_type == 'portrait':
                    portrait,_,_= get_portrait(miniFrames[j],goodContours[j],bbs[j],portraitSize)
                elif preprocessing_type == 'body':
                    portrait, _, _ = get_body(height, width, miniFrames[j], pixels[j], bbs[j], portraitSize, only_blob = False)
                elif preprocessing_type == 'body_blob':
                    portrait, _, _ = get_body(height, width, miniFrames[j], pixels[j], bbs[j], portraitSize, only_blob = True)
            else:
                portrait = np.zeros((portraitSize,portraitSize),dtype='uint8')
            rowPortrait.append(portrait)
            if (j+1) % numColumns == 0:
                portraitsMat.append(np.hstack(rowPortrait))
                rowPortrait = []
            j += 1

        portraitsMat = np.vstack(portraitsMat)
        cv2.imshow('Bars',np.squeeze(portraitsMat))
        cv2.imshow('IdPlayer', showerCopy)
        cv2.moveWindow('Bars', 10,10 )
        cv2.moveWindow('IdPlayer', 200, 10 )

    def scroll(trackbarValue):
        global frame, avFrame, frameGray, cap, currentSegment

        # Select segment dataframe and change cap if needed
        sNumber = frameIndices.loc[trackbarValue,'segment']
        sFrame = frameIndices.loc[trackbarValue,'frame']

        if sNumber != currentSegment: # we are changing segment
            print('Changing segment...')
            currentSegment = sNumber

            if len(videoPaths) > 1:
                cap = cv2.VideoCapture(videoPaths[sNumber-1])

        #Get frame from video file
        if len(videoPaths) > 1:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        ret, frame = cap.read()

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avIntensity = np.float32(np.mean(frameGray))
        avFrame = np.divide(frameGray,avIntensity)
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass


    def changeMinTh(minTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def changeMaxTh(maxTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def changeMinArea(x):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def changeMaxArea(maxArea):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def resizeImageUp(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def resizeImageDown(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    cv2.createTrackbar('start', 'Bars', 0, numFrames-1, scroll )
    cv2.createTrackbar('minTh', 'Bars', 0, 255, changeMinTh)
    cv2.createTrackbar('maxTh', 'Bars', 0, 255, changeMaxTh)
    cv2.createTrackbar('minArea', 'Bars', 0, 1000, changeMinArea)
    cv2.createTrackbar('maxArea', 'Bars', 0, 60000, changeMaxArea)
    cv2.createTrackbar('ResUp', 'Bars', 1, 20, resizeImageUp)
    cv2.createTrackbar('ResDown', 'Bars', 1, 20, resizeImageDown)

    defFrame = 1
    defMinTh = minThreshold
    defMaxTh = maxThreshold
    defMinA = minArea
    defMaxA = maxArea
    defRes = size

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
    resizeImageUp(defRes)
    cv2.setTrackbarPos('ResUp', 'Bars', defRes)
    resizeImageDown(defRes)
    cv2.setTrackbarPos('ResDown', 'Bars', defRes)

    #start = cv2.getTrackbarPos('start','Bars')
    #minThresholdStart = cv2.getTrackbarPos('minTh', 'Bars')
    #minAreaStart = cv2.getTrackbarPos('minArea', 'Bars')
    #maxAreaStart = cv2.getTrackbarPos('maxArea', 'Bars')

    cv2.waitKey(0)

    preprocParams = {
                'minThreshold': cv2.getTrackbarPos('minTh', 'Bars'),
                'maxThreshold': cv2.getTrackbarPos('maxTh', 'Bars'),
                'minArea': cv2.getTrackbarPos('minArea', 'Bars'),
                'maxArea': cv2.getTrackbarPos('maxArea', 'Bars'),
                'numAnimals': numAnimals,
                'preprocessing_type':preprocessing_type}

    cap.release()
    cv2.destroyAllWindows()

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return preprocParams

def selectPreprocParams(video, old_video, usePreviousPrecParams):
    if not usePreviousPrecParams:
        video._min_threshold = 0
        video._max_threshold = 155
        video._min_area = 150
        video._max_area = 60000
        video._resize = 1
        preprocParams = SegmentationPreview(video)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    else:
        video._min_threshold = old_video._min_threshold
        video._max_threshold = old_video._max_threshold
        video._min_area = old_video._min_area
        video._max_area = old_video._max_area
        video._resize = old_video._resize
        if hasattr(old_video, 'resolution_reduction'):
            video.resolution_reduction = old_video.resolution_reduction
        video.preprocessing_type = old_video.preprocessing_type
        video._number_of_animals = old_video._number_of_animals
        video._has_preprocessing_parameters = True

def selectPreprocParams_library(videoPaths, usePreviousPrecParams, width, height, bkg, mask, useBkg, frameIndices):
    if not usePreviousPrecParams:
        videoPath = videoPaths[0]
        preprocParams = {
                    'minThreshold': 0,
                    'maxThreshold': 155,
                    'minArea': 150,
                    'maxArea': 60000,
                    'numAnimals': None,
                    'preprocessing_type': None
                    }
        preprocParams = SegmentationPreview_library(videoPaths, width, height, bkg, mask, useBkg, preprocParams, frameIndices)

        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        saveFile(videoPaths[0], preprocParams, 'preprocparams',hdfpkl='pkl')
    else:
        preprocParams = loadFile(videoPaths[0], 'preprocparams',hdfpkl='pkl')
    return preprocParams
''' ****************************************************************************
Fragmentation inspector
*****************************************************************************'''
def fragmentation_inspector(video, blobs_in_video):
    """inputs:
    video: object containing video info and paths
    blobs_in_video: list of blob objects organised frame-wise:
                    [[blob_1_in_frame_1, ..., blob_i_frame_1], .... ,
                     [blob_1_in_frame_m, ..., blob_j_frame_m]]
    Given a frame it loops on the fragment of each blob and labels all the blobs
    belonging to the same fragment with a unique identifier.
    """
    cap = cv2.VideoCapture(video.video_path)
    numFrames = video._num_frames
    bkg = video.bkg
    mask = video.ROI
    subtract_bkg = video.subtract_bkg
    height = video._height
    width = video._width
    global currentSegment, cap, frame
    currentSegment = 0
    cv2.namedWindow('fragmentInspection')
    defFrame = 1


    def resizer(sizeValue):
        global frame
        print("sizeValue, ", sizeValue)
        real_size = sizeValue - 5
        print("real_size, ", real_size)
        if real_size > 0:
            print("I should enlarge")
            frame = cv2.resize(frame,None,fx = real_size+1, fy = real_size+1)
            print(frame.shape)
        elif real_size < 0:
            print("I should reduce")
            frame = cv2.resize(frame,None, fx = np.true_divide(1,abs(real_size)+1), fy = np.true_divide(1,abs(real_size)+1))
            print(frame.shape)
        # elif real_size == 0:
        #     print("real_size is zero")
        #     real_size = sizeValue - previousSize
        #     print("real_size, ", real_size)
        #     if real_size > 0:
        #         print("I should enlarge")
        #         frame = cv2.resize(frame,None,fx = real_size+1, fy = real_size+1)
        #         previousSize = sizeValue
        #     elif real_size < 0:
        #         print("I should reduce")
        #         frame = cv2.resize(frame,None, fx = np.true_divide(1,abs(real_size)+1), fy = np.true_divide(1,abs(real_size)+1))
        #         previousSize = sizeValue

        cv2.imshow('fragmentInspection', frame)

    def scroll(trackbarValue):
        global frame, currentSegment, cap
        # Select segment dataframe and change cap if needed
        sNumber = video.in_which_episode(trackbarValue)
        print('seg number ', sNumber)
        print('trackbarValue ', trackbarValue)
        sFrame = trackbarValue

        if sNumber != currentSegment: # we are changing segment
            print('Changing segment...')
            currentSegment = sNumber
            if video._paths_to_video_segments:
                cap = cv2.VideoCapture(video._paths_to_video_segments[sNumber])
        #Get frame from video file
        if video._paths_to_video_segments:
            start = video._episodes_start_end[sNumber][0]
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame - start)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        ret, frame = cap.read()
        blobs_in_frame = blobs_in_video[trackbarValue]

        for blob in blobs_in_frame:
            #draw the centroid
            cv2.circle(frame, tuple(blob.centroid.astype('int')), 2, (255,0,0),1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(blob.fragment_identifier),tuple(blob.centroid.astype('int')), font, 1,255, 5)

        sizeValue = cv2.getTrackbarPos('frameSize', 'Bars')
        resizer(sizeValue)
        cv2.imshow('fragmentInspection', frame)

    cv2.createTrackbar('start', 'Bars', 0, numFrames-1, scroll )
    cv2.createTrackbar('frameSize', 'Bars', 1, 9, resizer)
    cv2.setTrackbarPos('start', 'Bars', defFrame)
    cv2.setTrackbarPos('frameSize', 'Bars', 5)
    scroll(1)
    cv2.waitKey(0)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def playFragmentation_library(videoPaths,segmPaths,dfGlobal,visualize = False):
    from fragmentation import computeFrameIntersection ### FIXME For some reason it does not import well in the top and I have to import it here
    """
    IdInspector
    """
    if visualize:
        info = loadFile(videoPaths[0], 'videoInfo', hdfpkl = 'pkl')
        width = info['width']
        height = info['height']
        numAnimals = info['numAnimals']
        maxNumBlobs = info['maxNumBlobs']
        numSegment = 0
        frameIndices = loadFile(videoPaths[0], 'frameIndices')
        path = videoPaths[numSegment]

        def IdPlayerFragmentation(videoPaths,segmPaths,numAnimals, width, height,frameIndices):

            global segmDf, cap, currentSegment
            segmDf,sNumber = loadFile(segmPaths[0], 'segmentation')
            currentSegment = int(sNumber)
            print('Visualizing video %s' % path)
            cap = cv2.VideoCapture(videoPaths[0])
            numFrames = len(frameIndices)
            # numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

            def onChange(trackbarValue):
                global segmDf, cap, currentSegment

                # Select segment dataframe and change cap if needed
                sNumber = frameIndices.loc[trackbarValue,'segment']
                sFrame = frameIndices.loc[trackbarValue,'frame']

                if sNumber != currentSegment: # we are changing segment
                    print('Changing segment...')
                    prevSegmDf, _ = loadFile(segmPaths[sNumber-2], 'segmentation')
                    segmDf, _ = loadFile(segmPaths[sNumber-1], 'segmentation')
                    currentSegment = sNumber

                    if len(videoPaths) > 1:
                        cap = cv2.VideoCapture(videoPaths[sNumber-1])

                #Get frame from video file
                if len(videoPaths) > 1:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame)
                else:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
                ret, frame = cap.read()
                font = cv2.FONT_HERSHEY_SIMPLEX
                frameCopy = frame.copy()

                # print('**********************************')
                # print('sNumber, ', sNumber)
                # print('sFrame, ', sFrame)
                # print('trackbarValue, ', trackbarValue)
                #
                # permutation = dfGlobal.loc[trackbarValue,'permutations']
                # centroids = dfGlobal.loc[trackbarValue,'centroids']
                # pixelsA = segmDf.loc[sFrame-1,'pixels']
                # pixelsB = segmDf.loc[sFrame,'pixels']
                # print('------------------------------------------------------------')
                # print('previous frame, ', str(trackbarValue-1), ', permutation, ', dfGlobal.loc[trackbarValue-1,'permutations'])
                # print('current frame, ', str(trackbarValue), ', permutation, ', permutation)
                # trueFragment, s, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
                # print('overlapMat, ')
                # print(overlapMat)
                # print('permutation, ', s)

                #Color to gray scale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Plot segmentated blobs
                for i, pixel in enumerate(pixelsB):
                    px = np.unravel_index(pixel,(height,width))
                    frame[px[0],px[1]] = 255

                for i, centroid in enumerate(centroids):
                    cv2.putText(frame,'i'+ str(permutation[i]) + '|h' +str(i),centroid, font, .7,0)

                cv2.putText(frame,str(trackbarValue),(50,50), font, 3,(255,0,0))

                # Visualization of the process
                cv2.imshow('IdPlayerFragmentation',frame)
                pass

            cv2.namedWindow('IdPlayerFragmentation')
            cv2.createTrackbar( 'start', 'IdPlayerFragmentation', 0, numFrames-1, onChange )

            onChange(1)
            cv2.waitKey()

        IdPlayerFragmentation(videoPaths,segmPaths,numAnimals, width, height,frameIndices)

''' ****************************************************************************
Frame by frame identification inspector
*****************************************************************************'''
def get_n_previous_blobs_attribute(blob,attribute_name,number_of_previous):
    blobs_attrs = []
    current_blob = blob
    for i in range(number_of_previous):
        if current_blob.is_a_fish_in_a_fragment:
            blobs_attrs.append(getattr(current_blob,attribute_name))
            current_blob = current_blob.previous[0]
        else:
            break
    return blobs_attrs

def frame_by_frame_identity_inspector(video, blobs_in_video, number_of_previous = 10, save_video = False):
    cap = cv2.VideoCapture(video.video_path)
    numFrames = video._num_frames
    bkg = video.bkg
    mask = video.ROI
    subtract_bkg = video.subtract_bkg
    height = video._height
    width = video._width
    global currentSegment, cap
    currentSegment = 0
    cv2.namedWindow('frame_by_frame_identity_inspector')
    defFrame = 0
    colors = get_spaced_colors_util(video.number_of_animals,black=True)

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    name = video._session_folder +'/tracked.avi'
    out = cv2.VideoWriter(name, fourcc, 32.0, (video._width, video._height))

    def scroll(trackbarValue):
        global frame, currentSegment, cap

        # Select segment dataframe and change cap if needed
        sNumber = video.in_which_episode(trackbarValue)
        print('seg number ', sNumber)
        print('trackbarValue ', trackbarValue)
        sFrame = trackbarValue

        if sNumber != currentSegment: # we are changing segment
            print('Changing segment...')
            currentSegment = sNumber
            if video._paths_to_video_segments:
                cap = cv2.VideoCapture(video._paths_to_video_segments[sNumber])

        #Get frame from video file
        if video._paths_to_video_segments:
            start = video._episodes_start_end[sNumber][0]
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame - start)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        ret, frame = cap.read()
        if ret:
            if hasattr(video, 'resolution_reduction'):
                if video.resolution_reduction != 1:
                    frame = cv2.resize(frame, None, fx = video.resolution_reduction, fy = video.resolution_reduction)

            frameCopy = frame.copy()
            blobs_in_frame = blobs_in_video[trackbarValue]

            for b, blob in enumerate(blobs_in_frame):
                blobs_pixels = get_n_previous_blobs_attribute(blob,'pixels',number_of_previous)[::-1]
                blobs_identities = get_n_previous_blobs_attribute(blob,'identity',number_of_previous)[::-1]

                for i, (blob_pixels, blob_identity) in enumerate(zip(blobs_pixels,blobs_identities)):
                    # print(i)
                    pxs = np.unravel_index(blob_pixels,(video._height,video._width))
                    if i < number_of_previous-1:
                        # print(type(blob_identity))
                        if type(blob_identity) is not list and blob_identity is not None and blob_identity != 0:
                            # print('adding shadows to fish')
                            frame[pxs[0], pxs[1], :] = np.multiply(colors[blob_identity], .3).astype('uint8')+np.multiply(frame[pxs[0], pxs[1], :], .7).astype('uint8')
                        elif type(blob_identity) is list or blob_identity is None or blob_identity == 0:
                            # print('adding shadows to crossing')
                            frame[pxs[0], pxs[1], :] = np.multiply([0, 0, 0], .3).astype('uint8')+np.multiply(frame[pxs[0], pxs[1], :], .7).astype('uint8')
                    else:
                        frame[pxs[0], pxs[1], :] = frameCopy[pxs[0], pxs[1], :]

                #draw the centroid
                font = cv2.FONT_HERSHEY_SIMPLEX
                if type(blob.identity) is int:
                    cv2.circle(frame, tuple(blob.centroid.astype('int')), 2, colors[blob._identity], -1)
                elif type(blob.identity) is list:
                    cv2.circle(frame, tuple(blob.centroid.astype('int')), 2, [255, 255, 255], -1)
                if blob._assigned_during_accumulation:
                    cv2.putText(frame, str(blob.identity),tuple(blob.centroid.astype('int')), font, 1, colors[blob.identity], 3)
                else:
                    print("the current blob is a fish ", blob.is_a_fish)
                    print("id ", type(blob.identity) is int)
                    if blob.is_a_fish and type(blob.identity) is int:
                        cv2.putText(frame, str(blob.identity), tuple(blob.centroid.astype('int')), font, .5, colors[blob.identity], 3)
                    elif not blob.is_a_fish:
                        cv2.putText(frame, str(blob.identity), tuple(blob.centroid.astype('int')), font, 1, [255,255,255], 3)
                    else:
                        cv2.putText(frame, str(blob.identity), tuple(blob.centroid.astype('int')), font, .5, colors[blob.identity], 3)

                if not save_video:
                    print("\nblob ", b)
                    print("identity: ", blob._identity)
                    if hasattr(blob,"identities_before_crossing"):
                        print("identity_before_crossing: ", blob.identities_before_crossing)
                    if hasattr(blob,"identities_after_crossing"):
                        print("identity_after_crossing: ", blob.identities_after_crossing)
                    print("assigned during accumulation: ", blob.assigned_during_accumulation)
                    if not blob.assigned_during_accumulation and blob.is_a_fish_in_a_fragment:
                        try:
                            print("frequencies in fragment: ", blob.frequencies_in_fragment)
                        except:
                            print("this blob does not have frequencies in fragment")
                    print("P1_vector: ", blob.P1_vector)
                    print("P2_vector: ", blob.P2_vector)
                    print("is_a_fish: ", blob.is_a_fish)
                    print("is_in_a_fragment: ", blob.is_in_a_fragment)
                    print("is_a_fish_in_a_fragment: ", blob.is_a_fish_in_a_fragment)
                    print("is_a_jump: ", blob.is_a_jump)
                    print("is_a_ghost_crossing: ", blob.is_a_ghost_crossing)
                    print("is_a_crossing: ", blob.is_a_crossing)
                    # if blob.is_a_crossing:
                    #     print("bad_crossing ", blob.bad_crossing)
                        # print("number_of_animals_in_crossing: ", blob.number_of_animals_in_crossing)
                    print("next: ", blob.next)
                    print("previous: ", blob.previous)


            if not save_video:
                # frame = cv2.resize(frame,None, fx = np.true_divide(1,4), fy = np.true_divide(1,4))
                cv2.imshow('frame_by_frame_identity_inspector', frame)
                pass
            else:
                out.write(frame)
        else:
            print("Warning: unable to read frame ", scroll)
    cv2.createTrackbar('start', 'frame_by_frame_identity_inspector', 0, numFrames-1, scroll )

    scroll(1)
    cv2.setTrackbarPos('start', 'frame_by_frame_identity_inspector', defFrame)

    # cv2.waitKey(0)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    if save_video:
        for i in tqdm(range(video._num_frames)):
            scroll(i)

    cv2.waitKey(0)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    save_video = getInput('Saver' , 'Do you want to save a copy of the tracked video? [y]/n')
    if not save_video or save_video == 'y':
        frame_by_frame_identity_inspector(video, blobs_in_video, save_video = True)
    else:
        return

def frame_by_frame_identity_inspector_for_Liad(video, blobs_in_video, number_of_previous = 100, save_video = False):
    cap = cv2.VideoCapture(video.video_path)
    numFrames = video._num_frames
    bkg = video.bkg
    mask = video.ROI
    subtract_bkg = video.subtract_bkg
    height = video._height
    width = video._width
    global currentSegment, cap
    currentSegment = 0
    cv2.namedWindow('frame_by_frame_identity_inspector')
    defFrame = 0
    colors = get_spaced_colors_util(video.number_of_animals,black=True)

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    if save_video:
        name = video._session_folder +'/tracked.avi'
        out = cv2.VideoWriter(name, fourcc, 32.0, (video._width, video._height))

    def scroll(trackbarValue):
        global frame, currentSegment, cap

        # Select segment dataframe and change cap if needed
        sNumber = video.in_which_episode(trackbarValue)
        print('seg number ', sNumber)
        print('trackbarValue ', trackbarValue)
        sFrame = trackbarValue

        if sNumber != currentSegment: # we are changing segment
            print('Changing segment...')
            currentSegment = sNumber
            if video._paths_to_video_segments:
                cap = cv2.VideoCapture(video._paths_to_video_segments[sNumber])

        #Get frame from video file
        if video._paths_to_video_segments:
            start = video._episodes_start_end[sNumber][0]
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame - start)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        ret, frame = cap.read()
        if ret:
            # frameCopy = frame.copy()
            frame = np.zeros_like(frame)
            blobs_in_frame = blobs_in_video[trackbarValue]

            for b, blob in enumerate(blobs_in_frame):
                blobs_pixels = get_n_previous_blobs_attribute(blob,'pixels',number_of_previous)[::-1]
                blobs_identities = get_n_previous_blobs_attribute(blob,'identity',number_of_previous)[::-1]

                for i, (blob_pixels, blob_identity) in enumerate(zip(blobs_pixels,blobs_identities)):
                    # print(i)
                    pxs = np.unravel_index(blob_pixels,(video._height,video._width))
                    if i < number_of_previous-1:
                        # print(type(blob_identity))
                        if type(blob_identity) is not list and blob_identity is not None and blob_identity != 0:
                            # print('adding shadows to fish')
                            frame[pxs[0], pxs[1], :] = np.multiply(colors[blob_identity], .3).astype('uint8')+np.multiply(frame[pxs[0], pxs[1], :], .7).astype('uint8')
                        elif type(blob_identity) is list or blob_identity is None or blob_identity == 0:
                            # print('adding shadows to crossing')
                            frame[pxs[0], pxs[1], :] = np.multiply([0, 0, 0], .3).astype('uint8')+np.multiply(frame[pxs[0], pxs[1], :], .7).astype('uint8')
                    # else:
                    #     frame[pxs[0], pxs[1], :] = frameCopy[pxs[0], pxs[1], :]

                #draw the centroid
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # if type(blob.identity) is int:
                #     cv2.circle(frame, tuple(blob.centroid.astype('int')), 2, colors[blob._identity], -1)
                # elif type(blob.identity) is list:
                #     cv2.circle(frame, tuple(blob.centroid.astype('int')), 2, [255, 255, 255], -1)
                # if blob._assigned_during_accumulation:
                #     cv2.putText(frame, str(blob.identity),tuple(blob.centroid.astype('int')), font, 1, colors[blob.identity], 3)
                # else:
                #     print("the current blob is a fish ", blob.is_a_fish)
                #     print("id ", type(blob.identity) is int)
                #     if blob.is_a_fish and type(blob.identity) is int:
                #         cv2.putText(frame, str(blob.identity), tuple(blob.centroid.astype('int')), font, .5, colors[blob.identity], 3)
                #     elif not blob.is_a_fish:
                #         cv2.putText(frame, str(blob.identity), tuple(blob.centroid.astype('int')), font, 1, [255,255,255], 3)
                #     else:
                #         cv2.putText(frame, str(blob.identity), tuple(blob.centroid.astype('int')), font, .5, colors[blob.identity], 3)

                # if not save_video:
                    # print("\nblob ", b)
                    # print("identity: ", blob._identity)
                    # if hasattr(blob,"identities_before_crossing"):
                    #     print("identity_before_crossing: ", blob.identities_before_crossing)
                    # if hasattr(blob,"identities_after_crossing"):
                    #     print("identity_after_crossing: ", blob.identities_after_crossing)
                    # print("assigned during accumulation: ", blob.assigned_during_accumulation)
                    # if not blob.assigned_during_accumulation and blob.is_a_fish_in_a_fragment:
                    #     try:
                    #         print("frequencies in fragment: ", blob.frequencies_in_fragment)
                    #     except:
                    #         print("this blob does not have frequencies in fragment")
                    # print("P1_vector: ", blob.P1_vector)
                    # print("P2_vector: ", blob.P2_vector)
                    # print("is_a_fish: ", blob.is_a_fish)
                    # print("is_in_a_fragment: ", blob.is_in_a_fragment)
                    # print("is_a_fish_in_a_fragment: ", blob.is_a_fish_in_a_fragment)
                    # print("is_a_jump: ", blob.is_a_jump)
                    # print("is_a_ghost_crossing: ", blob.is_a_ghost_crossing)
                    # print("is_a_crossing: ", blob.is_a_crossing)
                    # # if blob.is_a_crossing:
                    # #     print("bad_crossing ", blob.bad_crossing)
                    #     # print("number_of_animals_in_crossing: ", blob.number_of_animals_in_crossing)
                    # print("next: ", blob.next)
                    # print("previous: ", blob.previous)


            if not save_video:
                # frame = cv2.resize(frame,None, fx = np.true_divide(1,4), fy = np.true_divide(1,4))
                cv2.imshow('frame_by_frame_identity_inspector', frame)
                pass
            else:
                out.write(frame)
        else:
            print("Warning: unable to read frame ", scroll)
    cv2.createTrackbar('start', 'frame_by_frame_identity_inspector', 0, numFrames-1, scroll )

    scroll(1)
    cv2.setTrackbarPos('start', 'frame_by_frame_identity_inspector', defFrame)

    # cv2.waitKey(0)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    if save_video:
        for i in tqdm(range(video._num_frames)):
            scroll(i)

    cv2.waitKey(0)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    save_video = getInput('Saver' , 'Do you want to save a copy of the tracked video? [y]/n')
    if not save_video or save_video == 'y':
        frame_by_frame_identity_inspector_for_Liad(video, blobs_in_video, save_video = True)
    else:
        return
