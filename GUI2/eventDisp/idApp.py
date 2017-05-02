from __future__ import division

import kivy

from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.event import EventDispatcher
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.scatter import Scatter
from kivy.uix.popup import Popup
from kivy.uix.switch import Switch
from kivy.uix.textinput import TextInput
from kivy.graphics import *
from kivy.graphics.transformation import Matrix
from kivy.clock import Clock
from kivy.config import Config #used before running the app to set the keyboard usage
from kivy.event import EventDispatcher

import os
import cPickle as pickle
import cv2
import numpy as np
import glob
import re
import sys
sys.path.append('../../utils')
import time

from py_utils import flatten, saveFile, loadFile, get_spaced_colors_util
from video_utils import getSegmPaths, computeBkg, segmentVideo, blobExtractor


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def scanFolder(path):
    ### NOTE if the video selected does not finish with '_1' the scanFolder function won't select all of them. This can be improved
    paths = [path]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    # maybe write check on video extension supported by opencv2
    if filename[-2:] == '_1':
        paths = natural_sort(glob.glob(folder + "/" + filename[:-1] + "*" + extension))
    return paths

def getLastSession(subFolders):
    if len(subFolders) == 0:
        lastIndex = 0
    else:
        subFolders = natural_sort(subFolders)[::-1]
        lastIndex = int(subFolders[0].split('_')[-1])
    return lastIndex


"""
Start kivy classes
"""
class Chosen_Video(EventDispatcher):
    chosen = StringProperty('')

    def __init__(self,**kwargs):
        super(Chosen_Video,self).__init__(**kwargs)
        self.chosen = 'Default String'
        self.bind(chosen=self.on_modified)
        self.flag_prep = True #used to control the desable property of Button()
        self.flag_bkg = True

    def set_chosen_item(self, chosen_string):
        self.chosen = chosen_string

    def get_chosen_item(self):
        return self.chosen

    def on_modified(self, instance, value):
        print("Chosen item in ", instance, " was modified to :",value)
        self.get_bkg_path()
        self.get_preproc_path()

    def get_bkg_path(self, *args):
        """
        If the chose path is a video checks for the bkg file and the preprocessing
        folder. If the first is there returns a flag, if the folder does not
        exist it creates it.
        """
        if '.avi' in self.chosen:
            #get video folder
            self.video_folder = os.path.dirname(self.chosen)
            #get path to the candidate preprocessing folder
            preproc_folder = self.video_folder + '/preprocessing'
            #and path to the candidate bkg pickle
            self.bkg_path = self.video_folder + '/preprocessing/bkg.pkl'

            if os.path.isfile(self.bkg_path):
                self.flag_bkg = False
            else:
                if not os.path.isdir(preproc_folder):
                    os.makedirs(self.preproc_folder)
        return self.flag_bkg

    def get_preproc_path(self):
        """
        If the chosen path is a video will check for the preprocessing folder.
        If the preprocesing folder exists, will look for the file in which the
        preprocessing parameters are collected and output False (to enable a
        button by setting btn.disabled = False). If the file does not existe will
        output True. If the preprocessing folder does not exist, it creates it.
        """
        if '.avi' in self.chosen:
            #get video folder
            self.video_folder = os.path.dirname(self.chosen)
            #get path to the candidate preprocessing folder
            preproc_folder = self.video_folder + '/preprocessing'
            #and path to the candidate preprocessing params pickle
            self.preproc_path = self.video_folder + '/preprocessing/preprocparams.pkl'

            if os.path.isfile(self.preproc_path):
                self.flag_prep = False
            else:
                if not os.path.isdir(preproc_folder):
                    os.makedirs(self.preproc_folder)
            print self.flag_prep
        return self.flag_prep


class SelectFile(BoxLayout):
    def __init__(self,**kwargs):
        super(SelectFile,self).__init__(**kwargs)
        self.update ='You did not select a video yet'

    global CHOSEN_VIDEO
    CHOSEN_VIDEO = Chosen_Video()

    def save_chosen(self):
      CHOSEN_VIDEO.set_chosen_item(chosen_item)

    def open(self, path, filename):
        videoFlag = True
        if filename:
            if '.avi' not in filename[0]:
                videoFlag = True
            else:
                print 'you selected something'
                self.update = filename[0]
                print self.update
                CHOSEN_VIDEO.set_chosen_item(self.update)
                videoFlag = False
                print CHOSEN_VIDEO.get_chosen_item()

        else:
            if self.parent.parent is not None:
                self.update ='You did not select a video yet'
                CHOSEN_VIDEO.set_chosen_item(self.update)
                print CHOSEN_VIDEO.get_chosen_item()
                print self.update
                videoFlag = True
            else:
                videoFlag = True
        return videoFlag

    def checkROI(self, pathToVideo):
        flagROI = True
        if pathToVideo != '' and pathToVideo != 'You did not select a video yet':
            self.videoPath = pathToVideo
            self.parent.videoPaths = scanFolder(self.videoPath)
            self.videoFolder = os.path.dirname(self.videoPath)
            preprocDir = self.videoFolder + '/preprocessing'
            ROIpath = self.videoFolder + '/preprocessing/ROI.hdf5'

            if os.path.isfile(ROIpath):
                self.ROIPath = ROIpath
                flagROI =  False
            elif not os.path.isdir(preprocDir):
                os.makedirs(preprocDir)

            self.parent.ROIPath = ROIpath

        return flagROI

    def checkPreprocessing(self, pathToVideo):
        flagPrep = True

        if pathToVideo != '' and pathToVideo != 'You did not select a video yet':
            self.videoPath = pathToVideo
            self.videoFolder = os.path.dirname(self.videoPath)
            preprocDir = self.videoFolder + '/preprocessing'
            preprocPath = self.videoFolder + '/preprocessing/preprocparams.pkl'

            if os.path.isfile(preprocPath):
                self.preprocPath = preprocPath
                self.parent.p_params = pickle.load(open(preprocPath, 'rb'))

                flagPrep = False
            else:
                self.parent.p_params = None
                if not os.path.isdir(preprocDir):
                    os.makedirs(preprocDir)
            #in any case we store the preprocessing path
            self.parent.preprocPath = preprocPath
            print 'here is the preprocessing path ', self.parent.preprocPath
            self.parent.flagPrep = flagPrep

        return flagPrep

    def checkSegmentation(self, pathToVideo):
        flagSeg = True

        if pathToVideo != '' and pathToVideo != 'You did not select a video yet':
            self.videoPath = pathToVideo
            self.videoFolder = os.path.dirname(self.videoPath)
            self.segDir = self.videoFolder + '/preprocessing/segmentation'

            if os.path.isdir(self.segDir) and len(glob.glob(self.segDir + '/*.hdf5')) > 0:
                print 'there is segmentation'
                flagSeg = False
            #in any case we store the path to the segmentation  folder
            elif not os.path.makedirs(self.segDir):
                os.makedirs(self.segDir)

            self.parent.segDir = self.segDir

        return flagSeg

    def checkSession(self, pathToVideo):
        sessFlag = True
        if pathToVideo != '' and pathToVideo != 'You did not select a video yet':
            print pathToVideo
            self.videoPath = pathToVideo
            self.videoFolder = os.path.dirname(self.videoPath)
            self.modelsDir = self.videoFolder + '/CNN_models'

            if os.path.isdir(self.modelsDir):
                sessions = natural_sort(glob.glob(self.modelsDir + '/Session_*'))
                self.lastSession = sessions[-1]
                if os.path.isfile(self.lastSession + '/statistics.pkl'):
                    sessFlag = False
            elif not os.path.isdir(self.modelsDir):
                os.makedirs(self.modelsDir + '/Session_1')

        return sessFlag

class ROISelector(BoxLayout):
    def __init__(self, videoPaths, **kwargs):
        super(ROISelector, self).__init__(**kwargs)
        self.orientation = 'horizontal'
        self.videoPaths = videoPaths
        self.size_hint = (1.,1.)
        self.ROIs = [] #store rectangles on the GUI
        self.ROIOut  = [] #pass them to opencv
        self.touches = [] #store touch events on the figure


    def playSegment(self, segNum=0):
        #capture video and get some parameters
        self.capture = cv2.VideoCapture(self.videoPaths[segNum])
        ret, frame = self.capture.read()
        frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
        self.frameHeight, self.frameWidth = frame.shape
        self.fps = self.capture.get(cv2.cv.CV_CAP_PROP_FPS)
        self.numFrame = self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        #create image to store the video
        self.showFrame = Image(keep_ratio=False, allow_stretch=True, size_hint = (1.,1.))
        self.add_widget(self.showFrame)
        #visualise everything
        self.visualiseFrame()
        Clock.schedule_once(self.visualiseFrame, 0) #NOTE: it is ugly, I know

    def visualiseFrame(self,value=0):
        self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,value)
        ret, frame = self.capture.read()
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        self.frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
        buf = buf1.tostring()
        textureFrame = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.showFrame.texture = textureFrame
        self.ROIcv2 = np.zeros_like(self.frame,dtype='uint8')
        self.initImW = self.width
        self.initImH = self.height

    def on_touch_down(self, touch):
            self.touches = []
            if self.showFrame.collide_point(*touch.pos):
                self.touches.append(touch.pos)

    def on_touch_up(self, touch):
            if self.showFrame.collide_point(*touch.pos) and len(self.touches) > 0:
                try:
                    self.touches.append(touch.pos)
                    rect = [self.touches[0], self.touches[-1]]
                    sorted(rect, key=lambda x:x[1], reverse=True)
                    rectS = np.diff(rect, axis=0)[0]
                    with self.showFrame.canvas:
                        Color(1, 1, 0,.5)
                        self.rect = Rectangle(pos=(rect[0][0], rect[0][1]), size=(rectS[0],rectS[1]))
                        self.ROIs.append(self.rect)
                        # print 'point1 ', (rect[0][0], rect[0][1])
                        # print 'point2', (rect[1][0], rect[1][1])
                        ratioH = self.initImH / self.showFrame.texture.height
                        ratioW = self.initImW / self.showFrame.texture.width
                        # print 'ratioh', ratioH
                        # print 'ratiow', ratioW
                        newRectP1 = (self.rect.pos[0] / ratioW, (self.rect.pos[0] + self.rect.size[0]) / ratioW)
                        newRectP2 = (self.rect.pos[1] / ratioH, (self.rect.pos[1] + self.rect.size[1]) / ratioH)
                        # print 'new rectangle position p1 ', newRectP1
                        # print 'new rectangle position p2 ', newRectP2
                        # print 'frame dimensions', self.ROIcv2.shape
                        point1 = (int(newRectP1[0]), int(self.frame.shape[1]-newRectP2[0]))
                        point2 = (int(newRectP1[1]), int(self.frame.shape[1]-newRectP2[1]))
                        self.ROIOut.append([point1,point2])

                    self.touches = []
                except:
                    print 'stay on the figure to draw a ROI'

    def delete_ROI(self, *args):
        try:
            rect = self.ROIs[-1] #clear from the app ROIs collection
            self.ROIs = self.ROIs[:-1] #clear from the cv2 ROIs collection
            self.ROIOut = self.ROIOut[:-1]
            self.showFrame.canvas.remove(rect) #clear from the image in the visualisation
        except:
            print('Select one ROI first')

class PreprocInterfaceManager(BoxLayout):
    def __init__(self, **kwargs):
        super(PreprocInterfaceManager, self).__init__(**kwargs)
        self.id = 'preproc_interface_manager'
        self.orientation = 'vertical'
        self.root = Root
        self.window = Window
        self.initWindowHeight = self.window.height
        self.initWindowWidth = self.window.width
        self.window.bind(on_resize=self.updateROIs)

        self.ROIcv2 = None
        self.ready = False
        global CHOSEN_VIDEO
        self.cur_video_path = CHOSEN_VIDEO.get_chosen_item()
        print 'cur video in init', self.cur_video_path
        CHOSEN_VIDEO.bind(chosen=self.do)

    def do(self, *args):
        print 'cur video in do', self.cur_video_path
        print 'the chose in do ', CHOSEN_VIDEO.get_chosen_item()
        if CHOSEN_VIDEO.get_chosen_item() != self.cur_video_path and 'avi' in CHOSEN_VIDEO.get_chosen_item():
            #once a video is selected roi is initialised to None
            print 'the parent is ', self.parent
            self.canLoadROI = True
            self.storePath()

    def storePath(self):
        self.clear_widgets()
        self.videoPath = CHOSEN_VIDEO.get_chosen_item()
        self.videoPaths = scanFolder(self.videoPath)
        self.gotoBtn = Button(id='go_to_preprocessing', text='Go to preprocessing',size_hint=(.2,.5))
        self.ImageLayout = BoxLayout()
        self.add_widget(self.ImageLayout)
        self.ROIselect = ROISelector(self.videoPaths)
        self.ImageLayout.add_widget(self.ROIselect)
        self.ready = True

        self.ROIselect.playSegment()
        self.videoSlider = Slider(id='video_slider', min=0, max=100, step=1, value=0, size_hint=(1.,.2))
        self.videoSlider.bind(value=self.getValue)

        self.layout = BoxLayout(orientation='vertical', size_hint=(1.,.2))
        self.add_widget(self.layout)
        self.layout.add_widget(self.videoSlider)
        self.layoutFooter = BoxLayout(orientation='horizontal', size_hint=(1.,.5))
        self.layout.add_widget(self.layoutFooter)

        self.clearROIBtn = Button(text='clear last ROI selected', size_hint=(.2,.5))
        self.layoutFooter.add_widget(self.clearROIBtn)
        self.saveROIBtn = Button(text='save ROIs', size_hint=(.2,.5))
        self.layoutFooter.add_widget(self.saveROIBtn)
        self.loadROIBtn = Button(text='load ROIs', size_hint=(.2,.5))
        self.layoutFooter.add_widget(self.loadROIBtn)
        self.clearROIBtn.bind(on_press = self.ROIselect.delete_ROI)
        self.saveROIBtn.bind(on_press = self.roiSaver)
        self.loadROIBtn.bind(disabled = self.checkROI)
        self.loadROIBtn.bind(on_press = self.loadROI)
        self.layoutFooter.add_widget(self.gotoBtn)

        self.updateROIs(self.window, self.window.width, self.window.height)

    def getValue(self, instance, value):
        ROISelector.visualiseFrame(self.ROIselect, value)

    def updateROIs(self, window, width, height):
        self.windowWidth = width
        self.windowHeight = height

        if self.ready == True:
            self.curImgH = self.ROIselect.showFrame.height
            self.curImgW = self.ROIselect.showFrame.width
            wRatio = abs(self.curImgW / self.ROIselect.initImW)
            hRatio = abs(self.curImgH / self.ROIselect.initImH)

            for rect in self.ROIselect.ROIs:
                rect.pos = (rect.pos[0] * wRatio, rect.pos[1] * hRatio)
                rect.size = (rect.size[0] * wRatio, rect.size[1] * hRatio)

            self.ROIselect.initImH = self.curImgH
            self.ROIselect.initImW = self.curImgW

    def roiSaver(self, *args):
        if len(self.ROIselect.ROIOut) > 0:
            for p in self.ROIselect.ROIOut:
                cv2.rectangle(self.ROIselect.ROIcv2,p[0], p[1],255,-1)

        saveFile(self.videoPaths[0], self.ROIselect.ROIcv2, 'ROI')
        self.parent.ROIcv2 = self.ROIselect.ROIcv2

    def checkROI(self, pathToVideo):
        if pathToVideo != 'You did not select a video yet':
            self.videoPath = pathToVideo
            self.videoFolder = os.path.dirname(self.videoPath)
            self.ROIpath = self.videoFolder + '/preprocessing/ROI.hdf5'
            if os.path.isfile(self.ROIpath):
                print 'there is a ROI'
                return False
        else:
            return True

    def loadROI(self):
        self.ROIcv2 = loadFile(self.videoPath, 'ROI')
        self.parent.ROIcv2 = self.ROIcv2


class BkgSubtraction(BoxLayout):
    def __init__(self, **kwargs):
        super(BkgSubtraction, self).__init__(**kwargs)
        self.bkg = None
        #set useful popups
        self.saving_popup = Popup(title='Saving',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.saving_popup.bind(on_open=self.save_bkg)
        self.loading_popup = Popup(title='Loading',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.loading_popup.bind(on_open=self.load_bkg)

        self.computing_popup = Popup(title='Computing',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.computing_popup.bind(on_open=self.compute_bkg)

    def load_bkg(self, *args):
        self.bkg = pickle.load(open(self.parent.parent.bkg_path,'rb') ).astype('float32')
        self.parent.parent.bkg = self.bkg
        self.loading_popup.dismiss()

    def save_bkg(self, *args):
        pickle.dump(self.bkg.astype('float32'), open(self.parent.parent.bkg_path, 'wb'))
        self.saving_popup.dismiss()

    def compute_bkg(self, *args):
        capture = cv2.VideoCapture(self.parent.parent.videoPath)
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frameHeight, frameWidth = frame.shape
        self.videoPaths = scanFolder(self.parent.parent.videoPath)
        capture.release()
        self.bkg = computeBkg(self.videoPaths, frameWidth, frameHeight)
        self.parent.parent.bkg = self.bkg.astype('float32')
        self.computing_popup.dismiss()
        self.saving_popup.open()


class VideoShowerPreprocessing(BoxLayout):
    def __init__(self, **kwargs):
        super(VideoShowerPreprocessing, self).__init__(**kwargs)
        self.id="video_layout_validation"
        self.orientation = "vertical"
        self.show_frame_preprocessing = Image(keep_ratio=False, allow_stretch=True, size_hint = (1.,1.))
        self.add_widget(self.show_frame_preprocessing)
        self.count_scrollup = 0
        self.scale = 1

    def on_touch_down(self, touch):
        self.touches = []
        print 'scrollup number ', self.count_scrollup
        if self.parent is not None and self.show_frame_preprocessing.collide_point(*touch.pos):
            print 'i think you are on the image'
            if touch.button == 'scrollup':
                self.count_scrollup += 1

                coords = self.parent.parent.fromShowFrameToTexture(touch.pos)
                rows,cols = self.parent.parent.frame.shape
                self.scale = 1.5 * self.count_scrollup
                self.M = cv2.getRotationMatrix2D((coords[0],coords[1]),0,self.scale)
                self.dst = cv2.warpAffine(self.parent.parent.frame,self.M,(cols,rows))
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='luminance')
                textureFrame.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
                self.show_frame_preprocessing.texture = textureFrame

            elif touch.button == 'scrolldown':
                # frame = self.parent.frame
                coords = self.parent.parent.fromShowFrameToTexture(touch.pos)
                rows,cols = self.parent.parent.frame.shape
                self.dst = self.parent.parent.frame
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='luminance')
                textureFrame.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
                self.show_frame_preprocessing.texture = textureFrame
                self.count_scrollup = 0

        else:
            self.scale = 1
            # self.count_scrollup = 0


class PreprocessingPreview(BoxLayout):
    def __init__(self, **kwargs):
        super(PreprocessingPreview, self).__init__(**kwargs)
        #get video path information and bind it
        global CHOSEN_VIDEO
        self.cur_video_path = CHOSEN_VIDEO.get_chosen_item()
        print 'cur video in init preprocessing', self.cur_video_path
        CHOSEN_VIDEO.bind(chosen=self.do)
        self.bkg_path = None
        self.flag_bkg = False

        self.container_layout = BoxLayout(orientation = 'vertical', size_hint = (.3, 1.))
        self.add_widget(self.container_layout)
        # number of animals
        # self.num_of_animals_slider = Slider()
        # self.num_of_animals_slider.min = 0
        # self.num_of_animals_slider.max = 150
        # self.num_of_animals_slider.value = 0
        # self.num_of_animals_slider.step = 1
        #
        # self.num_of_animals_slider.bind(value=self.get_num_animals)
        # # num of animals label
        # self.num_of_animals_label = Label(text = 'number of animals\n' + str(int(self.num_of_animals_slider.value)))
        # self.num_of_animals_label.text_size = self.num_of_animals_label.size
        # self.num_of_animals_label.size = self.num_of_animals_label.texture_size
        # self.num_of_animals_label.font_size = 16
        # self.num_of_animals_label.halign =  "center"
        # self.num_of_animals_label.valign = "middle"
        # self.container_layout.add_widget(self.num_of_animals_label)
        # # add slider num animals
        # self.container_layout.add_widget(self.num_of_animals_slider)

        # bkg_subtraction
        self.bkg_subtractor = BkgSubtraction(orientation = 'vertical')
        self.container_layout.add_widget(self.bkg_subtractor)
        #bkg sub label
        self.bkg_subtraction_label = Label(text = 'background subtraction')
        self.bkg_subtraction_label.text_size = self.bkg_subtraction_label.size
        self.bkg_subtraction_label.size = self.bkg_subtraction_label.texture_size
        self.bkg_subtraction_label.font_size = 16
        self.bkg_subtraction_label.halign =  "center"
        self.bkg_subtraction_label.valign = "middle"
        self.container_layout.add_widget(self.bkg_subtraction_label)
        #bkg sub switch
        self.bkg_subtractor_switch = Switch()
        self.container_layout.add_widget(self.bkg_subtractor_switch)

        self.saving_popup = Popup(title='Saving',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.saving_popup.bind(on_open=self.save_preproc)

        self.max_threshold = 165
        self.min_threshold = 0
        self.min_area = 100
        self.max_area = 1000
        ###max_threshold
        self.max_threshold_slider = Slider(id = 'max_threhsold')
        self.max_threshold_slider.min = 0
        self.max_threshold_slider.max = 255
        self.max_threshold_slider.value = self.max_threshold
        self.max_threshold_slider.step = 1

        self.max_threshold_lbl = Label( id = 'max_threshold_lbl')
        self.max_threshold_lbl.text = "Max threshold:\n" + str(int(self.max_threshold_slider.value))
        self.max_threshold_lbl.text_size = self.max_threshold_lbl.size
        self.max_threshold_lbl.size = self.max_threshold_lbl.texture_size
        self.max_threshold_lbl.font_size = 16
        self.max_threshold_lbl.halign =  "center"
        self.max_threshold_lbl.valign = "middle"

        ###min_threshold
        self.min_threshold_slider = Slider(id='min_threshold_slider')
        self.min_threshold_slider.min = 0
        self.min_threshold_slider.max = 255
        self.min_threshold_slider.value = self.min_threshold
        self.min_threshold_slider.step = 1

        self.min_threshold_lbl = Label(id='min_threshold_lbl')
        self.min_threshold_lbl.text = "Min threshold:\n" + str(int(self.min_threshold_slider.value))
        self.min_threshold_lbl.text_size = self.min_threshold_lbl.size
        self.min_threshold_lbl.size = self.min_threshold_lbl.texture_size
        self.min_threshold_lbl.font_size = 16
        self.min_threshold_lbl.halign =  "center"
        self.min_threshold_lbl.valign = "middle"
        ###max_area label
        self.max_area_slider = Slider(id='max_area_slider')
        self.max_area_slider.min = 0
        self.max_area_slider.max = 60000
        self.max_area_slider.value = self.max_area
        self.max_area_slider.step = 1

        self.max_area_lbl = Label(id='max_area_lbl')
        self.max_area_lbl.text = "Max area:\n" + str(int(self.max_area_slider.value))
        self.max_area_lbl.text_size = self.max_area_lbl.size
        self.max_area_lbl.size = self.max_area_lbl.texture_size
        self.max_area_lbl.font_size = 16
        self.max_area_lbl.halign =  "center"
        self.max_area_lbl.valign = "middle"
        ###min_area
        self.min_area_slider = Slider(id='min_area_slider')
        self.min_area_slider.min = 0
        self.min_area_slider.max = 1000
        self.min_area_slider.value = self.min_area
        self.min_area_slider.step = 1

        self.min_area_lbl = Label(id='min_area_lbl')
        self.min_area_lbl.text = "Min area:\n" + str(int(self.min_area_slider.value))
        self.min_area_lbl.text_size = self.min_area_lbl.size
        self.min_area_lbl.size = self.min_area_lbl.texture_size
        self.min_area_lbl.font_size = 16
        self.min_area_lbl.halign =  "center"
        self.min_area_lbl.valign = "middle"

        self.w_list = [self.max_threshold_lbl, self.max_threshold_slider,
                        self.min_threshold_lbl, self.min_threshold_slider,
                        self.max_area_lbl, self.max_area_slider,
                        self.min_area_lbl, self.min_area_slider ]
        self.add_widget_list()

        self.max_threshold_slider.bind(value=self.update_max_th_lbl)
        self.min_threshold_slider.bind(value=self.update_min_th_lbl)
        self.max_area_slider.bind(value=self.update_max_area_lbl)
        self.min_area_slider.bind(value=self.update_min_area_lbl)

        #create button to load parameters
        self.load_prec_params_btn = Button()
        self.load_prec_params_btn.text = "Load preprocessing params"
        self.load_prec_params_btn.bind(on_press = self.load_preproc_params)
        self.load_prec_params_btn.bind(disabled = CHOSEN_VIDEO.get_preproc_path)
        #create button to save parameter
        self.segment_video_btn = Button()
        self.segment_video_btn.text = "Segment video"
        # self.load_prec_params_btn.bind(on_press = self.laod_preproc_params)

    def do(self, *args):
        print 'cur video in do', self.cur_video_path
        print 'the chose in do ', CHOSEN_VIDEO.get_chosen_item()
        if CHOSEN_VIDEO.get_chosen_item() != self.cur_video_path and 'avi' in CHOSEN_VIDEO.get_chosen_item():
            self.videoPath = CHOSEN_VIDEO.get_chosen_item()
            self.videoPaths = scanFolder(self.videoPath)
            self.ROI = None

            #get info on the video (index of all the frames, number of frames)

            #whenever the bkg switch is active check the bgk path (this will be also updated whenever the path to the
            #video is updated).
            self.preproc_path = CHOSEN_VIDEO.preproc_path
            self.bkg_path = CHOSEN_VIDEO.bkg_path
            self.bkg = self.check_bkg()
            self.bkg_subtractor_switch.bind(active = self.set_bkg_path)
            self.bkg_subtractor_switch.active = self.flag_bkg
            print 'active? ', self.bkg_subtractor_switch.active
            #bind the switch to the background loader / computer
            self.bkg_path = CHOSEN_VIDEO.bkg_path
            # print 'bkg check func ', self.bkg_subtractor.check_bkg
            # print 'bkg path ', self.bkg_subtractor.bkg_path
            self.get_info()
            self.init_segment_zero()

    def set_bkg_path(self, instance, active):
        CHOSEN_VIDEO.get_bkg_path()
        self.flag_bkg = CHOSEN_VIDEO.flag_bkg
        self.bkg_path = CHOSEN_VIDEO.bkg_path
        print 'flag: ', self.flag_bkg
        print 'instance ', instance
        print 'active ', active
        if active == True:
            self.check_bkg()
            print 'bkg type', type(self.bkg)
            print self.bkg.dtype
        #update the bkg in show preprocessing
        self.show_preprocessing(self.videoSlider.value)
        return self.flag_bkg

    def check_bkg(self):
        """
        Checks if the background has already been computed. If it is,
        it loads it, otherwise it computes it.
        """
        print 'the bkg path is ===========', self.bkg_path

        if os.path.isfile(self.bkg_path):
            #load bkg
            self.bkg_subtractor.loading_popup.open()
        else:
            #compute it
            self.bkg_subtractor.computing_popup.open()


    def get_info(self):
        self.frameIndices, self.segmPaths  = getSegmPaths(self.videoPaths)
        video = os.path.basename(self.videoPath)
        self.num_frames = len(self.frameIndices)
        #capture video and get some parameters
        self.capture = cv2.VideoCapture(self.videoPaths[0])
        ret, frame = self.capture.read()
        frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
        self.video_height, self.video_width = frame.shape

    def add_widget_list(self):
        for w in self.w_list:
            self.container_layout.add_widget(w)

    def get_num_animals(self, instance, value):
        self.num_animals = int(value)
        self.num_of_animals_label.text = "Num of animals:\n" +  str(self.num_animals)

    def update_max_th_lbl(self,instance, value):
        self.max_threshold_lbl.text = "Max threshold:\n" +  str(int(value))
        self.show_preprocessing(self.videoSlider.value)

    def update_min_th_lbl(self,instance, value):
        self.min_threshold_lbl.text = "Min threshold:\n" + str(int(value))
        self.show_preprocessing(self.videoSlider.value)

    def update_max_area_lbl(self,instance, value):
        self.max_area_lbl.text = "Max area:\n" + str(int(value))
        self.show_preprocessing(self.videoSlider.value)

    def update_min_area_lbl(self,instance, value):
        self.min_area_lbl.text = "Min area:\n" + str(int(value))
        self.show_preprocessing(self.videoSlider.value)

    def load_preproc_params(self, *args):
        if not CHOSEN_VIDEO.flag_prep:
            p_params = pickle.load(open(self.preproc_path,'rb'))
            self.max_threshold_slider.value = p_params['maxThreshold']
            self.min_threshold_slider.value = p_params['minThreshold']
            self.min_area_slider.value = p_params['minArea']
            self.max_area_slider.value = p_params['maxArea']

    def save_preproc_params(self):
        self.saving_popup.open()
        self.p_params = {'maxThreshold':0,
                        'minThreshold':0,
                        'minArea':0,
                        'maxArea':0}
        self.p_params['maxThreshold'] = self.max_threshold_slider.value
        self.p_params['minThreshold'] = self.min_threshold_slider.value
        self.p_params['minArea'] = self.min_area_slider.value
        self.p_params['maxArea'] = self.max_area_slider.value

    def save_preproc(self, *args):
        pickle.dump(self.p_params, open(self.preprocPath, 'wb'))
        self.saving_popup.dismiss()

    def init_segment_zero(self):
        # create instance of video shower
        self.video_shower = VideoShowerPreprocessing()
        self.currentSegment = 0
        #create layout for video and slider
        self.video_layout_preprocessing = BoxLayout(orientation = 'vertical')
        self.add_widget(self.video_layout_preprocessing)
        self.sliderBox = BoxLayout(id="slider_box", orientation="horizontal", size_hint=(1.,.2))
        # add video slider
        #video slider
        self.videoSlider = Slider(id='preprocessing_video_slider',
                                    min=0,
                                    max=self.num_frames,
                                    step=1,
                                    value=0,
                                    size_hint=(1.,1.))

        self.sliderValue = Label(id="slider_value",
                                text = 'frame 0',
                                font_size = 18,
                                size_hint=(.2,1.))

        self.sliderValue.size  = self.sliderValue.texture_size
        self.sliderBox.add_widget(self.videoSlider)
        self.sliderBox.add_widget(self.sliderValue)
        self.videoSlider.bind(value=self.on_slider_value_change)
        #create image to store the video
        self.video_layout_preprocessing.add_widget(self.video_shower)
        self.video_layout_preprocessing.add_widget(self.sliderBox)
        self.button_layout = BoxLayout(orientation="horizontal", size_hint=(1.,.1))
        self.button_layout.add_widget(self.load_prec_params_btn)
        self.button_layout.add_widget(self.segment_video_btn)
        self.video_layout_preprocessing.add_widget(self.button_layout)

        # if no ROI
        if self.ROI == None:
            self.ROI = np.ones((self.video_height, self.video_width)).astype('uint8')
        #visualise everything
        self.show_preprocessing()

    def show_preprocessing(self, value = 0):
        sNumber = self.frameIndices.loc[value,'segment']
        sFrame = self.frameIndices.loc[value,'frame']

        if sNumber != self.currentSegment:
            print 'Changing segment...'
            self.currentSegment = sNumber

            if len(self.videoPaths) > 1:
                self.capture = cv2.VideoCapture(self.videoPaths[sNumber-1])
                self.numFrameCurSegment = int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        #Get frame from video file
        if len(self.videoPaths) == 1:
            self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,value)
        elif len(self.videoPaths) > 1:
            self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame)
        # read frame
        ret, frame = self.capture.read()
        # pass frame to grayscale
        self.frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
        # compute average intensity
        avIntensity = np.float32(np.mean(self.frame))
        # generate a averaged frame for segmentation
        self.av_frame = self.frame / avIntensity
        # threshold the frame according to the sliders' values
        print 'this is the current background ', self.bkg
        self.segmented_frame = segmentVideo(self.av_frame,
                                            int(self.min_threshold_slider.value),
                                            int(self.max_threshold_slider.value),
                                            self.bkg,
                                            self.ROI,
                                            self.bkg_subtractor_switch.active)
        #get information on the blobs find by thresholding
        boundingBoxes, miniFrames, _, _, _, goodContours, bkgSamples = blobExtractor(self.segmented_frame,
                                                                                    self.frame,
                                                                                    int(self.min_area_slider.value),
                                                                                    int(self.max_area_slider.value),
                                                                                    self.video_height,
                                                                                    self.video_width)
        #draw the blobs on the original frame
        cv2.drawContours(self.frame, goodContours, -1, color=255, thickness = -1)
        #display the segmentation
        if self.video_shower.count_scrollup != 0:
            self.dst = cv2.warpAffine(self.frame,self.video_shower.M,(self.video_width, self.video_height))
            buf1 = cv2.flip(self.dst,0)
        else:
            buf1 = cv2.flip(self.frame, 0)

        buf = buf1.tostring()
        textureFrame = Texture.create(size=(self.video_width, self.video_height), colorfmt='luminance')
        textureFrame.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
        # display image from the texture
        self.video_shower.show_frame_preprocessing.texture = textureFrame

    def on_slider_value_change(self,instance,value):
        self.sliderValue.text = 'frame ' + str(int(value))
        self.show_preprocessing(int(value))

    def fromShowFrameToTexture(self, coords):
        """
        Maps coordinate in show_frame_preprocessing (the image whose texture is the frame) to
        the coordinates of the original image
        """
        coords = np.asarray(coords)
        origFrameW = self.video_width
        origFrameH = self.video_height

        actualFrameW, actualFrameH = self.video_shower.show_frame_preprocessing.size
        self.y_offset = self.sliderBox.height + self.button_layout.height
        self.x_offset = self.container_layout.width
        coords[0] = coords[0] - self.x_offset
        coords[1] = coords[1] - self.y_offset
        wRatio = abs(origFrameW / actualFrameW)
        hRatio = abs(origFrameH / actualFrameH)
        ratios = np.asarray([wRatio, hRatio])
        coords =  np.multiply(coords, ratios)
        coords[1] = origFrameH - coords[1]
        return coords


#
# class Tracker(BoxLayout):
#     pass
#
class VideoShower(BoxLayout):
    def __init__(self, **kwargs):
        super(VideoShower, self).__init__(**kwargs)
        self.id="video_layout_validation"
        self.orientation = "vertical"
        self.showFrame = Image(keep_ratio=False, allow_stretch=True, size_hint = (1.,1.))
        self.add_widget(self.showFrame)
        self.count_scrollup = 0
        self.scale = 1

    def modifyIdOpenPopup(self, id_to_modify):
        self.container = BoxLayout()
        self.id_to_modify = id_to_modify
        text = str(self.id_to_modify + 1)
        self.old_id_box = BoxLayout(orientation="vertical")
        self.new_id_box = BoxLayout(orientation="vertical")
        self.selected_label = Label(text='You selected animal:\n')
        self.selected_label_num = Label(text=text)
        self.selected_label.text_size = self.selected_label.size
        self.selected_label.texture_size = self.selected_label.size
        self.new_id_label = Label(text='Type the new identity and press enter to confirm\n')
        self.new_id_label.text_size = self.new_id_label.size
        self.new_id_label.texture_size = self.new_id_label.size
        self.container.add_widget(self.old_id_box)
        self.container.add_widget(self.new_id_box)

        self.old_id_box.add_widget(self.selected_label)
        self.old_id_box.add_widget(self.selected_label_num)

        self.new_id_box.add_widget(self.new_id_label)
        self.identityInput = TextInput(text ='', multiline=False)
        self.new_id_box.add_widget(self.identityInput)

        self.popup = Popup(title='Correcting identity',
            content=self.container,
            size_hint=(.4,.4))
        self.popup.color = (0.,0.,0.,0.)

        if self.parent is not None:
            self.identityInput.bind(on_text_validate=self.parent.on_enter)
            self.popup.open()

    def on_touch_down(self, touch):
        self.touches = []
        print 'scrollup number ', self.count_scrollup
        if self.parent is not None and self.showFrame.collide_point(*touch.pos):
            if touch.button =='left':
                self.touches.append(touch.pos)
                self.parent.id_to_modify = self.parent.correctIdentity()
                print 'fish id to modify: ', self.parent.id_to_modify
                self.modifyIdOpenPopup(self.parent.id_to_modify)

            elif touch.button == 'scrollup':
                self.count_scrollup += 1

                coords = self.parent.fromShowFrameToTexture(touch.pos)
                rows,cols, channels = self.parent.frame.shape
                self.scale = 1.5 * self.count_scrollup
                self.M = cv2.getRotationMatrix2D((coords[0],coords[1]),0,self.scale)
                self.dst = cv2.warpAffine(self.parent.frame,self.M,(cols,rows))
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='bgr')
                textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.showFrame.texture = textureFrame

            elif touch.button == 'scrolldown':
                # frame = self.parent.frame
                coords = self.parent.fromShowFrameToTexture(touch.pos)
                rows,cols, channels = self.parent.frame.shape
                self.dst = self.parent.frame
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='bgr')
                textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.showFrame.texture = textureFrame
                self.count_scrollup = 0

            elif touch.button == 'right':
                pass
        else:
            self.scale = 1


class Validator(BoxLayout):
    def __init__(self, **kwargs):
        super(Validator, self).__init__(**kwargs)

    def show_saving(self, *args):
        self.popup_saving = Popup(title='Saving',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.popup_saving.open()

    def showLoading(self):
        self.popup = Popup(title='Loading',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.popup.open()

    def validate(self, path):
        self.clear_widgets()
        self.videoPath = path
        self.videoPaths = scanFolder(self.videoPath)
        self.videoLayout = BoxLayout()
        self.get_info()
        self.frameIndices = self.frameIndices
        self.segmPaths = self.segmPaths
        self.init_segmentZero()
        self.writeIds()
        self.popup.dismiss()

    def get_info(self):
        self.frameIndices, self.segmPaths  = getSegmPaths(self.videoPaths)
        videoInfo = loadFile(self.videoPath, 'videoInfo', hdfpkl='pkl')
        video = os.path.basename(self.videoPath)
        self.folder = os.path.dirname(self.videoPath)
        # self.filename, self.extension = os.path.splitext(video)
        self.subFolder = self.folder + '/CNN_models'
        self.subSubFolders = glob.glob(self.subFolder +"/*")
        self.lastIndex = getLastSession(self.subSubFolders)
        self.sessionPath = self.subFolder + '/Session_' + str(self.lastIndex)
        self.stats = pickle.load( open( self.sessionPath + "/statistics.pkl", "rb" ) )
        self.dfGlobal = loadFile(self.videoPath, 'portraits')
        self.numAnimals = videoInfo['numAnimals']
        self.video_width = videoInfo['width']
        self.video_height = videoInfo['height']
        self.allIdentities = self.stats['fragmentIds']
        self.colors = get_spaced_colors_util(self.numAnimals)
        if len(self.videoPaths) > 1:
            self.partialFrameSum = np.zeros(len(self.videoPaths))
            self.partialFrameSum[0] = 0
            for i in range(0,len(self.videoPaths)-1):
                cap = cv2.VideoCapture(self.videoPaths[i])
                numFrame = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                self.partialFrameSum[i + 1] = self.partialFrameSum[i] + numFrame

            self.partialFrameSum = self.partialFrameSum.astype('int')

    def init_segmentZero(self):
        self.segmDf, sNumber = loadFile(self.segmPaths[0], 'segmentation')
        self.currentSegment = int(sNumber)
        self.capture = cv2.VideoCapture(self.videoPaths[0])
        self.numFrames = len(self.frameIndices)
        self.numFrameCurSegment = int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        self.videoShower = VideoShower()
        # self.videoLayout = BoxLayout(id="video_layout_validation", orientation = "vertical")
        self.videoSlider = Slider(id='validation_slider', min=0, max=self.numFrames, step=1, value=0, size_hint=(1.,.2))
        self.add_widget(self.videoShower)

        self.sliderBox = BoxLayout(id="slider_box", orientation="horizontal", size_hint=(1,.3))
        self.sliderValue = Label(id="slider_value",text = 'frame 0', font_size = 18, size_hint=(.2,.2))
        self.sliderValue.size  = self.sliderValue.texture_size

        self.add_widget(self.sliderBox)
        self.sliderBox.add_widget(self.videoSlider)
        self.sliderBox.add_widget(self.sliderValue)
        self.videoSlider.bind(value=self.OnSliderValueChange)
        self.buttonBox = BoxLayout(orientation='vertical', size_hint=(.3,1.))
        self.sliderBox.add_widget(self.buttonBox)
        self.crossButton = Button(id='crossing_btn', text='Go to next crossing', size_hint=(1,1))
        self.crossButton.bind(on_press=self.goToCrossing)
        self.buttonBox.add_widget(self.crossButton)
        #it is used to detect crossing given the list of identities:
        self.setRangeAnimals = set(range(self.numAnimals))

        self.saveGroundtruthBtn = Button(id='save_groundtruth_btn', text='Save updated identities',size_hint = (1,1))
        self.saveGroundtruthBtn.bind(on_press=self.show_saving)
        self.saveGroundtruthBtn.bind(on_release=self.save_groundtruth)
        self.saveGroundtruthBtn.disabled = True
        self.buttonBox.add_widget(self.saveGroundtruthBtn)
        print 'disabled ', self.parent.disabled

    def goToCrossing(self,instance):
        unCross = True
        ind = int(self.videoSlider.value)
        while unCross == True:
            ind = ind + 1
            newids = self.allIdentities[ind]
            if len(set.intersection(set(newids),self.setRangeAnimals)) < self.numAnimals:
                unCross = False
                self.videoSlider.value = ind
                self.writeIds(ind)

    def correctIdentity(self):
        mouse_coords = self.videoShower.touches[0]
        mouse_coords = self.fromShowFrameToTexture(mouse_coords)
        cur_frame = int(self.videoSlider.value) #get the current frame from the slider


        centroids = self.dfGlobal.loc[cur_frame,'centroids'] # get the centroids
        print 'centroids in original image', centroids
        # print 'centroids ', centroids
        if self.videoShower.scale != 1:
            print 'transformation: ', self.videoShower.M
            print 'zoom shape ', self.videoShower.dst.shape
            R = self.videoShower.M[:,:-1]
            T = self.videoShower.M[:,-1]


            centroids = [np.dot(R, centroid) + T for centroid in centroids]
        print 'transformed centroids ', centroids
        centroid_ind = self.getNearestCentroid(mouse_coords, centroids) # compute the nearest centroid
        # print 'centroid ind ', centroid_ind
        # print 'self.all'
        id_to_modify = self.allIdentities[cur_frame, centroid_ind] # get the identity
        return id_to_modify

    def fromShowFrameToTexture(self, coords):
        """
        Maps coordinate in showFrame (the image whose texture is the frame) to
        the coordinates of the original image
        """
        coords = np.asarray(coords)
        origFrameW = self.video_width
        origFrameH = self.video_height

        actualFrameW, actualFrameH = self.videoShower.showFrame.size
        self.offset = self.sliderBox.height
        coords[1] = coords[1] - self.offset
        wRatio = abs(origFrameW / actualFrameW)
        hRatio = abs(origFrameH / actualFrameH)
        ratios = np.asarray([wRatio, hRatio])
        coords =  np.multiply(coords, ratios)
        coords[1] = origFrameH - coords[1]
        return coords

    def getNearestCentroid(self, point, cents):
        """
        Finds the nearest neighbour in cents with respect to point (in 2D)
        """
        cents = np.asarray(cents)
        print self.videoShower.scale
        point = np.asarray(point)
        # print 'point rescaled ', point
        cents_x = cents[:,0]
        cents_y = cents[:,1]
        dist_x = cents_x - point[0]
        dist_y = cents_y - point[1]
        distances = dist_x**2 + dist_y**2
        return np.argmin(distances)

    def writeIds(self, value = 0):
        sNumber = self.frameIndices.loc[value,'segment']
        sFrame = self.frameIndices.loc[value,'frame']

        if sNumber != self.currentSegment:
            print 'Changing segment...'
            self.segmDf, _ = loadFile(self.segmPaths[sNumber-1], 'segmentation')
            prevSegmDf, _ = loadFile(self.segmPaths[sNumber-2], 'segmentation')
            self.currentSegment = sNumber

            if len(self.videoPaths) > 1:
                self.capture = cv2.VideoCapture(self.videoPaths[sNumber-1])
                self.numFrameCurSegment = int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        #Get frame from video file
        if len(self.videoPaths) == 1:
            self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,value)
        elif len(self.videoPaths) > 1:
            self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,value - self.partialFrameSum[sNumber - 1])

        ret, frame = self.capture.read()
        self.frame = frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        noses = self.dfGlobal.loc[value,'noses']
        centroids = self.dfGlobal.loc[value,'centroids']
        pixels = self.segmDf.loc[sFrame,'pixels']

        for i in range(len(noses)):
            cur_id = self.allIdentities[value - 1,i]

        for i, (centroid,nose) in enumerate(zip(centroids,noses)):
            # print centroid
            cur_id = self.allIdentities[value,i]
            fontSize = .5
            text = str(cur_id)
            color = [0,0,0]
            thickness = 2
            cv2.putText(frame, str(cur_id + 1),(centroid[0] - 10, centroid[1] - 10) , font, 1, self.colors[cur_id + 1],2)
            cv2.circle(frame, centroid, 2, self.colors[cur_id + 1], 2)
            cv2.circle(frame, nose, 2, self.colors[cur_id + 1], 2)

        # Visualization of the process
        if self.videoShower.scale != 1:
            self.dst = cv2.warpAffine(frame,self.videoShower.M,(frame.shape[1], frame.shape[0]))
            buf1 = cv2.flip(self.dst,0)
        else:
            buf1 = cv2.flip(frame, 0)

        buf = buf1.tostring()
        textureFrame = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.videoShower.showFrame.texture = textureFrame

    def OnSliderValueChange(self,instance,value):
        self.sliderValue.text = 'frame ' + str(int(value))
        self.writeIds(int(value))

    def on_enter(self,value):
        self.identity_update = int(self.videoShower.identityInput.text) - 1
        print self.identity_update
        self.overwriteIdentity()
        self.videoShower.popup.dismiss()

    def overwriteIdentity(self):
        self.saveGroundtruthBtn.disabled = False
        num_frame = int(self.videoSlider.value)
        old_id = self.id_to_modify
        new_id = self.identity_update
        #update in the past
        id_is_there_past = True #check if the identity is there
        id_is_there_future = True #check if the identity is there
        num_frame = int(self.videoSlider.value)
        while id_is_there_past == True:
            num_frame = num_frame - 1
            ids = self.allIdentities[num_frame]
            if old_id in ids:
                ids[ids == old_id] = new_id
                self.allIdentities[num_frame] = ids
            else:
                id_is_there_past = False

        #update in the future
        num_frame = int(self.videoSlider.value)
        while id_is_there_future == True:
            ids = self.allIdentities[num_frame]
            if old_id in ids:
                ids[ids == old_id] = new_id
                self.allIdentities[num_frame] = ids
            else:
                id_is_there_future = False

            num_frame = num_frame + 1
        self.writeIds(value = int(self.videoSlider.value))

    def on_press_show_saving(selg, *args):
        self.show_saving()

    def save_groundtruth(self, *args):
        new_stats = self.stats
        new_stats['fragmentIds'] = self.allIdentities
        pathToGroundtruth = self.sessionPath + "/groundtruth.pkl"
        pickle.dump( new_stats , open( pathToGroundtruth, "wb" ) )
        self.popup_saving.dismiss()

    def go_and_save(self, path, dict_to_save):
        pickle.dump( dict_to_save , open( path, "wb" ) )


class Root(TabbedPanel):

    def __init__(self, **kwargs):
        super(Root, self).__init__(**kwargs)
        self.bind(current_tab=self.content_changed_cb)


    def content_changed_cb(self, obj, value):
        print 'CONTEENT', value.content.id
        print type(value.content.id)

    def on_switch(self, header):
        super(Root, self). switch_to(header)
        print 'switch_to, content is ', header.content
        self.cur_content = header.content

    def goToTab(self,id):
        self.switch_to(id)

    def assignPath(self, string):
        self.pathToVideo = string

    def getPath(self):
        return self.pathToVideo

class Screen(BoxLayout):
    pass

class idApp(App):
    Config.set('kivy', 'keyboard_mode', '')
    Config.write()
    def build(self):
        return Screen()

if __name__ == '__main__':
    idApp().run()
