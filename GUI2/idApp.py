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
from kivy.graphics import *
from kivy.graphics.svg import Svg
from kivy.clock import Clock

import os
import cPickle as pickle
import cv2
import numpy as np
import glob
import re
import sys
sys.path.append('../utils')

from py_utils import flatten, saveFile, loadFile





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

class SelectFile(BoxLayout):
    def __init__(self,**kwargs):
        super(SelectFile,self).__init__(**kwargs)
        self.update = ''

    def open(self, path, filename):
        if filename:
            if '.avi' not in filename[0]:
                return True
            else:
                print 'you selected something'
                self.update = filename[0]
                return False
        else:
            print('Select a file before opening it!')
            return True

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
                        print 'point1 ', (rect[0][0], rect[0][1])
                        print 'point2', (rect[1][0], rect[1][1])
                        ratioH = self.initImH / self.showFrame.texture.height
                        ratioW = self.initImW / self.showFrame.texture.width
                        print 'ratioh', ratioH
                        print 'ratiow', ratioW
                        newRectP1 = (self.rect.pos[0] / ratioW, (self.rect.pos[0] + self.rect.size[0]) / ratioW)
                        newRectP2 = (self.rect.pos[1] / ratioH, (self.rect.pos[1] + self.rect.size[1]) / ratioH)
                        print 'new rectangle position p1 ', newRectP1
                        print 'new rectangle position p2 ', newRectP2
                        print 'frame dimensions', self.ROIcv2.shape
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
        self.canLoadROI = True
        self.ready = False

    def storePath(self,string):
        self.path = string
        print self.path
        self.videoPaths = scanFolder(self.path)
        self.clear_widgets()
        self.gotoBtn = Button(id='go_to_preprocessing', text='Go to preprocessing',size_hint=(.2,.5))
        # tabList = Root.get_tab_list(Root)
        # print tabList
        # print tabList[0]
        # self.gotoBtn.bind(on_press = Root.switch_to(self.parent.parent.parent.tab_list[0]))
        print '...........'
        print Root.ids
        print '............'
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
        self.clearROIBtn.bind(on_press=self.ROIselect.delete_ROI)
        self.saveROIBtn.bind(on_press=self.roiSaver)
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
                rect.size = (rect.size[0] * wRatio, rect.size[1]*hRatio)

            self.ROIselect.initImH = self.curImgH
            self.ROIselect.initImW = self.curImgW

    def roiSaver(self, *args):
        if len(self.ROIselect.ROIOut) > 0:
            for p in self.ROIselect.ROIOut:
                cv2.rectangle(self.ROIselect.ROIcv2,p[0], p[1],255,-1)

        saveFile(self.videoPaths[0], self.ROIselect.ROIcv2, 'ROI')

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



class Root(TabbedPanel):
    pathToVideo = StringProperty("You did not select a video yet")

    def goToTab(self,id):
        self.switch_to(id)

    def assignPath(self, string):
        self.pathToVideo = string

    pass

class Screen(BoxLayout):
    pass

class idApp(App):
    def build(self):
        return Screen()

if __name__ == '__main__':
    idApp().run()
