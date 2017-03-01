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
from kivy.uix.image import Image, AsyncImage
from kivy.uix.slider import Slider
from kivy.uix.scatter import Scatter
from kivy.graphics import *
from kivy.graphics.svg import Svg
from kivy.clock import Clock
from kivy.loader import Loader

import os
import cPickle as pickle
import cv2
import numpy as np
import glob
import re

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

path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesShort/Caffeine5fish_20140206T122428_1.avi'

class ROISelector(BoxLayout):
    def __init__(self, **kwargs):
        super(ROISelector, self).__init__(**kwargs)
        self.orientation = 'horizontal'
        self.videoPaths = scanFolder(path)
        print self.parent
        self.ROIs = []
        self.window = Window
        self.initWindowHeight = self.window.height
        self.initWindowWidth = self.window.width
        self.playSegment()
        self.window.bind(on_resize=self.updateROIs)

    def playSegment(self, segNum=0):
        #capture video and get some parameters
        self.capture = cv2.VideoCapture(self.videoPaths[segNum])
        ret, frame = self.capture.read()
        print ret
        frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
        self.frameHeight, self.frameWidth = frame.shape
        self.fps = self.capture.get(cv2.cv.CV_CAP_PROP_FPS)
        self.numFrame = self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        #create image to store the video
        self.showFrame = Image(keep_ratio=False, allow_stretch=True, size_hint = (1.,1.))
        self.add_widget(self.showFrame)
        #visualise everything
        Clock.schedule_once(self.visualiseFrame, 1)


    def visualiseFrame(self,value=1):
        self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,value)
        ret, frame = self.capture.read()
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        textureFrame = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.showFrame.texture = textureFrame
        self.ROIcv2 = np.ones_like(frame,dtype='uint8')*255
        self.initImW = self.width
        self.initImH = self.height

    def on_touch_down(self, touch):
            self.touches = []
            if self.showFrame.collide_point(*touch.pos):
                self.touches.append(touch.pos)
                # print self.touches
                # with self.showFrame.canvas:
                #     Color(1, 1, 0)
                #     d = 4.
                #     self.ellStart = Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))

    def on_touch_up(self, touch):
            if self.showFrame.collide_point(*touch.pos):
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
                        cv2.rectangle(self.ROIcv2,(rect[0][0], rect[0][1]),(rect[1][0], rect[1][1]),0,-1)
                        cv2.imshow('test',self.ROIcv2)
                        cv2.waitKey(0)
                    self.touches = []
                except:
                    print 'stay on the figure to draw a ROI'

    def delete_ROI(self, *args):
        try:
            print 'you rectangles should be ', self.ROIs
            rect = self.ROIs[-1]
            self.ROIs = self.ROIs[:-1]
            print 'the rectangle you should have selected: ',rect
            self.showFrame.canvas.remove(rect)
                # clear(rect)
        except:
            print('Select one ROI first')

    def updateROIs(self, window, width, height):
        print '----------------'
        print 'initial window height ', self.initWindowHeight
        print 'initial window width ',self.initWindowWidth
        self.windowWidth = width
        self.windowHeight = height
        print 'actual window height: ', self.windowHeight
        print 'actual window width: ', self.windowWidth
        print '\n'
        # if self.ready == True:
        # try:
        print '>>>>>>>>>>>>'
        print 'initial image height ',self.initImH
        print 'initial image width ', self.initImW

        print 'actual height image ', self.showFrame.height
        print 'actual width image ', self.showFrame.width
        self.curImgH = self.showFrame.height
        self.curImgW = self.showFrame.width
        wRatio = abs(self.curImgW / self.initImW)
        hRatio = abs(self.curImgH / self.initImH)
        print 'ratios w, h', wRatio, hRatio
        for rect in self.ROIs:
            print '[][][][][][][][][][][]'
            print 'current rectangle position', rect.pos
            print 'current rectangle size', rect.size
            rect.pos = (rect.pos[0] * wRatio, rect.pos[1] * hRatio)
            print 'new rectangle position', rect.pos

            rect.size = (rect.size[0] * wRatio, rect.size[1]*hRatio)

        self.initImH = self.curImgH
        self.initImW = self.curImgW


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
