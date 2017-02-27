# native
from __future__ import division
import sys

# import matplotlib
# matplotlib.use('module://kivy.garden.matplotlib.backend_kivyagg')
# from matplotlib.widgets import RectangleSelector
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

#idTracker
sys.path.append('../utils')
from py_utils import *
# from GUI_utils import *
# third party
try:
    import pyautogui
except:
    print("install pyautogui to get the app fullscreen")


#kivy
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.graphics.texture import Texture
from kivy.graphics import *
from kivy.input.shape import ShapeRect
import cv2

Builder.load_file('./idTracker.kv')


class IOButton(Button):
    def __init__(self,**kwargs):
        super(IOButton,self).__init__(**kwargs)
        self.activeColor = [0,0,1,1]
        self.inactiveColor = [0,1,0,1]
        self.color = self.inactiveColor

    def on_press(self):
        if self.color == self.inactiveColor:
            self.color = self.activeColor
        else:
            self.color=self.inactiveColor

    def button_press(self):
        self.bind(on_state = self.on_event)

    def on_event(self):
        if self.color == self.inactiveColor:
            self.color = self.activeColor
        else:
            self.color=self.inactiveColor

    def on_release(self):
        self.color = self.color





class videoWin(FloatLayout):
    def __init__(self,**kwargs):
        super(videoWin,self).__init__(**kwargs)
        self.videoPaths = scanFolder('../videos/Conflicto8/conflict3and4_20120316T155032_1.avi')
        self.window = Window
        self.window.bind(on_resize=self.getSizeImg)
        self.playSegment()
        # print 'height of the root', self.parent.height

    def getSizeImg(self, window, width, height):
        self.windowWidth = width
        self.windowHeight = height

        # if self.img1.height != .9 * height:
        #     self.img1.height = .9 * height
        #
        # if self.img1.width != .9 * width:
        #     self.img1.width = .9 * width
        #
        # print self.img1.height
        # print self.img1.width


    def playSegment(self):
        self.capture = cv2.VideoCapture(self.videoPaths[0])
        ret, frame = self.capture.read()
        frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
        self.frameHeight, self.frameWidth = frame.shape
        self.fps = self.capture.get(cv2.cv.CV_CAP_PROP_FPS)
        self.numFrame = self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

        self.visualiseFrame()
        self.createSlider()

    def createSlider(self):
        self.childLayoutSlide = FloatLayout()
        self.videoSlider = Slider(min=0, max=self.numFrame - 1, value=1,step=1)
        self.videoSlider.pos_hint = {"y":0}
        self.videoSlider.size_hint = (1,.1)
        self.add_widget(self.childLayoutSlide)
        self.childLayoutSlide.add_widget(self.videoSlider)
        self.videoSlider.bind(value=self.on_changeSlider)

    def on_changeSlider(self, instance, value):
        self.visualiseFrame(value)

    def createImageWidget(self,width=0, height=0):
        self.childLayoutIm = FloatLayout()
        self.add_widget(self.childLayoutIm)
        if height == 0 and width == 0:
            self.img1 = Image(size_hint= (.5,.5), pos_hint = {"x":0.25, "y":0.25})
        else:
            self.img1 = Image( size_hint= (None,None), size=(width, height))

        self.childLayoutIm.add_widget(self.img1)
        self.touches = []


    def visualiseFrame(self,value=0):

        self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,value)
        ret, frame = self.capture.read()
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.imTrueW = frame.shape[1]
        self.imTrueH = frame.shape[0]
        self.createImageWidget()#frame.shape[1], frame.shape[0])
        self.img1.texture = texture1

    def on_touch_down(self, touch):
        self.touches = []
        if self.img1.collide_point(*touch.pos):
            with self.canvas:
                Color(1, 1, 0)
                d = 4.
                Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))

    def on_touch_up(self, touch):
        if self.img1.collide_point(*touch.pos):
            rect = [self.touches[0], self.touches[-1]]
            sorted(rect, key=lambda x:x[1], reverse=True)
            rectS = np.diff(rect, axis=0)[0]
            print rect
            print rectS
            with self.canvas:
                Color(1, 1, 0,.5)
                Rectangle(pos=(rect[0][0], rect[0][1]), size=(rectS[0],rectS[1]))



            with self.canvas:
                Color(1, 1, 0)
                d = 4.
                Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))

            self.touches = []

    def on_touch_move(self, touch):

        if self.img1.collide_point(*touch.pos):
            self.touches.append(touch.pos)





class tabs(TabbedPanel):
    pass

class idTrackerApp(App):
    def build(self):
        try:
            self.wScreen, self.hScreen = self.getScreenSize()
            Window.size = (self.wScreen, self.hScreen)
        except:
            print("Couldn't measure your screen")
        return tabs()

    def getScreenSize(self):
        wScreen, hScreen = pyautogui.size()
        return wScreen, hScreen

    def OnSliderValueChange(self,instance,value):
        print int(value)

    def onSwitchValueChange(self,instance, value):
        print('the switch', instance, 'is', value)

    def generateTrackBar(self, minV, maxV, value):
        self.s = Slider(min=minV, max=maxV, value=value, step=1, pos=(0,0), size=(100,200))

        return self.s

if __name__ == '__main__':
    idTrackerApp().run()
