from __future__ import absolute_import, division, print_function
import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.uix.switch import Switch
from kivy.graphics import *
from kivy.graphics.transformation import Matrix
from idtrackerai.gui.kivy_utils import CustomLabel
import cv2

class VisualiseVideo(BoxLayout):
    def __init__(self,
                chosen_video = None,
                **kwargs):
        super(VisualiseVideo, self).__init__(**kwargs)
        global CHOSEN_VIDEO
        CHOSEN_VIDEO = chosen_video
        self.orientation = "vertical"
        self.display_layout = Image(keep_ratio=False,
                                    allow_stretch=True,
                                    size_hint = (1.,1.))
        self.footer = BoxLayout()
        self.footer.size_hint = (1.,.2)

    def visualise_video(self, video_object, func = None, frame_index_to_start = 0):
        self.video_object = video_object
        self.add_widget(self.display_layout)
        self.add_slider()
        self.add_widget(self.footer)
        self.cap = cv2.VideoCapture(self.video_object.video_path)
        self.func = func
        self.video_slider.value = frame_index_to_start
        self.visualise(frame_index_to_start, func = func)

    def add_slider(self):
        self.video_slider = Slider(id='video_slider',
                                min=0,
                                max= int(self.video_object.number_of_frames) - 1,
                                step=1,
                                value=0,
                                size_hint=(.8,1.))
        self.video_slider.bind(value=self.get_value)
        self.video_slider_lbl = CustomLabel( id = 'max_threshold_lbl')
        self.video_slider_lbl.text = "Frame number:" + str(int(self.video_slider.value))
        self.footer.add_widget(self.video_slider)
        self.footer.add_widget(self.video_slider_lbl)

    def visualise(self, trackbar_value, func = None):
        self.func = func
        sNumber = self.video_object.in_which_episode(int(trackbar_value))
        sFrame = trackbar_value
        current_segment = sNumber
        if self.video_object.paths_to_video_segments:
            self.cap = cv2.VideoCapture(self.video_object.paths_to_video_segments[sNumber])
        if self.video_object.paths_to_video_segments:
            start = self.video_object._episodes_start_end[sNumber][0]
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame - start)
        else:
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbar_value)
        ret, self.frame = self.cap.read()
        if ret == True:
            if hasattr(CHOSEN_VIDEO.video, 'resolution_reduction') and CHOSEN_VIDEO.video.resolution_reduction != 1:
                self.frame = cv2.resize(self.frame, None, fx = CHOSEN_VIDEO.video.resolution_reduction, fy = CHOSEN_VIDEO.video.resolution_reduction)
            if self.func is None:
                self.func = self.simple_visualisation
            self.func(self.frame)

    def simple_visualisation(self, frame):
        buf1 = cv2.flip(frame, 0)
        self.frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
        buf = buf1.tostring()
        textureFrame = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.display_layout.texture = textureFrame
        self.initImW = self.display_layout.width
        self.initImH = self.display_layout.height

    def get_value(self, instance, value):
        self.video_slider_lbl.text = "Frame number:" + str(int(value))
        self.visualise(value, func = self.func)
