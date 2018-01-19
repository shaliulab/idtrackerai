from __future__ import absolute_import, division, print_function
import kivy

from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.graphics import *
from kivy.graphics.transformation import Matrix
from visualise_video import VisualiseVideo
from kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process
import os
import sys
sys.path.append('../')
sys.path.append('../utils')
import numpy as np
import cv2

from video import Video

class ROISelector(BoxLayout):
    def __init__(self, chosen_video = None,
                    deactivate_roi = None,
                    **kwargs):
        super(ROISelector, self).__init__(**kwargs)
        global CHOSEN_VIDEO, DEACTIVATE_ROI
        CHOSEN_VIDEO = chosen_video
        DEACTIVATE_ROI = deactivate_roi
        self.ROIs = [] #store rectangles on the GUI
        self.ROIOut  = [] #pass them to opencv
        self.touches = [] #store touch events on the figure
        self.control_panel = BoxLayout(orientation = "vertical")
        self.control_panel.size_hint = (.26, 1.)
        self.btn_load_roi = Button(text = "load ROIs")
        self.btn_save_roi = Button(text = "save ROIs")
        self.btn_clear_roi = Button(text = "clear last ROI")
        self.btn_no_roi = Button(text = "do not use any ROI")
        self.control_panel.add_widget(self.btn_load_roi)
        self.control_panel.add_widget(self.btn_save_roi)
        self.control_panel.add_widget(self.btn_clear_roi)
        self.control_panel.add_widget(self.btn_no_roi)
        self.btn_save_roi.bind(on_press = self.save_ROI)
        self.btn_load_roi.bind(on_press = self.load_ROI)
        self.btn_no_roi.bind(on_press = self.no_ROI)
        self.btn_clear_roi.bind(on_press = self.delete_ROI)
        self.has_been_executed = False
        global CHOSEN_VIDEO
        CHOSEN_VIDEO.bind(chosen=self.do)

    def do(self, *args):
        if CHOSEN_VIDEO.video.video_path is not None:
            CHOSEN_VIDEO.video.resolution_reduction = 1
            self.visualiser = VisualiseVideo(chosen_video = CHOSEN_VIDEO)
            self.add_widget(self.visualiser)
            self.add_widget(self.control_panel)
            self.window = Window
            self.window.bind(on_resize=self.updateROIs)
            self.visualiser.visualise_video(CHOSEN_VIDEO.video)
            if CHOSEN_VIDEO.old_video is not None and CHOSEN_VIDEO.old_video.ROI is not None:
                self.btn_load_roi.disabled = not hasattr(CHOSEN_VIDEO.old_video, "ROI")
            else:
                self.btn_load_roi.disabled = True
            self.has_been_executed = True

    def on_touch_down(self, touch):
        self.touches = []
        if self.visualiser.display_layout.collide_point(*touch.pos):
            self.touches.append(touch.pos)
        else:
            self.disable_touch_down_outside_collided_widget(touch)

    def disable_touch_down_outside_collided_widget(self, touch):
        return super(ROISelector, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if self.visualiser.display_layout.collide_point(*touch.pos) and len(self.touches) > 0:
            try:
                self.touches.append(touch.pos)
                rect = np.asarray([self.touches[0], self.touches[-1]])
                sorted(rect, key=lambda x:x[1], reverse=True)
                rectS = np.diff(rect, axis=0)[0]
                with self.visualiser.display_layout.canvas:
                    Color(1, 1, 0,.5)
                    self.rect = Rectangle(pos=(rect[0][0], rect[0][1]), size=(rectS[0],rectS[1]))
                    self.ROIs.append(self.rect)
                    #scale
                    ratioH = self.visualiser.display_layout.texture.height / self.visualiser.display_layout.height
                    ratioW = self.visualiser.display_layout.texture.width / self.visualiser.display_layout.width
                    scale = np.asarray([[ratioW, 0], [0, ratioH]])
                    #translate
                    translation = np.asarray([0, - self.visualiser.footer.height])
                    #apply transform
                    p1_ = np.round(np.dot(scale, (rect[0].T + translation))).astype('int')
                    p2_ = np.round(np.dot(scale, (rect[1].T + translation))).astype('int')
                    # inverse y-axis
                    p1_[1] = self.visualiser.display_layout.texture.height - p1_[1]
                    p2_[1] = self.visualiser.display_layout.texture.height - p2_[1]
                    self.ROIOut.append([tuple(p1_), tuple(p2_)])
                self.touches = []
            except:
                print('stay on the figure to draw a ROI')

    def delete_ROI(self, *args):
        try:
            rect = self.ROIs[-1] #clear from the app ROIs collection
            self.ROIs = self.ROIs[:-1] #clear from the cv2 ROIs collection
            self.ROIOut = self.ROIOut[:-1]
            self.visualiser.display_layout.canvas.remove(rect) #clear from the image in the visualisation
        except:
            print('Select one ROI first')

    def updateROIs(self, window, width, height):
        self.cur_image_height = self.visualiser.display_layout.height
        self.cur_image_width = self.visualiser.display_layout.width
        if not (self.visualiser.initImH == 100 and self.visualiser.initImW == 100):
            wRatio = abs(self.cur_image_width / self.visualiser.initImW)
            hRatio = abs(self.cur_image_height / self.visualiser.initImH)

            for rect in self.ROIs:
                rect.pos = (rect.pos[0] * wRatio, rect.pos[1] * hRatio)
                rect.size = (rect.size[0] * wRatio, rect.size[1] * hRatio)

        self.visualiser.initImH = self.cur_image_height
        self.visualiser.initImW = self.cur_image_width

    def save_ROI(self, *args):
        if len(self.ROIOut) > 0:
            self.ROIcv2 = np.zeros_like(self.visualiser.frame, dtype='uint8')
            for p in self.ROIOut:
                cv2.rectangle(self.ROIcv2, p[0], p[1], 255, -1)
        else:
            self.ROIcv2 = np.ones_like(self.visualiser.frame, dtype='uint8') * 255
        CHOSEN_VIDEO.video._original_ROI = self.ROIcv2
        CHOSEN_VIDEO.video.save()

    def no_ROI(self, *args):
        CHOSEN_VIDEO.video.ROI = np.ones_like(self.visualiser.frame ,dtype='uint8') * 255
        CHOSEN_VIDEO.apply_ROI = False
        CHOSEN_VIDEO.video.save()

    def load_ROI(self, *args):
        CHOSEN_VIDEO.video.ROI = CHOSEN_VIDEO.old_video.ROI
