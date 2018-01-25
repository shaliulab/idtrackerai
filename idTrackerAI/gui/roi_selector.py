from __future__ import absolute_import, division, print_function
import kivy

from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.splitter import Splitter
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
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
        self.roi_shape_label = CustomLabel(font_size = 14, text = "Select ROI shape")
        self.btn_rectangular = ToggleButton(text = "Rectangle", group = "ROI_shape")
        self.btn_rectangular.state = 'down'
        self.btn_circular = ToggleButton(text = "Circle", group = "ROI_shape")
        self.separator = BoxLayout()
        self.btn_load_roi = Button(text = "load ROIs")
        self.btn_save_roi = Button(text = "save ROIs")
        self.btn_clear_roi = Button(text = "clear last ROI")
        self.btn_no_roi = Button(text = "do not use any ROI")
        w_list = [self.btn_load_roi, self.btn_save_roi,
                self.btn_clear_roi, self.btn_no_roi, self.separator,
                self.roi_shape_label, self.btn_rectangular, self.btn_circular]
        [self.control_panel.add_widget(w) for w in w_list]
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
            ud = touch.ud
            with self.visualiser.display_layout.canvas:
                Color(1, 0, 0, .2)
                self.circle = Line(circle = (touch.pos[0], touch.pos[1], 5, 0, 360, 10))
                if self.btn_rectangular.state == "down":
                    ud['lines'] = Rectangle(pos=(self.touches[0][0], self.touches[0][1]), size=(0, 0))
                elif self.btn_circular.state == "down":
                    ud['lines'] = Ellipse(pos=(self.touches[0][0], self.touches[0][1]),
                                            size=(0, 0))
                touch.grab(self)
        else:
            self.disable_touch_down_outside_collided_widget(touch)

    def on_touch_move(self, touch):
        if self.visualiser.display_layout.collide_point(*touch.pos):
            self.touches.append(touch.pos)
            with self.visualiser.display_layout.canvas:
                ud = touch.ud
                rect = np.asarray([self.touches[0], self.touches[-1]])
                sorted(rect, key=lambda x:x[1], reverse=True)
                rectS = np.diff(rect, axis=0)[0]
                if self.btn_rectangular.state == "down":
                    ud['lines'].pos = rect[0][0], rect[0][1]
                    ud['lines'].size = rectS[0],rectS[1]
                elif self.btn_circular.state == "down":
                    ax_x = rectS[0]
                    ax_y = rectS[1]
                    c_x = (rect[0][0] + rect[1][0]) / 2
                    c_y = (rect[0][1] + rect[1][1]) / 2
                    ud['lines'].pos = c_x - ax_x / 2, c_y - ax_y / 2
                    ud['lines'].size = ax_x, ax_y
        else:
            self.disable_touch_down_outside_collided_widget(touch)

    def disable_touch_down_outside_collided_widget(self, touch):
        return super(ROISelector, self).on_touch_down(touch)

    @staticmethod
    def affine_transform(point, translation, scale):
        return np.round(np.dot(scale, (point.T + translation))).astype('int')

    @staticmethod
    def inverse_y_axis(point, height):
        point[1] = height - point[1]
        return point

    def on_touch_up(self, touch):
        if self.visualiser.display_layout.collide_point(*touch.pos) and len(self.touches) > 0:
            # try:
            touch.ungrab(self)
            ud = touch.ud
            self.visualiser.display_layout.canvas.remove(ud['lines'])
            self.visualiser.display_layout.canvas.remove(self.circle)
            self.touches.append(touch.pos)
            rect = np.asarray([self.touches[0], self.touches[-1]])
            sorted(rect, key=lambda x:x[1], reverse=True)
            rectS = np.diff(rect, axis=0)[0]
            with self.visualiser.display_layout.canvas:
                Color(1, 1, 0,.5)
                if self.btn_rectangular.state == "down":
                    self.cur_ROI = Rectangle(pos=(rect[0][0], rect[0][1]), size=(rectS[0],rectS[1]))
                elif self.btn_circular.state == "down":
                    ax_x = rectS[0]
                    ax_y = rectS[1]
                    c_x = (rect[0][0] + rect[1][0]) / 2
                    c_y = (rect[0][1] + rect[1][1]) / 2
                    self.cur_ROI = Ellipse(pos=(c_x - ax_x / 2, c_y - ax_y / 2),
                                        size=(ax_x, ax_y))

                self.ROIs.append(self.cur_ROI)
                #scale
                ratioH = self.visualiser.display_layout.texture.height / self.visualiser.display_layout.height
                ratioW = self.visualiser.display_layout.texture.width / self.visualiser.display_layout.width
                scale = np.asarray([[ratioW, 0], [0, ratioH]])
                #translate
                translation = np.asarray([0, - self.visualiser.footer.height])
                if self.btn_rectangular.state == "down":
                    print("out rect")
                    p1_ = self.affine_transform(rect[0], translation, scale)
                    p2_ = self.affine_transform(rect[1], translation, scale)
                    p1_ = self.inverse_y_axis(p1_, self.visualiser.display_layout.texture.height)
                    p2_ = self.inverse_y_axis(p2_, self.visualiser.display_layout.texture.height)
                    self.ROIOut.append([tuple(p1_), tuple(p2_)])
                elif self.btn_circular.state == "down":
                    print("out ellipse")
                    c = np.asarray([c_x, c_y])
                    c_ = self.affine_transform(c, translation, scale)
                    c_ = self.inverse_y_axis(c_, self.visualiser.display_layout.texture.height)
                    semi_ax_x = int(abs(c_[0] - self.affine_transform(np.asarray([c_x + ax_x / 2, c_y]), translation, scale)[0]))
                    semi_ax_y = int(abs(c_[1] - self.affine_transform(np.asarray([c_x, c_y + ax_y / 2]), translation, scale)[1]))
                    self.ROIOut.append([tuple(c_), semi_ax_x, semi_ax_y])
                print(len(self.ROIOut))
                self.touches = []
            # except:
            #     print('stay on the figure to draw a ROI')

    def delete_ROI(self, *args):
        try:
            rect = self.ROIs[-1]
            self.ROIs = self.ROIs[:-1] #clear from the app ROIs collection
            self.ROIOut = self.ROIOut[:-1] #clear from the cv2 ROIs collection
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
        print("in save ROI")
        print("number of ROIs ", len(self.ROIOut))
        if len(self.ROIOut) > 0:
            self.ROIcv2 = np.zeros_like(self.visualiser.frame, dtype='uint8')
            print("ROI saving")
            for p in self.ROIOut:
                if len(p) == 2:
                    print("rectangle")
                    cv2.rectangle(self.ROIcv2, p[0], p[1], 255, -1)
                elif len(p) == 3:
                    print("ellipse")
                    angle = 90
                    cv2.ellipse(self.ROIcv2, p[0], (p[1], p[2]),angle,0,360,255,-1)
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
