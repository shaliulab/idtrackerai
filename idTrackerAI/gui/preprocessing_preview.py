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
from kivy.uix.switch import Switch
from kivy.uix.slider import Slider
from visualise_video import VisualiseVideo
from bkg_subtraction import BkgSubtraction
from kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process

import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing')
from segmentation import segmentVideo
from video_utils import blobExtractor
import numpy as np
import cv2

from video import Video

class PreprocessingPreview(BoxLayout):
    def __init__(self, chosen_video = None,
                deactivate_preprocessing = None,
                **kwargs):
        super(PreprocessingPreview, self).__init__(**kwargs)
        global CHOSEN_VIDEO, DEACTIVATE_PREPROCESSING
        CHOSEN_VIDEO = chosen_video
        DEACTIVATE_PREPROCESSING = deactivate_preprocessing
        CHOSEN_VIDEO.bind(chosen=self.do)
        self.container_layout = BoxLayout(orientation = 'vertical', size_hint = (.3, 1.))
        self.reduce_resolution_btn = Button(text = "Reduce resolution")
        self.bkg_subtractor = BkgSubtraction(orientation = 'vertical', chosen_video = CHOSEN_VIDEO)
        self.bkg_subtraction_label = CustomLabel(text = "background subtraction")
        self.bkg_subtractor_switch = Switch()
        self.ROI_label = CustomLabel(text = 'apply ROI')
        self.ROI_switch = Switch()
        self.container_layout.add_widget(self.reduce_resolution_btn)
        self.container_layout.add_widget(self.ROI_label)
        self.container_layout.add_widget(self.ROI_switch)
        self.container_layout.add_widget(self.bkg_subtraction_label)
        self.container_layout.add_widget(self.bkg_subtractor_switch)
        self.count_scrollup = 0
        self.scale = 1
        self.has_been_executed = False
        self.ROI_popup_text = CustomLabel(text='It seems that the ROI you are trying to apply corresponds to the entire frame. Please, go to the ROI selection tab to select and save a ROI')
        self.ROI_popup = Popup(title='ROI warning',
            content=self.ROI_popup_text,
            size_hint=(.5,.5))
        self.saving_popup = Popup(title='Saving',
            content=CustomLabel(text='wait ...'),
            size_hint=(.3,.3))
        self.saving_popup.bind(on_open=self.save_preproc)
        self.help_button_preprocessing = HelpButton()
        self.help_button_preprocessing.size_hint = (1.,1.)
        self.help_button_preprocessing.create_help_popup("Preprocessing",\
                                                "The aim of this part of the process is to separate the animals from the background, by setting the following parameters.\n1) Apply ROI: Allows to consider only a region of interest on the frame. Select it by using the table ROI selection.\n2) background subtraction: Perform background subtraction by computing a model of the background on the fly.\n3) Max\Min intensity: Set the maximum intensity used to separate the animals from the background.\n4) Max\Min area: Filter the blob by area.")

    def init_preproc_parameters(self):
        if CHOSEN_VIDEO.old_video is not None and CHOSEN_VIDEO.old_video._has_been_preprocessed == True:
            self.max_threshold = CHOSEN_VIDEO.old_video.max_threshold
            self.min_threshold = CHOSEN_VIDEO.old_video.min_threshold
            self.min_area = CHOSEN_VIDEO.old_video.min_area
            self.max_area = CHOSEN_VIDEO.old_video.max_area
            self.resolution_reduction = CHOSEN_VIDEO.old_video.resolution_reduction
        else:
            self.max_threshold = 165
            self.min_threshold = 0
            self.min_area = 100
            self.max_area = 1000
            self.resolution_reduction = 1.
            CHOSEN_VIDEO.video.resolution_reduction = 1.
            if CHOSEN_VIDEO.video._original_ROI is None:
                CHOSEN_VIDEO.video._original_ROI = np.ones( (CHOSEN_VIDEO.video.height, CHOSEN_VIDEO.video.width), dtype='uint8') * 255

        ###max_threshold
        self.max_threshold_slider = Slider(id = 'max_threhsold', min = 0, max = 255, value = self.max_threshold, step = 1)
        self.max_threshold_lbl = CustomLabel( id = 'max_threshold_lbl', text = "Max threshold:\n" + str(int(self.max_threshold_slider.value)))
        ###min_threshold
        self.min_threshold_slider = Slider(id='min_threshold_slider', min = 0, max = 255, value = self.min_threshold, step = 1)
        self.min_threshold_lbl = CustomLabel(id='min_threshold_lbl', text = "Min threshold:\n" + str(int(self.min_threshold_slider.value)))
        ###max_area label
        self.max_area_slider = Slider(id='max_area_slider', min = 0, max = 60000, value = self.max_area, step = 1)
        self.max_area_lbl = CustomLabel(id='max_area_lbl', text = "Max area:\n" + str(int(self.max_area_slider.value)))
        ###min_area
        self.min_area_slider = Slider(id='min_area_slider', min = 0, max = 1000, value = self.min_area, step = 1)
        self.min_area_lbl = CustomLabel(id='min_area_lbl', text = "Min area:\n" + str(int(self.min_area_slider.value)))
        self.w_list = [ self.max_threshold_lbl, self.max_threshold_slider,
                        self.min_threshold_lbl, self.min_threshold_slider,
                        self.max_area_lbl, self.max_area_slider,
                        self.min_area_lbl, self.min_area_slider]
        self.add_widget_list()
        self.max_threshold_slider.bind(value=self.update_max_th_lbl)
        self.min_threshold_slider.bind(value=self.update_min_th_lbl)
        self.max_area_slider.bind(value=self.update_max_area_lbl)
        self.min_area_slider.bind(value=self.update_min_area_lbl)
        self.container_layout.add_widget(self.help_button_preprocessing)
        self.segment_video_btn = Button(text = "Segment video")

    def do(self, *args):
        if CHOSEN_VIDEO.video is not None and CHOSEN_VIDEO.video.video_path is not None:
            self.init_preproc_parameters()
            self.create_resolution_reduction_popup()
            self.res_red_input.bind(on_text_validate = self.on_enter_res_red_coeff)
            self.reduce_resolution_btn.bind(on_press = self.open_resolution_reduction_popup)
            self.ROI_switch.bind(active = self.apply_ROI)
            self.ROI_switch.active = False
            self.bkg_subtractor_switch.active = False
            self.bkg_subtractor_switch.bind(active = self.apply_bkg_subtraction)
            self.bkg = CHOSEN_VIDEO.video.bkg
            self.ROI = CHOSEN_VIDEO.video.ROI
            self.init_segment_zero()
            self.has_been_executed = True

    def create_resolution_reduction_popup(self):
        self.res_red_popup_container = BoxLayout()
        self.res_red_coeff = BoxLayout(orientation="vertical")
        self.res_red_label = CustomLabel(text='Type the resolution reduction coefficient (0.5 will reduce by half).')
        self.res_red_coeff.add_widget(self.res_red_label)
        self.res_red_input = TextInput(text ='', multiline=False)
        self.res_red_coeff.add_widget(self.res_red_input)
        self.res_red_popup_container.add_widget(self.res_red_coeff)
        self.res_red_popup = Popup(title = 'Resolution reduction',
                            content = self.res_red_popup_container,
                            size_hint = (.4, .4))

    def open_resolution_reduction_popup(self, *args):
        self.res_red_popup.open()

    def on_enter_res_red_coeff(self, *args):
        CHOSEN_VIDEO.video.resolution_reduction = float(self.res_red_input.text)
        self.resolution_reduction = CHOSEN_VIDEO.video.resolution_reduction
        self.res_red_popup.dismiss()
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)


    def apply_ROI(self, instance, active):
        # print("applying ROI")
        CHOSEN_VIDEO.video._apply_ROI = active
        if active  == True:
            num_valid_pxs_in_ROI = len(sum(np.where(CHOSEN_VIDEO.video.ROI == 255)))
            num_pxs_in_frame = CHOSEN_VIDEO.video.height * CHOSEN_VIDEO.video.width
            self.ROI_is_trivial = num_pxs_in_frame == num_valid_pxs_in_ROI

            if CHOSEN_VIDEO.video.ROI is not None and not self.ROI_is_trivial:
                self.ROI = CHOSEN_VIDEO.video.ROI
            elif self.ROI_is_trivial:
                self.ROI_popup.open()
                instance.active = False
                CHOSEN_VIDEO.apply_ROI = False
        elif active == False:
            self.ROI = np.ones((CHOSEN_VIDEO.video.height, CHOSEN_VIDEO.video.width) ,dtype='uint8') * 255
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def apply_bkg_subtraction(self, instance, active):
        CHOSEN_VIDEO.video._subtract_bkg = active
        if CHOSEN_VIDEO.video.subtract_bkg == True:
            if CHOSEN_VIDEO.old_video.bkg is not None:
                CHOSEN_VIDEO.video._bkg = CHOSEN_VIDEO.old_video.bkg
            elif CHOSEN_VIDEO.video.bkg is None:
                self.bkg_subtractor.computing_popup.open()
                self.bkg_subtractor.saving_popup.open()
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def add_widget_list(self):
        for w in self.w_list:
            self.container_layout.add_widget(w)

    def update_max_th_lbl(self,instance, value):
        self.max_threshold_lbl.text = "Max threshold:\n" +  str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def update_min_th_lbl(self,instance, value):
        self.min_threshold_lbl.text = "Min threshold:\n" + str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def update_max_area_lbl(self,instance, value):
        self.max_area_lbl.text = "Max area:\n" + str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def update_min_area_lbl(self,instance, value):
        self.min_area_lbl.text = "Min area:\n" + str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def save_preproc_params(self):
        self.saving_popup.open()
        CHOSEN_VIDEO.video._max_threshold = self.max_threshold_slider.value
        CHOSEN_VIDEO.video._min_threshold = self.min_threshold_slider.value
        CHOSEN_VIDEO.video._min_area = self.min_area_slider.value
        CHOSEN_VIDEO.video._max_area = self.max_area_slider.value
        CHOSEN_VIDEO.video.resolution_reduction = self.resolution_reduction

    def save_preproc(self, *args):
        CHOSEN_VIDEO.video.save()
        self.saving_popup.dismiss()

    def init_segment_zero(self):
        self.visualiser = VisualiseVideo(chosen_video = CHOSEN_VIDEO)
        self.add_widget(self.container_layout)
        self.add_widget(self.visualiser)
        self.visualiser.visualise_video(CHOSEN_VIDEO.video, func = self.show_preprocessing)
        self.currentSegment = 0
        self.number_of_detected_blobs = []

    def show_preprocessing(self, frame):
        if len(frame.shape) > 2:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
        else:
            self.frame = frame
        if hasattr(self, 'area_bars'):
            self.visualiser.remove_widget(self.area_bars)
        avIntensity = np.float32(np.mean(self.frame))
        self.av_frame = self.frame / avIntensity
        self.segmented_frame = segmentVideo(self.av_frame,
                                            int(self.min_threshold_slider.value),
                                            int(self.max_threshold_slider.value),
                                            CHOSEN_VIDEO.video.bkg,
                                            CHOSEN_VIDEO.video.ROI,
                                            self.bkg_subtractor_switch.active)
        boundingBoxes, miniFrames, _, areas, _, goodContours, _ = blobExtractor(self.segmented_frame,
                                                                        self.frame,
                                                                        int(self.min_area_slider.value),
                                                                        int(self.max_area_slider.value))
        if hasattr(self, "number_of_detected_blobs"):
            self.number_of_detected_blobs.append(len(areas))
        fig, ax = plt.subplots(1)
        width = .5
        plt.bar(range(len(areas)), areas, width)
        plt.axhline(np.mean(areas), color = 'k', linewidth = .2)
        ax.set_xlabel('blob')
        ax.set_ylabel('area')
        ax.set_facecolor((.345, .345, .345))
        self.area_bars = FigureCanvasKivyAgg(fig)
        self.area_bars.size_hint = (1., .2)
        self.visualiser.add_widget(self.area_bars)
        cv2.drawContours(self.frame, goodContours, -1, color=255, thickness = -1)
        if self.count_scrollup != 0:
            self.dst = cv2.warpAffine(self.frame, self.M, (self.frame.shape[1], self.frame.shape[1]))
            buf1 = cv2.flip(self.dst,0)
        else:
            buf1 = cv2.flip(self.frame, 0)
        buf = buf1.tostring()
        textureFrame = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='luminance')
        textureFrame.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
        self.visualiser.display_layout.texture = textureFrame

    def fromShowFrameToTexture(self, coords):
        """Maps coordinate in visualiser.display_layout (the image whose texture is the frame) to
        the coordinates of the original image
        """
        coords = np.asarray(coords)
        origFrameW = CHOSEN_VIDEO.video.width
        origFrameH = CHOSEN_VIDEO.video.height
        return coords

    def on_touch_down(self, touch):
        self.touches = []
        if self.parent is not None and self.visualiser.display_layout.collide_point(*touch.pos):
            if touch.button == 'scrollup':
                self.count_scrollup += 1
                coords = self.fromShowFrameToTexture(touch.pos)
                rows,cols = self.frame.shape
                self.scale = 1.5 * self.count_scrollup
                self.M = cv2.getRotationMatrix2D((coords[0],coords[1]),0,self.scale)
                self.dst = cv2.warpAffine(self.frame,self.M,(cols,rows))
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]),
                                            colorfmt='luminance')
                textureFrame.blit_buffer(buf,
                                        colorfmt='luminance', bufferfmt='ubyte')
                self.visualiser.display_layout.texture = textureFrame
            elif touch.button == 'scrolldown':
                coords = self.fromShowFrameToTexture(touch.pos)
                rows,cols = self.frame.shape
                self.dst = self.frame
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]),
                                            colorfmt='luminance')
                textureFrame.blit_buffer(buf,
                                        colorfmt='luminance',
                                        bufferfmt='ubyte')
                self.visualiser.display_layout.texture = textureFrame
                self.count_scrollup = 0
        else:
            self.scale = 1
            self.disable_touch_down_outside_collided_widget(touch)

    def disable_touch_down_outside_collided_widget(self, touch):
        return super(PreprocessingPreview, self).on_touch_down(touch)
