from __future__ import absolute_import, division, print_function
import kivy

from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.properties import BooleanProperty
from kivy.event import EventDispatcher
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.tabbedpanel import TabbedPanelItem
from kivy.uix.tabbedpanel import TabbedPanelHeader
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.scatter import Scatter
from kivy.uix.popup import Popup
from kivy.uix.switch import Switch
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import *
from kivy.graphics.transformation import Matrix
from kivy.clock import Clock
from kivy.config import Config
from kivy.uix.filechooser import FileChooserListView
from kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process
from select_file import SelectFile
from validator import Validator
from individual_validator import IndividualValidator
from visualise_video import VisualiseVideo

import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

import os
import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing')
sys.path.append('../groundtruth_utils')
import numpy as np
import logging.config
import yaml
import cv2

from video import Video
from py_utils import getExistentFiles, get_spaced_colors_util
from video_utils import computeBkg, blobExtractor
from segmentation import segmentVideo
from list_of_blobs import ListOfBlobs
from list_of_fragments import ListOfFragments
from generate_groundtruth import generate_groundtruth
from compute_groundtruth_statistics import get_accuracy_wrt_groundtruth

"""
Init variables
    PROCESSES: list of strings.
        list of all the processes that can be saved and loaded while
        tracking a video
    THRESHOLD_ACCEPTABLE_ACCUMULATION: float (0,1)
        minimum ratio of images to be accumulated in order to consider a
        protocol succesfull
    RESTORE_CRITERION: string ['last', 'best']
        criterion used during accumulation in order to choose a model. best
        will save the model that realises the minimum of the loss in validation,
        whilst 'last' will simply save the last one according to the
        early stopping criteria see
            network/identification_model/stop_training_criteria.py
    VEL_PERCENTILE: integer [0, 100]
        percentile used to compute the maximal accpetable individual velocity
"""
PROCESSES = ['use_previous_knowledge_transfer_decision', 'preprocessing',
            'first_accumulation', 'pretraining', 'second_accumulation',
            'assignment', 'solving_duplications', 'crossings', 'trajectories',
            'trajectories_wo_gaps']
THRESHOLD_ACCEPTABLE_ACCUMULATION = .9
RESTORE_CRITERION = 'last'
VEL_PERCENTILE = 99

def setup_logging(
    default_path='logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG',
    path_to_save_logs = './',
    video_object = None):
    """Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        if os.path.exists(path_to_save_logs) and video_object is not None:
            video_object.logs_folder = os.path.join(path_to_save_logs, 'log_files')
            if not os.path.isdir(video_object.logs_folder):
                os.makedirs(video_object.logs_folder)
            config['handlers']['info_file_handler']['filename'] = os.path.join(video_object.logs_folder, 'info.log')
            config['handlers']['error_file_handler']['filename'] = os.path.join(video_object.logs_folder, 'error.log')
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    logger = logging.getLogger(__name__)
    logger.propagate = True
    logger.setLevel("INFO")
    return logger

"""
Start kivy classes
"""
# class ROISelector(BoxLayout):
#     def __init__(self,**kwargs):
#         super(ROISelector, self).__init__(**kwargs)
#         self.orientation = "vertical"
#         self.ROIs = [] #store rectangles on the GUI
#         self.ROIOut  = [] #pass them to opencv
#         self.touches = [] #store touch events on the figure
#         self.footer = BoxLayout()
#         self.footer.size_hint = (1.,.1)
#         self.btn_load_roi = Button(text = "load ROIs")
#         self.btn_save_roi = Button(text = "save ROIs")
#         self.btn_clear_roi = Button(text = "clear last ROI")
#         self.btn_no_roi = Button(text = "do not use any ROI")
#         self.footer.add_widget(self.btn_load_roi)
#         self.footer.add_widget(self.btn_save_roi)
#         self.footer.add_widget(self.btn_clear_roi)
#         self.footer.add_widget(self.btn_no_roi)
#         self.btn_save_roi.bind(on_press = self.save_ROI)
#         self.btn_load_roi.bind(on_press = self.load_ROI)
#         self.btn_no_roi.bind(on_press = self.no_ROI)
#         self.btn_clear_roi.bind(on_press = self.delete_ROI)
#         global CHOSEN_VIDEO
#         CHOSEN_VIDEO.bind(chosen=self.do)
#
#     def do(self, *args):
#         if hasattr(CHOSEN_VIDEO.video, "video_path") and CHOSEN_VIDEO.video.video_path is not None:
#             self.visualiser = VisualiseVideo()
#             self.add_widget(self.visualiser)
#             self.add_widget(self.footer)
#             self.window = Window
#             self.window.bind(on_resize=self.updateROIs)
#             self.video_object = CHOSEN_VIDEO.video
#             self.visualiser.visualise_video(self.video_object)
#             if hasattr(CHOSEN_VIDEO, "old_video") and CHOSEN_VIDEO.old_video.ROI is not None:
#                 self.btn_load_roi.disabled = not hasattr(CHOSEN_VIDEO.old_video, "ROI")
#             else:
#                 self.btn_load_roi.disabled = True
#
#     def on_touch_down(self, touch):
#         # print("touch down dispatch")
#         self.touches = []
#         if self.visualiser.display_layout.collide_point(*touch.pos):
#             self.touches.append(touch.pos)
#         else:
#             self.disable_touch_down_outside_collided_widget(touch)
#
#     def disable_touch_down_outside_collided_widget(self, touch):
#         return super(ROISelector, self).on_touch_down(touch)
#
#     def on_touch_up(self, touch):
#         if self.visualiser.display_layout.collide_point(*touch.pos) and len(self.touches) > 0:
#             try:
#                 self.touches.append(touch.pos)
#                 rect = [self.touches[0], self.touches[-1]]
#                 sorted(rect, key=lambda x:x[1], reverse=True)
#                 rectS = np.diff(rect, axis=0)[0]
#                 with self.visualiser.display_layout.canvas:
#                     Color(1, 1, 0,.5)
#                     self.rect = Rectangle(pos=(rect[0][0], rect[0][1]), size=(rectS[0],rectS[1]))
#                     self.ROIs.append(self.rect)
#                     ratioH = self.visualiser.display_layout.height / self.visualiser.display_layout.texture.height
#                     ratioW = self.visualiser.display_layout.width / self.visualiser.display_layout.texture.width
#                     newRectP1 = (self.rect.pos[0] / ratioW, (self.rect.pos[0] + self.rect.size[0]) / ratioW)
#                     newRectP2 = (self.rect.pos[1] / ratioH, (self.rect.pos[1] + self.rect.size[1]) / ratioH)
#                     point1 = (int(newRectP1[0]), int(self.visualiser.frame.shape[1]-newRectP2[0]))
#                     point2 = (int(newRectP1[1]), int(self.visualiser.frame.shape[1]-newRectP2[1]))
#                     self.ROIOut.append([point1,point2])
#
#                 self.touches = []
#             except:
#                 print('stay on the figure to draw a ROI')
#
#     def delete_ROI(self, *args):
#         try:
#             rect = self.ROIs[-1] #clear from the app ROIs collection
#             self.ROIs = self.ROIs[:-1] #clear from the cv2 ROIs collection
#             self.ROIOut = self.ROIOut[:-1]
#             self.visualiser.display_layout.canvas.remove(rect) #clear from the image in the visualisation
#         except:
#             print('Select one ROI first')
#
#     def updateROIs(self, window, width, height):
#         self.cur_image_height = self.visualiser.display_layout.height
#         self.cur_image_width = self.visualiser.display_layout.width
#         if not (self.visualiser.initImH == 100 and self.visualiser.initImW == 100):
#             wRatio = abs(self.cur_image_width / self.visualiser.initImW)
#             hRatio = abs(self.cur_image_height / self.visualiser.initImH)
#
#             for rect in self.ROIs:
#                 rect.pos = (rect.pos[0] * wRatio, rect.pos[1] * hRatio)
#                 rect.size = (rect.size[0] * wRatio, rect.size[1] * hRatio)
#
#         self.visualiser.initImH = self.cur_image_height
#         self.visualiser.initImW = self.cur_image_width
#
#     def save_ROI(self, *args):
#         # print("saving ROI")
#         if len(self.ROIOut) > 0:
#             self.ROIcv2 = np.zeros_like(self.visualiser.frame,dtype='uint8')
#             for p in self.ROIOut:
#                 # print("adding rectangles to ROI")
#                 # print("rect ", p)
#                 cv2.rectangle(self.ROIcv2, p[0], p[1], 255, -1)
#         CHOSEN_VIDEO.video.ROI = self.ROIcv2
#         CHOSEN_VIDEO.video.save()
#
#     def no_ROI(self, *args):
#         CHOSEN_VIDEO.video.ROI = np.ones_like(self.visualiser.frame ,dtype='uint8') * 255
#         CHOSEN_VIDEO.apply_ROI = False
#         CHOSEN_VIDEO.video.save()
#
#     def load_ROI(self, *args):
#         CHOSEN_VIDEO.video.ROI = CHOSEN_VIDEO.old_video.ROI
#

# class PreprocessingPreview(BoxLayout):
#     def __init__(self, **kwargs):
#         super(PreprocessingPreview, self).__init__(**kwargs)
#         #get video path information and bind it
#         self.visualiser = VisualiseVideo()
#         global CHOSEN_VIDEO
#         CHOSEN_VIDEO.bind(chosen=self.do)
#         self.container_layout = BoxLayout(orientation = 'vertical', size_hint = (.3, 1.))
#         self.add_widget(self.container_layout)
#         # bkg_subtraction
#         self.bkg_subtractor = BkgSubtraction(orientation = 'vertical')
#         self.container_layout.add_widget(self.bkg_subtractor)
#         #bkg sub label
#         self.bkg_subtraction_label = Label(text = 'background subtraction')
#         self.bkg_subtraction_label.text_size = self.bkg_subtraction_label.size
#         self.bkg_subtraction_label.size = self.bkg_subtraction_label.texture_size
#         self.bkg_subtraction_label.font_size = 16
#         self.bkg_subtraction_label.halign =  "center"
#         self.bkg_subtraction_label.valign = "middle"
#         #bkg sub switch
#         self.bkg_subtractor_switch = Switch()
#         #ROI label
#         self.ROI_label = Label(text = 'apply ROI')
#         self.ROI_label.text_size = self.ROI_label.size
#         self.ROI_label.size = self.ROI_label.texture_size
#         self.ROI_label.font_size = 16
#         self.ROI_label.halign =  "center"
#         self.ROI_label.valign = "middle"
#         self.container_layout.add_widget(self.ROI_label)
#         #ROI switch
#         self.ROI_switch = Switch()
#         self.container_layout.add_widget(self.ROI_switch)
#         self.container_layout.add_widget(self.bkg_subtraction_label)
#         self.container_layout.add_widget(self.bkg_subtractor_switch)
#         self.count_scrollup = 0
#         self.scale = 1
#         self.ROI_popup_text = Label(text='It seems that the ROI you are trying to apply corresponds to the entire frame. Please, go to the ROI selection tab to select and save a ROI')
#         self.ROI_popup = Popup(title='ROI warning',
#             content=self.ROI_popup_text,
#             size_hint=(.5,.5))
#         self.ROI_popup_text.text_size = self.ROI_popup.size
#         self.ROI_popup_text.size = self.ROI_popup_text.texture_size
#         self.ROI_popup_text.font_size = 16
#         self.ROI_popup_text.halign =  "center"
#         self.ROI_popup_text.valign = "middle"
#         self.saving_popup = Popup(title='Saving',
#             content=Label(text='wait ...'),
#             size_hint=(.3,.3))
#         self.ROI_popup_text.bind(size=lambda s, w: s.setter('text_size')(s, w))
#         self.saving_popup.bind(on_open=self.save_preproc)
#
#     def init_preproc_parameters(self):
#         if hasattr(CHOSEN_VIDEO, "old_video") and CHOSEN_VIDEO.old_video._has_been_preprocessed == True:
#             self.max_threshold = CHOSEN_VIDEO.old_video.max_threshold
#             self.min_threshold = CHOSEN_VIDEO.old_video.min_threshold
#             self.min_area = CHOSEN_VIDEO.old_video.min_area
#             self.max_area = CHOSEN_VIDEO.old_video.max_area
#         else:
#             self.max_threshold = 165
#             self.min_threshold = 0
#             self.min_area = 100
#             self.max_area = 1000
#         ###max_threshold
#         self.max_threshold_slider = Slider(id = 'max_threhsold')
#         self.max_threshold_slider.min = 0
#         self.max_threshold_slider.max = 255
#         self.max_threshold_slider.value = self.max_threshold
#         self.max_threshold_slider.step = 1
#
#         self.max_threshold_lbl = Label( id = 'max_threshold_lbl')
#         self.max_threshold_lbl.text = "Max threshold:\n" + str(int(self.max_threshold_slider.value))
#         self.max_threshold_lbl.text_size = self.max_threshold_lbl.size
#         self.max_threshold_lbl.size = self.max_threshold_lbl.texture_size
#         self.max_threshold_lbl.font_size = 16
#         self.max_threshold_lbl.halign =  "center"
#         self.max_threshold_lbl.valign = "middle"
#
#         ###min_threshold
#         self.min_threshold_slider = Slider(id='min_threshold_slider')
#         self.min_threshold_slider.min = 0
#         self.min_threshold_slider.max = 255
#         self.min_threshold_slider.value = self.min_threshold
#         self.min_threshold_slider.step = 1
#
#         self.min_threshold_lbl = Label(id='min_threshold_lbl')
#         self.min_threshold_lbl.text = "Min threshold:\n" + str(int(self.min_threshold_slider.value))
#         self.min_threshold_lbl.text_size = self.min_threshold_lbl.size
#         self.min_threshold_lbl.size = self.min_threshold_lbl.texture_size
#         self.min_threshold_lbl.font_size = 16
#         self.min_threshold_lbl.halign =  "center"
#         self.min_threshold_lbl.valign = "middle"
#         ###max_area label
#         self.max_area_slider = Slider(id='max_area_slider')
#         self.max_area_slider.min = 0
#         self.max_area_slider.max = 60000
#         self.max_area_slider.value = self.max_area
#         self.max_area_slider.step = 1
#
#         self.max_area_lbl = Label(id='max_area_lbl')
#         self.max_area_lbl.text = "Max area:\n" + str(int(self.max_area_slider.value))
#         self.max_area_lbl.text_size = self.max_area_lbl.size
#         self.max_area_lbl.size = self.max_area_lbl.texture_size
#         self.max_area_lbl.font_size = 16
#         self.max_area_lbl.halign =  "center"
#         self.max_area_lbl.valign = "middle"
#         ###min_area
#         self.min_area_slider = Slider(id='min_area_slider')
#         self.min_area_slider.min = 0
#         self.min_area_slider.max = 1000
#         self.min_area_slider.value = self.min_area
#         self.min_area_slider.step = 1
#
#         self.min_area_lbl = Label(id='min_area_lbl')
#         self.min_area_lbl.text = "Min area:\n" + str(int(self.min_area_slider.value))
#         self.min_area_lbl.text_size = self.min_area_lbl.size
#         self.min_area_lbl.size = self.min_area_lbl.texture_size
#         self.min_area_lbl.font_size = 16
#         self.min_area_lbl.halign =  "center"
#         self.min_area_lbl.valign = "middle"
#
#         self.w_list = [self.max_threshold_lbl, self.max_threshold_slider,
#                         self.min_threshold_lbl, self.min_threshold_slider,
#                         self.max_area_lbl, self.max_area_slider,
#                         self.min_area_lbl, self.min_area_slider ]
#         self.add_widget_list()
#
#         self.max_threshold_slider.bind(value=self.update_max_th_lbl)
#         self.min_threshold_slider.bind(value=self.update_min_th_lbl)
#         self.max_area_slider.bind(value=self.update_max_area_lbl)
#         self.min_area_slider.bind(value=self.update_min_area_lbl)
#
#         #create button to load parameters
#         self.save_prec_params_btn = Button()
#         self.save_prec_params_btn.text = "Load preprocessing params"
#         self.save_prec_params_btn.bind(on_press = self.save_preproc_params)
#         #create button to save parameter
#         self.segment_video_btn = Button()
#         self.segment_video_btn.text = "Segment video"
#         # self.load_prec_params_btn.bind(on_press = self.laod_preproc_params)
#
#     def do(self, *args):
#         if hasattr(CHOSEN_VIDEO.video, "video_path") and CHOSEN_VIDEO.video.video_path is not None:
#             self.init_preproc_parameters()
#             self.ROI_switch.bind(active = self.apply_ROI)
#             self.ROI_switch.active = CHOSEN_VIDEO.video.apply_ROI
#             self.bkg_subtractor_switch.active = CHOSEN_VIDEO.video.subtract_bkg
#             self.bkg_subtractor_switch.bind(active = self.apply_bkg_subtraction)
#             self.bkg = CHOSEN_VIDEO.video.bkg
#             self.ROI = CHOSEN_VIDEO.video.ROI if CHOSEN_VIDEO.video.ROI is not None else np.ones((CHOSEN_VIDEO.video.height, CHOSEN_VIDEO.video.width) ,dtype='uint8') * 255
#             CHOSEN_VIDEO.video.ROI = self.ROI
#             # print("ROI in do - preprocessing", self.ROI)
#             self.init_segment_zero()
#
#     def apply_ROI(self, instance, active):
#         # print("applying ROI")
#         CHOSEN_VIDEO.video.apply_ROI = active
#         if active  == True:
#             num_valid_pxs_in_ROI = len(sum(np.where(CHOSEN_VIDEO.video.ROI == 255)))
#             num_pxs_in_frame = CHOSEN_VIDEO.video.height * CHOSEN_VIDEO.video.width
#             self.ROI_is_trivial = num_pxs_in_frame == num_valid_pxs_in_ROI
#
#             if CHOSEN_VIDEO.video.ROI is not None and not self.ROI_is_trivial:
#                 self.ROI = CHOSEN_VIDEO.video.ROI
#             elif self.ROI_is_trivial:
#                 self.ROI_popup.open()
#                 instance.active = False
#                 CHOSEN_VIDEO.apply_ROI = False
#         elif active == False:
#             self.ROI = np.ones((CHOSEN_VIDEO.video.height, CHOSEN_VIDEO.video.width) ,dtype='uint8') * 255
#         self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)
#
#     def apply_bkg_subtraction(self, instance, active):
#         CHOSEN_VIDEO.video.subtract_bkg = active
#         if CHOSEN_VIDEO.video.subtract_bkg == True:
#             self.bkg_subtractor.subtract_bkg()
#             self.bkg = self.bkg_subtractor.bkg
#         else:
#             self.bkg = None
#         return active
#
#     def add_widget_list(self):
#         for w in self.w_list:
#             self.container_layout.add_widget(w)
#
#     def update_max_th_lbl(self,instance, value):
#         self.max_threshold_lbl.text = "Max threshold:\n" +  str(int(value))
#         self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)
#
#     def update_min_th_lbl(self,instance, value):
#         self.min_threshold_lbl.text = "Min threshold:\n" + str(int(value))
#         self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)
#         # self.show_preprocessing(self.frame)
#
#     def update_max_area_lbl(self,instance, value):
#         self.max_area_lbl.text = "Max area:\n" + str(int(value))
#         self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)
#
#     def update_min_area_lbl(self,instance, value):
#         self.min_area_lbl.text = "Min area:\n" + str(int(value))
#         self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)
#
#     def save_preproc_params(self):
#         self.saving_popup.open()
#         CHOSEN_VIDEO.video.max_threshold = self.max_threshold_slider.value
#         CHOSEN_VIDEO.video.min_threshold = self.min_threshold_slider.value
#         CHOSEN_VIDEO.video.min_area = self.min_area_slider.value
#         CHOSEN_VIDEO.video.max_area = self.max_area_slider.value
#
#     def save_preproc(self, *args):
#         CHOSEN_VIDEO.video.save()
#         self.saving_popup.dismiss()
#
#     def init_segment_zero(self):
#         # create instance of video shower
#         self.visualiser = VisualiseVideo()
#         self.add_widget(self.visualiser)
#         self.visualiser.visualise_video(CHOSEN_VIDEO.video, func = self.show_preprocessing)
#         self.currentSegment = 0
#         #create layout for video and slider
#         # self.button_layout = BoxLayout(orientation="horizontal", size_hint=(1.,.1))
#         # self.button_layout.add_widget(self.load_prec_params_btn)
#         # self.button_layout.add_widget(self.segment_video_btn)
#         # self.video_layout_preprocessing.add_widget(self.button_layout)
#
#     def show_preprocessing(self, frame):
#         # pass frame to grayscale
#         if len(frame.shape) > 2:
#             self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
#         else:
#             self.frame = frame
#         # compute average intensity
#         avIntensity = np.float32(np.mean(self.frame))
#         # generate a averaged frame for segmentation
#         self.av_frame = self.frame / avIntensity
#         # threshold the frame according to the sliders' values
#         # print('this is the current background ', self.bkg)
#         self.segmented_frame = segmentVideo(self.av_frame,
#                                             int(self.min_threshold_slider.value),
#                                             int(self.max_threshold_slider.value),
#                                             self.bkg,
#                                             self.ROI,
#                                             self.bkg_subtractor_switch.active)
#         #get information on the blobs find by thresholding
#         boundingBoxes, miniFrames, _, _, _, goodContours, _ = blobExtractor(self.segmented_frame,
#                                                                         self.frame,
#                                                                         int(self.min_area_slider.value),
#                                                                         int(self.max_area_slider.value))
#         #draw the blobs on the original frame
#         cv2.drawContours(self.frame, goodContours, -1, color=255, thickness = -1)
#         #display the segmentation
#         if self.count_scrollup != 0:
#             self.dst = cv2.warpAffine(self.frame, self.M, (self.frame.shape[1], self.frame.shape[1]))
#             buf1 = cv2.flip(self.dst,0)
#         else:
#             buf1 = cv2.flip(self.frame, 0)
#
#         buf = buf1.tostring()
#         textureFrame = Texture.create(size=(self.frame.shape[1], self.frame.shape[1]), colorfmt='luminance')
#         textureFrame.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
#         # display image from the texture
#         self.visualiser.display_layout.texture = textureFrame
#
#     def fromShowFrameToTexture(self, coords):
#         """
#         Maps coordinate in visualiser.display_layout (the image whose texture is the frame) to
#         the coordinates of the original image
#         """
#         coords = np.asarray(coords)
#         origFrameW = CHOSEN_VIDEO.video.width
#         origFrameH = CHOSEN_VIDEO.video.height
#
#         # actualFrameW, actualFrameH = self.visualiser.display_layout.size
#         # self.y_offset = self.sliderBox.height + self.button_layout.height
#         # self.x_offset = self.container_layout.width
#         # coords[0] = coords[0] - self.x_offset
#         # coords[1] = coords[1] - self.y_offset
#         # wRatio = abs(origFrameW / actualFrameW)
#         # hRatio = abs(origFrameH / actualFrameH)
#         # ratios = np.asarray([wRatio, hRatio])
#         # coords =  np.multiply(coords, ratios)
#         # coords[1] = origFrameH - coords[1]
#         return coords
#
#     def on_touch_down(self, touch):
#         self.touches = []
#         # print( 'scrollup number ', self.count_scrollup)
#         if self.parent is not None and self.visualiser.display_layout.collide_point(*touch.pos):
#             # print( 'i think you are on the image')
#             if touch.button == 'scrollup':
#                 self.count_scrollup += 1
#
#                 coords = self.fromShowFrameToTexture(touch.pos)
#                 rows,cols = self.frame.shape
#                 self.scale = 1.5 * self.count_scrollup
#                 self.M = cv2.getRotationMatrix2D((coords[0],coords[1]),0,self.scale)
#                 self.dst = cv2.warpAffine(self.frame,self.M,(cols,rows))
#                 buf1 = cv2.flip(self.dst, 0)
#                 buf = buf1.tostring()
#                 textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]),
#                                             colorfmt='luminance')
#                 textureFrame.blit_buffer(buf,
#                                         colorfmt='luminance', bufferfmt='ubyte')
#                 self.visualiser.display_layout.texture = textureFrame
#
#             elif touch.button == 'scrolldown':
#                 # frame = self.parent.frame
#                 coords = self.fromShowFrameToTexture(touch.pos)
#                 rows,cols = self.frame.shape
#                 self.dst = self.frame
#                 buf1 = cv2.flip(self.dst, 0)
#                 buf = buf1.tostring()
#                 textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]),
#                                             colorfmt='luminance')
#                 textureFrame.blit_buffer(buf,
#                                         colorfmt='luminance',
#                                         bufferfmt='ubyte')
#                 self.visualiser.display_layout.texture = textureFrame
#                 self.count_scrollup = 0
#
#         else:
#             self.scale = 1
#             self.disable_touch_down_outside_collided_widget(touch)
#
#     def disable_touch_down_outside_collided_widget(self, touch):
#         return super(PreprocessingPreview, self).on_touch_down(touch)
#
# class Accumulator(BoxLayout):
#     def __init__(self, **kwargs):
#         super(Accumulator, self).__init__(**kwargs)
#         global CHOSEN_VIDEO
#         CHOSEN_VIDEO.bind(chosen=self.do)
#
#     def do(self, *args):
#         if hasattr(CHOSEN_VIDEO.video, "video_path") and CHOSEN_VIDEO.video.video_path is not None:
#             if CHOSEN_VIDEO.video.has_been_assigned == True or  CHOSEN_VIDEO.old_video.has_been_assigned == True:
#                 return False
#             else:
#                 return True

class Root(TabbedPanel):
    global DEACTIVATE_VALIDATION, CHOSEN_VIDEO
    DEACTIVATE_VALIDATION = Deactivate_Process()
    CHOSEN_VIDEO = Chosen_Video(processes_list = PROCESSES)

    def __init__(self, **kwargs):
        super(Root, self).__init__(**kwargs)
        self.bind(current_tab = self.content_changed_cb)
        self.add_welcome_tab()
        self.add_validation_tab()
        self.add_individual_validator_tab()
        DEACTIVATE_VALIDATION.bind(process = self.manage_validation)
        DEACTIVATE_VALIDATION.bind(process = self.manage_individual_validation)


    def add_welcome_tab(self):
        self.welcome_tab = TabbedPanelItem(text = "Welcome")
        self.select_file = SelectFile(chosen_video = CHOSEN_VIDEO,
                                    deactivate_validation = DEACTIVATE_VALIDATION,
                                    setup_logging = setup_logging)
        self.welcome_tab.add_widget(self.select_file)
        self.add_widget(self.welcome_tab)

    def add_validation_tab(self):
        self.validation_tab = TabbedPanelItem(text='Global Validation')
        self.validation_tab.id = "Global validation"
        self.validation_tab.disabled = True
        self.add_widget(self.validation_tab)

    def manage_validation(self, *args):
        print("from root: ", DEACTIVATE_VALIDATION.process)
        self.validation_tab.disabled = DEACTIVATE_VALIDATION.process
        if not DEACTIVATE_VALIDATION.process:
            self.validator = Validator(chosen_video = CHOSEN_VIDEO,
                                        deactivate_validation = DEACTIVATE_VALIDATION)
            self.validator.id = "validator"
            self.validation_tab.add_widget(self.validator)
        else:
            if hasattr(self, 'validator'):
                self.validation_tab.clean(self.validator)

    def add_individual_validator_tab(self):
        self.individual_validation_tab = TabbedPanelItem(text='Individual Validation')
        self.individual_validation_tab.id = "Individual validation"
        self.individual_validation_tab.disabled = True
        self.add_widget(self.individual_validation_tab)

    def manage_individual_validation(self, *args):
        print("from root: ", DEACTIVATE_VALIDATION.process)
        self.individual_validation_tab.disabled = DEACTIVATE_VALIDATION.process
        if not DEACTIVATE_VALIDATION.process:
            self.individual_validator = IndividualValidator(chosen_video = CHOSEN_VIDEO,
                                        deactivate_validation = DEACTIVATE_VALIDATION)
            self.individual_validator.id = "individual_validator"
            self.individual_validation_tab.add_widget(self.individual_validator)
        else:
            if hasattr(self, 'individual_validator'):
                self.individual_validation_tab.clean(self.individual_validator)

    def content_changed_cb(self, obj, value):
        print('VALUE', value.__dict__)
        print('CONTENT', value.content)
        print("OBJECT", obj)
        print("ID", value.content.id)
        if value.content.id == "validator":
            self.validator.do()
        if value.content.id == "individual_validator":
            self.individual_validator.do()



    def on_switch(self, header):
        super(Root, self). switch_to(header)
        print('switch_to, content is ', header.content)
        self.cur_content = header.content

class MainWindow(BoxLayout):
    pass

class idtrackerdeepApp(App):
    Config.set('kivy', 'keyboard_mode', '')
    Config.set('graphics', 'fullscreen', '0')
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

    Config.write()
    def build(self):
        return MainWindow()

if __name__ == '__main__':
    idtrackerdeepApp().run()
