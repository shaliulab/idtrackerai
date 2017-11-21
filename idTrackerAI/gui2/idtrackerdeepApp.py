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

class HelpButton(ButtonBehavior, Image):
    def __init__(self, **kwargs):
        super(HelpButton, self).__init__(**kwargs)
        self.source = './help_button.png'
        self.size_hint = (.15,.15)

    def on_press(self):
        self.source = './help_button_on.png'

    def on_release(self):
        self.source = './help_button.png'

    def create_help_popup(self, title, text):
        self.help_popup_container = BoxLayout()
        self.help_label = Label(text=text)
        self.help_popup_container.add_widget(self.help_label)
        self.help_label.bind(width=lambda s, w:
                   s.setter('text_size')(s, (w, None)))
        self.help_label.size_hint = (1,1)
        self.help_popup = Popup(title = title,
                            content = self.help_popup_container,
                            size_hint = (.5, .5))
        self.bind(on_press = self.open_popup)

    def open_popup(self, *args):
        self.help_popup.open()

class Chosen_Video(EventDispatcher):
    chosen = StringProperty('')

    def __init__(self, processes_list = None, **kwargs):
        super(Chosen_Video,self).__init__(**kwargs)
        self.chosen = 'Default String'
        self.video = Video()
        self.processes_list = processes_list
        self.bind(chosen=self.on_modified)
        self.processes_to_restore = None

    def set_chosen_item(self, chosen_string):
        self.chosen = chosen_string

    def on_modified(self, instance, value):
        try:
            self.video.video_path = value
        except Exception,e:
            print(str(e))
            print("Choose a video to proceed")

class Deactivate_Process(EventDispatcher):
    process = BooleanProperty(True)

    def __init__(self, **kwargs):
        super(Deactivate_Process,self).__init__(**kwargs)
        self.process = True
        self.bind(process = self.on_modified)

    def setter(self, new_value):
        self.process = new_value

    def on_modified(self, instance, value):
        print("modifying validation to ", value)
        return value


class CustomLabel(Label):
    def __init__(self, font_size = 16, text = '', **kwargs):
        super(CustomLabel,self).__init__(**kwargs)
        self.text = text
        self.bind(size=lambda s, w: s.setter('text_size')(s, w))
        self.text_size = self.size
        self.size = self.texture_size
        self.font_size = font_size
        self.halign = "center"
        self.valign = "middle"

class SelectFile(BoxLayout):

    def __init__(self,**kwargs):
        super(SelectFile,self).__init__(**kwargs)
        global DEACTIVATE_VALIDATION
        DEACTIVATE_VALIDATION.bind(process = self.activate_process)
        self.update ='You did not select a video yet'
        self.main_layout = BoxLayout()
        self.main_layout.orientation = "vertical"
        self.logo = Image(source = "./logo.png")
        self.main_layout.add_widget(self.logo)
        self.welcome_label = CustomLabel(font_size = 20, text = "Welcome to idTrackerAI")
        self.main_layout.add_widget(self.welcome_label)
        self.add_widget(self.main_layout)
        self.video = None
        self.old_video = None
        self.help_button_welcome = HelpButton()
        self.main_layout.add_widget(self.help_button_welcome)
        self.help_button_welcome.create_help_popup("Getting started",\
                                                "Use the menu on the right to browse and select a video file. The supported formats are avi, mp4 and mpg. Albeit compressed video formats are accepted, we suggest to use uncompressed ones for an optimal tracking. See the documentation for more details.\n\nClick on the main window to close the popup.")
        self.filechooser = FileChooserListView(path = os.getcwd(), size_hint = (1., 1.))
        self.filechooser.bind(selection = self.open)
        self.add_widget(self.filechooser)


    global CHOSEN_VIDEO
    CHOSEN_VIDEO = Chosen_Video(processes_list = PROCESSES)

    def on_enter_session_folder(self,value):
        new_name_session_folder = self.session_name_input.text
        CHOSEN_VIDEO.video.create_session_folder(name = new_name_session_folder)
        CHOSEN_VIDEO.logger = setup_logging(path_to_save_logs = CHOSEN_VIDEO.video.session_folder, video_object = CHOSEN_VIDEO.video)
        self.welcome_popup.dismiss()
        if CHOSEN_VIDEO.video.previous_session_folder != '':
            CHOSEN_VIDEO.existent_files, CHOSEN_VIDEO.old_video = getExistentFiles(CHOSEN_VIDEO.video, CHOSEN_VIDEO.processes_list)
            self.create_restore_popup()
            self.restore_popup.open()

    def open(self, *args):
        try:
            CHOSEN_VIDEO.set_chosen_item(self.filechooser.selection[0])
            if CHOSEN_VIDEO.video.video_path is not None:
                self.create_welcome_popup()
                self.session_name_input.bind(on_text_validate = self.on_enter_session_folder)
                self.welcome_popup.open()
        except Exception,e:
            print(str(e))

    def create_welcome_popup(self):
        self.popup_container = BoxLayout()
        self.session_name_box = BoxLayout(orientation="vertical")
        self.session_name_label = CustomLabel(font_size = 16, text='Give a name to the current tracking session. Use the name of an existent session to load it.')
        self.session_name_box.add_widget(self.session_name_label)
        self.session_name_input = TextInput(text ='', multiline=False)
        self.session_name_box.add_widget(self.session_name_input)
        self.popup_container.add_widget(self.session_name_box)
        self.welcome_popup = Popup(title = 'Session name',
                            content = self.popup_container,
                            size_hint = (.4, .4))

    def create_restore_checkboxes(self):
        self.processes_checkboxes = []

        for i, process in enumerate(CHOSEN_VIDEO.processes_list):

            process_container = BoxLayout()
            if CHOSEN_VIDEO.existent_files[process] == '1':
                checkbox = CheckBox(size_hint = (.1, 1))
                checkbox.group = process
                checkbox.active = True
                process_container.add_widget(checkbox)
                self.processes_checkboxes.append(checkbox)
            else:
                checkbox = BoxLayout(size_hint = (.1, 1))
                process_container.add_widget(checkbox)
            checkbox_label = Label(text = process.replace("_", " "), size_hint = (.9, 1))
            process_container.add_widget(checkbox_label)
            self.restore_popup_container.add_widget(process_container)

        self.restore_button = Button(text = "Restore selected processes")
        self.restore_popup_container.add_widget(self.restore_button)
        self.restore_button.bind(on_press = self.get_processes_to_restore)

    def get_processes_to_restore(self, *args):
        CHOSEN_VIDEO.processes_to_restore = {checkbox.group: checkbox.active for checkbox
                                        in self.processes_checkboxes}

        if CHOSEN_VIDEO.processes_to_restore['assignment'] or CHOSEN_VIDEO.processes_to_restore['correct_duplications']:
            DEACTIVATE_VALIDATION.setter(False)
        self.restore_popup.dismiss()

    def activate_process(self, *args):
        print("in activate process. Returning ", DEACTIVATE_VALIDATION.process)
        return DEACTIVATE_VALIDATION.process

    def on_checkbox_active(self, checkbox, value):
        index = self.processes_checkboxes.index(checkbox)
        if value:

            for i, checkbox in enumerate(self.processes_checkboxes):

                if i <= index:
                    checkbox.active = True
        else:

            for i, checkbox in enumerate(self.processes_checkboxes):

                if i > index:
                    checkbox.active = False

    def bind_processes_checkboxes(self):
        for checkbox in self.processes_checkboxes:
            checkbox.bind(active = self.on_checkbox_active)

    def create_restore_popup(self):
        self.restore_popup_container = BoxLayout(orientation = "vertical")
        self.create_restore_checkboxes()
        self.bind_processes_checkboxes()
        self.restore_popup = Popup(title = 'Some processes have already been executed.\nDo you want to restore them?',
                                    content = self.restore_popup_container,
                                    size_hint = (.6, .8))


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

class Validator(BoxLayout):
    def __init__(self, **kwargs):
        super(Validator, self).__init__(**kwargs)
        global CHOSEN_VIDEO, DEACTIVATE_VALIDATION
        self.visualiser = VisualiseVideo(chosen_video = CHOSEN_VIDEO)
        self.warning_popup = Popup(title = 'Warning',
                            content = CustomLabel(text = 'The video has not been tracked yet. Track it before performing validation.'),
                            size_hint = (.3,.3))
        self.loading_popup = Popup(title='Loading',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self.do()

    def show_saving(self, *args):
        self.popup_saving = Popup(title='Saving',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.popup_saving.open()

    def create_count_bad_crossing_popup(self):
        self.wc_popup_container = BoxLayout()
        self.wc_identity_box = BoxLayout(orientation="vertical")
        self.wc_label = CustomLabel(text='Type the identity associated to a badly corrected crossing')
        self.wc_identity_box.add_widget(self.wc_label)
        self.wc_identity_input = TextInput(text ='', multiline=False)
        self.wc_identity_box.add_widget(self.wc_identity_input)
        self.wc_popup_container.add_widget(self.wc_identity_box)
        self.wc_popup = Popup(title = 'Count wrong crossings',
                            content = self.wc_popup_container,
                            size_hint = (.4, .4))

    def on_enter_wrong_crossing_identity(self, value):
        self.wrong_crossing_counter[int(self.wc_identity_input.text)] += 1
        self.wc_popup.dismiss()

    def create_choose_list_of_blobs_popup(self):
        self.lob_container = BoxLayout()
        self.lob_box = BoxLayout(orientation="vertical")
        self.lob_label = CustomLabel(text='We detected two different trajectory files. Which one do you want to use for validation?')
        self.lob_btns_container = BoxLayout()
        self.lob_btn1 = Button(text = "With gaps")
        self.lob_btn2 = Button(text = "Without gaps")
        self.lob_btns_container.add_widget(self.lob_btn1)
        self.lob_btns_container.add_widget(self.lob_btn2)
        self.lob_box.add_widget(self.lob_label)
        self.lob_box.add_widget(self.lob_btns_container)
        self.lob_container.add_widget(self.lob_box)
        self.choose_list_of_blobs_popup = Popup(title = 'Choose validation trajectories',
                            content = self.lob_container,
                            size_hint = (.4, .4))

    def show_loading_text(self, *args):
        self.lob_label.text = "Loading..."

    def on_choose_list_of_blobs_btns_press(self, instance):
        if instance.text == 'With gaps':
            self.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video.blobs_path,
                                        video_has_been_segmented = CHOSEN_VIDEO.video.has_been_segmented)
            self.list_of_blobs_save_path = CHOSEN_VIDEO.video.blobs_path
        else:
            self.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video.blobs_no_gaps_path,
                                        video_has_been_segmented = CHOSEN_VIDEO.video.has_been_segmented)
            blobs_path, blobs_path_extension = CHOSEN_VIDEO.video.blobs_no_gaps_path
        self.choose_list_of_blobs_popup.dismiss()
        self.populate_validation_tab()

    def populate_validation_tab(self):
        self.blobs_in_video = self.list_of_blobs.blobs_in_video
        self.count_scrollup = 0
        self.scale = 1
        self.wrong_crossing_counter = {identity: 0 for identity in range(1, CHOSEN_VIDEO.video.number_of_animals + 1)}
        self.create_count_bad_crossing_popup()
        self.wc_identity_input.bind(on_text_validate = self.on_enter_wrong_crossing_identity)
        self.loading_popup.dismiss()
        self.init_segmentZero()

    def get_first_frame(self):
        if not hasattr(CHOSEN_VIDEO.video, 'first_frame_first_global_fragment'):
            CHOSEN_VIDEO.video._first_frame_first_global_fragment = CHOSEN_VIDEO.old_video.first_frame_first_global_fragment
        return CHOSEN_VIDEO.video.first_frame_first_global_fragment

    def do(self, *args):
        # try:
        if CHOSEN_VIDEO.processes_to_restore is not None and CHOSEN_VIDEO.processes_to_restore['assignment']:
            CHOSEN_VIDEO.video.__dict__.update(CHOSEN_VIDEO.old_video.__dict__)
        if  CHOSEN_VIDEO.processes_to_restore is not None\
            and 'crossings' in CHOSEN_VIDEO.processes_to_restore\
            and CHOSEN_VIDEO.processes_to_restore['crossings']:
            self.create_choose_list_of_blobs_popup()
            self.lob_btn1.bind(on_press = self.show_loading_text)
            self.lob_btn2.bind(on_press = self.show_loading_text)
            self.lob_btn1.bind(on_release = self.on_choose_list_of_blobs_btns_press)
            self.lob_btn2.bind(on_release = self.on_choose_list_of_blobs_btns_press)
            self.choose_list_of_blobs_popup.open()
        else:
            self.loading_popup.open()
            self.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video.blobs_path,
                                        video_has_been_segmented = CHOSEN_VIDEO.video.has_been_segmented)
            self.list_of_blobs_save_path = CHOSEN_VIDEO.video.blobs_path
            self.populate_validation_tab()
        # except Exception as e:
        #     exc_type, exc_obj, exc_tb = sys.exc_info()
        #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #     print(exc_type, fname, exc_tb.tb_lineno)
        #     self.warning_popup.open()

    def init_segmentZero(self):
        self.add_widget(self.visualiser)
        self.colors = get_spaced_colors_util(CHOSEN_VIDEO.video.number_of_animals)
        self.button_box = BoxLayout(orientation='vertical', size_hint=(.3,1.))
        self.add_widget(self.button_box)
        self.next_cross_button = Button(id='crossing_btn', text='Next fucking crossing', size_hint=(1,1))
        self.next_cross_button.bind(on_press=self.go_to_next_crossing)
        self.button_box.add_widget(self.next_cross_button)
        self.previous_cross_button = Button(id='crossing_btn', text='Previous fucking crossing', size_hint=(1,1))
        self.previous_cross_button.bind(on_press=self.go_to_previous_crossing)
        self.button_box.add_widget(self.previous_cross_button)
        self.go_to_first_global_fragment_button = Button(id='back_to_first_gf_btn', text='First global fragment', size_hint=(1,1))
        self.go_to_first_global_fragment_button.bind(on_press = self.go_to_first_global_fragment)
        self.button_box.add_widget(self.go_to_first_global_fragment_button)
        self.save_groundtruth_btn = Button(id='save_groundtruth_btn', text='Save updated identities',size_hint = (1,1))
        self.save_groundtruth_btn.bind(on_press=self.show_saving)
        self.save_groundtruth_btn.bind(on_release=self.save_groundtruth_list_of_blobs)
        self.save_groundtruth_btn.disabled = True
        self.button_box.add_widget(self.save_groundtruth_btn)
        self.compute_accuracy_button = Button(id = "compute_accuracy_button", text = "Compute accuracy", size_hint  = (1.,1.))
        self.compute_accuracy_button.disabled = False
        self.compute_accuracy_button.bind(on_press = self.compute_and_save_session_accuracy_wrt_groundtruth_APP)
        self.button_box.add_widget(self.compute_accuracy_button)
        self.visualiser.visualise_video(CHOSEN_VIDEO.video, func = self.writeIds, frame_index_to_start = self.get_first_frame())

    def go_to_next_crossing(self,instance):
        non_crossing = True
        frame_index = int(self.visualiser.video_slider.value)

        while non_crossing == True:
            if frame_index < CHOSEN_VIDEO.video.number_of_frames:
                frame_index = frame_index + 1
                blobs_in_frame = self.blobs_in_video[frame_index]
                for blob in blobs_in_frame:
                    if not blob.is_an_individual:
                        non_crossing = False
                        self.visualiser.video_slider.value = frame_index
                        self.visualiser.visualise(frame_index, func = self.writeIds)
            else:
                break

    def go_to_previous_crossing(self,instance):
        non_crossing = True
        frame_index = int(self.visualiser.video_slider.value)

        while non_crossing == True:
            if frame_index > 0:
                frame_index = frame_index - 1
                blobs_in_frame = self.blobs_in_video[frame_index]
                for blob in blobs_in_frame:
                    if not blob.is_an_individual:
                        non_crossing = False
                        self.visualiser.video_slider.value = frame_index
                        self.visualiser.visualise(frame_index, func = self.writeIds)
            else:
                break

    def go_to_first_global_fragment(self, instance):
        self.visualiser.visualise(self.get_first_frame(), func = self.writeIds)
        self.visualiser.video_slider.value = self.get_first_frame()

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):

        frame_index = int(self.visualiser.video_slider.value)
        if keycode[1] == 'left':
            frame_index -= 1
        elif keycode[1] == 'right':
            frame_index += 1
        elif keycode[1] == 'c':
            self.wc_popup.open()
        self.visualiser.video_slider.value = frame_index
        self.visualiser.visualise(frame_index, func = self.writeIds)
        return True

    @staticmethod
    def get_clicked_blob(point, contours):
        """
        Get the contour that contains point
        """
        indices = [i for i, cnt in enumerate(contours) if cv2.pointPolygonTest(cnt, tuple(point), measureDist = False) >= 0]
        if len(indices) != 0:
            return indices[0]
        else:
            return None

    def apply_affine_transform_on_point(self, affine_transform_matrix, point):
        R = affine_transform_matrix[:,:-1]
        T = affine_transform_matrix[:,-1]
        return np.dot(R, np.squeeze(point)) + T

    def apply_inverse_affine_transform_on_point(self, affine_transform_matrix, point):
        inverse_affine_transform_matrix = cv2.invertAffineTransform(affine_transform_matrix)
        return self.apply_affine_transform_on_point(inverse_affine_transform_matrix, point)

    def apply_affine_transform_on_contour(self, affine_transform_matrix, contour):
        return np.expand_dims(np.asarray([self.apply_affine_transform_on_point(affine_transform_matrix, point) for point in contour]).astype(int),axis = 1)

    def get_blob_to_modify_and_mouse_coordinate(self):
        mouse_coords = self.touches[0]
        frame_index = int(self.visualiser.video_slider.value) #get the current frame from the slider
        blobs_in_frame = self.blobs_in_video[frame_index]
        contours = [getattr(blob, "contour") for blob in blobs_in_frame]
        if self.scale != 1:
            contours = [self.apply_affine_transform_on_contour(self.M, cnt) for cnt in contours]
        mouse_coords = self.fromShowFrameToTexture(mouse_coords)
        if self.scale != 1:
            mouse_coords = self.apply_inverse_affine_transform_on_point(self.M, mouse_coords)
        blob_ind = self.get_clicked_blob(mouse_coords, contours)
        if blob_ind is not None:
            blob_to_modify = blobs_in_frame[blob_ind]
            return blob_to_modify, mouse_coords
        else:
            return None, None

    def fromShowFrameToTexture(self, coords):
        """
        Maps coordinate in showFrame (the image whose texture is the frame) to
        the coordinates of the original image
        """
        coords = np.asarray(coords)
        if hasattr(CHOSEN_VIDEO.video, 'resolution_reduction') and  CHOSEN_VIDEO.video.resolution_reduction != 1:
            original_frame_width = int(CHOSEN_VIDEO.video.width * CHOSEN_VIDEO.video.resolution_reduction)
            original_frame_height = int(CHOSEN_VIDEO.video.height * CHOSEN_VIDEO.video.resolution_reduction)
        else:
            original_frame_width = int(CHOSEN_VIDEO.video.width)
            original_frame_height = int(CHOSEN_VIDEO.video.height)
        actual_frame_width, actual_frame_height = self.visualiser.display_layout.size
        self.offset = self.visualiser.footer.height
        coords[1] = coords[1] - self.offset
        wRatio = abs(original_frame_width / actual_frame_width)
        hRatio = abs(original_frame_height / actual_frame_height)
        ratios = np.asarray([wRatio, hRatio])
        coords =  np.multiply(coords, ratios)
        coords[1] = original_frame_height - coords[1]
        return coords

    @staticmethod
    def get_attributes_from_blobs_in_frame(blobs_in_frame, attributes_to_get):
        return {attr: [getattr(blob, attr) for blob in blobs_in_frame] for attr in attributes_to_get}

    def writeIds(self, frame):
        blobs_in_frame = self.blobs_in_video[int(self.visualiser.video_slider.value)]
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = self.visualiser.frame
        frame_number = blobs_in_frame[0].frame_number

        for blob in blobs_in_frame:
            cur_id = blob.final_identity
            cur_id_str = str(cur_id)
            roots = ['a-', 'd-', 'c-','i-', 'u-']
            if blob.user_generated_identity is not None:
                root = roots[4]
            elif blob.identity_corrected_closing_gaps is not None and blob.is_an_individual:
                root = roots[3]
            elif blob.identity_corrected_closing_gaps is not None:
                root = roots[2]
            elif blob.identity_corrected_solving_duplication is not None:
                root = roots[1]
            elif not blob.used_for_training:
                root = roots[0]
            else:
                root  = ''
            if isinstance(cur_id, int):
                cur_id_str = root + cur_id_str
                int_centroid = np.asarray(blob.centroid).astype('int')
                cv2.circle(frame, tuple(int_centroid), 2, self.colors[cur_id], -1)
                cv2.putText(frame, cur_id_str,tuple(int_centroid), font, 1, self.colors[cur_id], 3)
            elif isinstance(cur_id, list):
                for c_id, c_centroid in zip(cur_id, blob.interpolated_centroids):
                    c_id_str = root + str(c_id)
                    int_centroid = tuple([int(centroid_coordinate) for centroid_coordinate in c_centroid])
                    cv2.circle(frame, int_centroid, 2, self.colors[c_id], -1)
                    cv2.putText(frame, c_id_str, int_centroid, font, 1, self.colors[c_id], 3)

                self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
                self._keyboard.bind(on_key_down=self._on_keyboard_down)
            elif blob.is_a_crossing:
                bounding_box = blob.bounding_box_in_frame_coordinates
                cv2.rectangle(frame, bounding_box[0], bounding_box[1], (255, 0, 0) , 2)
        if self.scale != 1:
            self.dst = cv2.warpAffine(frame, self.M, (frame.shape[1], frame.shape[0]))
            buf = cv2.flip(self.dst,0)
            buf = buf.tostring()
        else:
            buf = cv2.flip(frame,0)
            buf = buf.tostring()
        textureFrame = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.visualiser.display_layout.texture = textureFrame

    def on_enter(self,value):
        self.identity_update = int(self.identityInput.text)
        self.overwriteIdentity()
        self.popup.dismiss()

    def propagate_groundtruth_identity_in_individual_fragment(self):
        modified_blob = self.blob_to_modify
        count_past_corrections = 1 #to take into account the modification already done in the current frame
        count_future_corrections = 0
        new_blob_identity = modified_blob.user_generated_identity
        if modified_blob.is_an_individual_in_a_fragment:
            current = modified_blob

            while current.next[0].is_an_individual_in_a_fragment:
                # print("propagating forward")
                current.next[0]._user_generated_identity = current.user_generated_identity
                current = current.next[0]
                count_future_corrections += 1
                # print(count_future_corrections)

            current = modified_blob

            while current.previous[0].is_an_individual_in_a_fragment:
                # print("propagating backward")
                current.previous[0]._user_generated_identity = current.user_generated_identity
                current = current.previous[0]
                count_past_corrections += 1

        #init and bind keyboard again
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def overwriteIdentity(self):
        # enable buttons to save corrected version and compute the accuracy
        self.save_groundtruth_btn.disabled = False
        self.compute_accuracy_button.disabled = False
        if not self.blob_to_modify.is_a_crossing:
            self.blob_to_modify._user_generated_identity = self.identity_update
            self.propagate_groundtruth_identity_in_individual_fragment()
        else:
            self.blob_to_modify._user_generated_centroids.append(self.user_generated_centroids)
            self.blob_to_modify._user_generated_identities.append(self.identity_update)

        self.visualiser.visualise(trackbar_value = int(self.visualiser.video_slider.value), func = self.writeIds)

    def on_press_show_saving(selg, *args):
        self.show_saving()

    def save_groundtruth_list_of_blobs(self, *args):
        self.go_and_save()
        self.popup_saving.dismiss()

    def go_and_save(self):
        self.list_of_blobs.save(path_to_save = self.list_of_blobs_save_path)
        CHOSEN_VIDEO.video.save()

    def modifyIdOpenPopup(self, blob_to_modify):
        self.container = BoxLayout()
        self.blob_to_modify = blob_to_modify
        if blob_to_modify.user_generated_identity is None:
            self.id_to_modify = blob_to_modify.identity
        else:
            self.id_to_modify = blob_to_modify.user_generated_identity
        text = str(self.id_to_modify)
        self.old_id_box = BoxLayout(orientation="vertical")
        self.new_id_box = BoxLayout(orientation="vertical")
        self.selected_label = CustomLabel(text='You selected animal:\n')
        self.selected_label_num = Label(text=text)
        self.new_id_label = CustomLabel(text='Type the new identity and press enter to confirm\n')
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
        self.identityInput.bind(on_text_validate=self.on_enter)
        self.popup.open()

    def show_blob_attributes(self, blob_to_explore):
        self.container = BoxLayout()
        self.blob_to_explore = blob_to_explore
        self.show_attributes_box = BoxLayout(orientation="vertical")
        self.id_label = CustomLabel(text='Assigned identity: ' + str(blob_to_explore.final_identity))
        self.frag_id_label = CustomLabel(text='Fragment identifier: ' + str(blob_to_explore.fragment_identifier))
        self.accumulation_label = CustomLabel(text='Used for training: ' + str(blob_to_explore.used_for_training))
        self.in_a_fragment_label = CustomLabel(text='It is in an individual fragment: ' + str(blob_to_explore.is_in_a_fragment))
        self.fish_label = CustomLabel(text='It is a fish: ' + str(blob_to_explore.is_an_individual))
        self.ghost_crossing_label = CustomLabel(text='It is a ghost crossing: ' + str(blob_to_explore.is_a_ghost_crossing))
        self.jump_label = CustomLabel(text='It is a jump: ' + str(blob_to_explore.is_a_jump))
        self.container.add_widget(self.show_attributes_box)
        widget_list = [self.id_label, self.frag_id_label,
                        self.accumulation_label, self.in_a_fragment_label,
                        self.ghost_crossing_label, self.jump_label]
        [self.show_attributes_box.add_widget(w) for w in widget_list]
        self.popup = Popup(title='Blob attributes',
            content=self.container,
            size_hint=(.4,.4))
        self.popup.color = (0.,0.,0.,0.)
        self.popup.open()

    def on_touch_down(self, touch):
        self.touches = []
        if self.visualiser.display_layout.collide_point(*touch.pos):
            if touch.button =='left':
                self.touches.append(touch.pos)
                self.id_to_modify, self.user_generated_centroids = self.get_blob_to_modify_and_mouse_coordinate()
                if self.id_to_modify is not None:
                    self.modifyIdOpenPopup(self.id_to_modify)
            elif touch.button == 'scrollup':
                self.count_scrollup += 1
                coords = self.fromShowFrameToTexture(touch.pos)
                rows, cols, channels = self.visualiser.frame.shape
                self.scale = 1.5 * self.count_scrollup
                self.M = cv2.getRotationMatrix2D((coords[0],coords[1]),0,self.scale)
                self.dst = cv2.warpAffine(self.visualiser.frame,self.M,(cols,rows))
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='bgr')
                textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.visualiser.display_layout.texture = textureFrame
            elif touch.button == 'scrolldown':
                coords = self.fromShowFrameToTexture(touch.pos)
                rows,cols, channels = self.visualiser.frame.shape
                self.dst = self.visualiser.frame
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='bgr')
                textureFrame.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.visualiser.display_layout.texture = textureFrame
                self.count_scrollup = 0
                self.scale = 1
            elif touch.button == 'right':
                self.touches.append(touch.pos)
                self.id_to_modify, self.user_generated_centroids = self.get_blob_to_modify_and_mouse_coordinate()
                if self.id_to_modify is not None:
                    self.show_blob_attributes(self.id_to_modify)
        else:
            self.scale = 1
            self.disable_touch_down_outside_collided_widget(touch)

    def disable_touch_down_outside_collided_widget(self, touch):
        return super(Validator, self).on_touch_down(touch)

    def get_groundtruth_path(self):
        groundtruth_path = os.path.join(CHOSEN_VIDEO.video.video_folder, '_groundtruth.npy')
        return groundtruth_path if os.path.isfile(groundtruth_path) else None

    def on_groundtruth_popup_button_press(self, instance):
        if instance.text == "Use pre-existent ground truth":
            self.groundtruth = np.load(self.groundtruth_path).item()
            self.plot_groundtruth_statistics()
            self.popup_start_end_groundtruth.dismiss()
        else:
            self.gt_start_end_container.remove_widget(self.gt_start_end_btn1)
            self.gt_start_end_container.remove_widget(self.gt_start_end_btn2)
            self.gt_start_end_label.text = "Insert the start and ending frame (e.g. 100 - 2050) on which the ground truth has been computed"
            self.gt_start_end_text_input = TextInput(text ='', multiline=False)
            self.gt_start_end_container.add_widget(self.gt_start_end_text_input)
            self.gt_start_end_text_input.bind(on_text_validate = self.on_enter_start_end)

    def create_frame_interval_popup(self):
        self.gt_start_end_container = BoxLayout(orientation = "vertical")
        self.groundtruth_path = self.get_groundtruth_path()
        if self.groundtruth_path is not None:
            if self.save_groundtruth_btn.disabled:
                self.groundtruth = np.load(self.groundtruth_path).item()
                self.plot_groundtruth_statistics()
                return True
            if not self.save_groundtruth_btn.disabled:
                self.gt_start_end_label = CustomLabel(text = "A pre-existent ground truth file has been detected. Do you want to use it to compute the accuracy or use a new one?")
                self.gt_start_end_btn1 = Button(text = "Use pre-existent ground truth")
                self.gt_start_end_btn2 = Button(text = "Generate new ground truth")
                self.gt_start_end_container.add_widget(self.gt_start_end_label)
                self.gt_start_end_container.add_widget(self.gt_start_end_btn1)
                self.gt_start_end_container.add_widget(self.gt_start_end_btn2)
                self.gt_start_end_btn1.bind(on_press = self.on_groundtruth_popup_button_press)
                self.gt_start_end_btn2.bind(on_press = self.on_groundtruth_popup_button_press)
        else:
            if self.save_groundtruth_btn.disabled:
                self.gt_start_end_label = CustomLabel(text = "No pre-existent groundtruth file has been detected. Validate the video to compute a ground truth first.\n\n Need help? To modify a wrong identity click on the badly identified animal and fill the popup. Use the mouse wheel to zoom if necessary.")
                self.gt_start_end_container.add_widget(self.gt_start_end_label)
            else:
                self.gt_start_end_label = CustomLabel(text = "Insert the start and ending frame (e.g. 100 - 2050) on which the ground truth has been computed")
                self.gt_start_end_container.add_widget(self.gt_start_end_label)
                self.gt_start_end_text_input = TextInput(text ='', multiline=False)
                self.gt_start_end_container.add_widget(self.gt_start_end_text_input)
                self.gt_start_end_text_input.bind(on_text_validate = self.on_enter_start_end)

        self.popup_start_end_groundtruth = Popup(title='Groundtruth Accuracy - Frame Interval',
                    content=self.gt_start_end_container,
                    size_hint=(.4,.4))

    def on_enter_start_end(self, value):
        start, end = self.gt_start_end_text_input.text.split('-')
        self.gt_start_frame = int(start)
        self.gt_end_frame = int(end)
        self.generate_groundtruth()
        self.save_groundtruth()
        self.plot_groundtruth_statistics()
        if not self.prevent_open_popup:
            self.popup_start_end_groundtruth.dismiss()

    def generate_groundtruth(self):
        self.groundtruth = generate_groundtruth(CHOSEN_VIDEO.video, self.blobs_in_video, self.gt_start_frame, self.gt_end_frame, save_gt = False)

    def save_groundtruth(self):
        self.groundtruth.save()

    def plot_groundtruth_statistics(self):
        blobs_in_video_groundtruth = self.groundtruth.blobs_in_video[self.groundtruth.start:self.groundtruth.end]
        blobs_in_video = self.blobs_in_video[self.groundtruth.start:self.groundtruth.end]
        gt_accuracies, _ = get_accuracy_wrt_groundtruth(CHOSEN_VIDEO.video, blobs_in_video_groundtruth, blobs_in_video)
        if gt_accuracies is not None:
            self.individual_accuracy = gt_accuracies['individual_accuracy']
            self.accuracy = gt_accuracies['accuracy']
            self.plot_final_statistics()
            self.statistics_popup.open()
            CHOSEN_VIDEO.video.gt_start_end = (self.groundtruth.start, self.groundtruth.end)
            CHOSEN_VIDEO.video.gt_accuracy = gt_accuracies
            CHOSEN_VIDEO.video.save()

    def compute_and_save_session_accuracy_wrt_groundtruth_APP(self, *args):
        self.prevent_open_popup = self.create_frame_interval_popup()
        if not self.prevent_open_popup:
            self.popup_start_end_groundtruth.open()

    def plot_final_statistics(self):
        content = BoxLayout()
        self.statistics_popup = Popup(title = "Statistics",
                                    content = content,
                                    size_hint = (.5, .5))
        fig, ax = plt.subplots(1)
        colors = get_spaced_colors_util(CHOSEN_VIDEO.video.number_of_animals, norm = True)
        width = .5
        plt.bar(self.individual_accuracy.keys(), self.individual_accuracy.values(), width, color=colors)
        plt.axhline(self.accuracy, color = 'k', linewidth = .2)
        ax.set_xlabel('individual')
        ax.set_ylabel('Individual accuracy')
        content.add_widget(FigureCanvasKivyAgg(fig))

class Root(TabbedPanel):
    global DEACTIVATE_VALIDATION
    DEACTIVATE_VALIDATION = Deactivate_Process()

    def __init__(self, **kwargs):
        super(Root, self).__init__(**kwargs)
        self.bind(current_tab = self.content_changed_cb)
        self.add_welcome_tab()
        self.add_validation_tab()
        DEACTIVATE_VALIDATION.bind(process = self.manage_validation)

    def add_welcome_tab(self):
        self.welcome_tab = TabbedPanelItem(text = "Welcome")
        self.select_file = SelectFile(size_hint = (1., 1.))
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
            self.validator = Validator()
            self.validation_tab.add_widget(self.validator)
        else:
            if hasattr(self, 'validator'):
                self.validation_tab.clean(self.validator)

    def content_changed_cb(self, obj, value):
        pass
        # print('CONTENT', value.__dict__)
        # print(type(value.content.id))

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
