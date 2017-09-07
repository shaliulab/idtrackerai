from __future__ import absolute_import, division, print_function
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
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from kivy.graphics import *
from kivy.graphics.transformation import Matrix
from kivy.clock import Clock
from kivy.config import Config #used before running the app to set the keyboard usage
from kivy.event import EventDispatcher

import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing')
sys.path.append('../groundtruth_utils')
sys.setrecursionlimit(100000)
import cv2
import numpy as np
from video import Video
from py_utils import getExistentFiles, get_spaced_colors_util
from video_utils import computeBkg, blobExtractor
from segmentation import segmentVideo
from blob import ListOfBlobs
from globalfragment import order_global_fragments_by_distance_travelled
# from validator import Validator
"""
Start kivy classes
"""
class Chosen_Video(EventDispatcher):
    chosen = StringProperty('')

    def __init__(self,**kwargs):
        super(Chosen_Video,self).__init__(**kwargs)
        self.chosen = 'Default String'
        self.video = Video()
        self.bind(chosen=self.on_modified)

    def set_chosen_item(self, chosen_string):
        self.chosen = chosen_string

    def on_modified(self, instance, value):
        print("Chosen item in ", instance, " was modified to :",value)
        try:
            print("you choose ----------->", value)
            self.video.video_path = value
            new_name_session_folder = raw_input('Session name: ')
            self.video.create_session_folder(name = new_name_session_folder)
            processes_list = ['bkg', 'ROI', 'preprocparams', 'preprocessing', 'pretraining', 'accumulation', 'training', 'assignment']
            #get existent files and paths to load them
            self.existentFiles, self.old_video = getExistentFiles(self.video, processes_list)
            if self.old_video._has_been_assigned: self.video._has_been_assigned
            if hasattr(self.old_video, 'resolution_reduction'):
                print("--------------------------------------------------------")
                self.video.resolution_reduction = self.old_video.resolution_reduction
                self.video.bkg = self.old_video.bkg
                self.video.ROI = self.old_video.ROI

                print(self.video.resolution_reduction, self.video.ROI.shape)
                if self.video.ROI.shape[0] != self.video._height * self.video.resolution_reduction:
                    print("resizing ROI")
                    self.video.ROI = cv2.resize(self.video.ROI, None, fx = self.video.resolution_reduction, fy = self.video.resolution_reduction, interpolation = cv2.INTER_CUBIC)

                    print("***********************resized ROI shape ", self.video.ROI.shape)
                if self.video.bkg is not None and self.video.bkg.shape[0] != self.video._height * self.video.resolution_reduction:
                    print("resizing BKG")
                    self.video.bkg = cv2.resize(self.video.bkg, None, fx = self.video.resolution_reduction, fy = self.video.resolution_reduction, interpolation = cv2.INTER_CUBIC)
            if CHOSEN_VIDEO.video.video_path is not None:
                if CHOSEN_VIDEO.old_video.preprocessing_type is None or CHOSEN_VIDEO.old_video.number_of_animals is None:
                    self.create_preprocessing_type_and_number_popup()
                    self.preprocessing_type_input.bind(on_text_validate = self.on_enter)
                    self.animal_number_input.bind(on_text_validate = self.on_enter)
                    self.popup.open()
                else:
                    CHOSEN_VIDEO.video._preprocessing_type = CHOSEN_VIDEO.old_video.preprocessing_type
                    CHOSEN_VIDEO.video._number_of_animals = CHOSEN_VIDEO.old_video.number_of_animals
                self.enable_ROI_and_preprocessing_tabs = True
        except Exception,e:
            print(str(e))
            print("Choose a video to proceed")

class SelectFile(BoxLayout):
    def __init__(self,**kwargs):
        super(SelectFile,self).__init__(**kwargs)
        self.update ='You did not select a video yet'
        self.main_layout = BoxLayout()
        self.main_layout.orientation = "vertical"
        self.logo = Image(source = "./logo.png")
        self.welcome_label = Label()
        self.main_layout.add_widget(self.logo)
        self.main_layout.add_widget(self.welcome_label)
        self.add_widget(self.main_layout)
        self.welcome_label.text = "Select a video"
        self.welcome_label.text_size = self.welcome_label.size
        self.welcome_label.size = self.welcome_label.texture_size
        self.welcome_label.font_size = 20
        self.welcome_label.halign = "center"
        self.welcome_label.valign = "middle"
        self.video = None
        self.old_video = None

    global CHOSEN_VIDEO
    CHOSEN_VIDEO = Chosen_Video()

    def on_enter(self,value):
        CHOSEN_VIDEO.video._preprocessing_type = self.preprocessing_type_input.text
        CHOSEN_VIDEO.video._number_of_animals = int(self.animal_number_input.text)
        self.popup.dismiss()

    def open(self, path, filename):
        print("opening video file")
        print("filename  ", filename)
        if len(filename) > 0:
            CHOSEN_VIDEO.set_chosen_item(filename[0])

        return not hasattr(self, 'enable_ROI_and_preprocessing_tabs')

    def enable_validation(self, path, filename):
        if filename:
            CHOSEN_VIDEO.set_chosen_item(filename[0])
            if CHOSEN_VIDEO.video.video_path is not None:
                self.video = CHOSEN_VIDEO.video
                self.old_video = CHOSEN_VIDEO.old_video
        return not (hasattr(self.video, '_has_been_assigned') or hasattr(self.old_video, "_has_been_assigned"))

    def create_preprocessing_type_and_number_popup(self):
        self.popup_container = BoxLayout()
        self.preprocessing_type_box = BoxLayout(orientation="vertical")
        self.preprocessing_type_label = Label(text='What animal are you tracking? [fish/flies]:\n')
        self.preprocessing_type_label.text_size = self.preprocessing_type_label.size
        self.preprocessing_type_label.texture_size = self.preprocessing_type_label.size
        self.preprocessing_type_box.add_widget(self.preprocessing_type_label)
        self.preprocessing_type_input = TextInput(text ='', multiline=False)
        self.preprocessing_type_box.add_widget(self.preprocessing_type_input)
        self.popup_container.add_widget(self.preprocessing_type_box)

        self.animal_number_box = BoxLayout(orientation="vertical")
        self.animal_number_label = Label(text='How many animals are you going to track:\n')
        self.animal_number_label.text_size = self.animal_number_label.size
        self.animal_number_label.texture_size = self.animal_number_label.size
        self.animal_number_box.add_widget(self.animal_number_label)
        self.animal_number_input = TextInput(text ='', multiline=False)
        self.animal_number_box.add_widget(self.animal_number_input)
        self.popup_container.add_widget(self.animal_number_box)
        self.popup = Popup(title='Animal model and number of animals',
                    content=self.popup_container,
                    size_hint=(.4,.4))

class VisualiseVideo(BoxLayout):
    def __init__(self, **kwargs):
        super(VisualiseVideo, self).__init__(**kwargs)
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
                                max= int(self.video_object._num_frames) - 1,
                                step=1,
                                value=0,
                                size_hint=(.8,1.))
        self.video_slider.bind(value=self.get_value)
        self.video_slider_lbl = Label( id = 'max_threshold_lbl')
        self.video_slider_lbl.text = "Frame number:" + str(int(self.video_slider.value))
        self.video_slider_lbl.text_size = self.video_slider_lbl.size
        self.video_slider_lbl.size = self.video_slider_lbl.texture_size
        self.video_slider_lbl.font_size = 16
        self.video_slider_lbl.halign =  "center"
        self.video_slider_lbl.valign = "middle"
        self.footer.add_widget(self.video_slider)
        self.footer.add_widget(self.video_slider_lbl)

    def visualise(self, trackbar_value, func = None):
        self.func = func
        # print('trackbar_value ', trackbar_value)
        sNumber = self.video_object.in_which_episode(int(trackbar_value))
        # print('seg number ', sNumber)
        sFrame = trackbar_value
        current_segment = sNumber
        if self.video_object._paths_to_video_segments:
            self.cap = cv2.VideoCapture(self.video_object._paths_to_video_segments[sNumber])
        #Get frame from video file
        if self.video_object._paths_to_video_segments:
            start = self.video_object._episodes_start_end[sNumber][0]
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame - start)
        else:
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbar_value)
        ret, self.frame = self.cap.read()
        if ret == True:
            if hasattr(CHOSEN_VIDEO.video, 'resolution_reduction'):
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
        # display image from the texture
        self.display_layout.texture = textureFrame
        self.initImW = self.width
        self.initImH = self.height

    def get_value(self, instance, value):
        self.video_slider_lbl.text = "Frame number:" + str(int(value))
        self.visualise(value, func = self.func)


class ROISelector(BoxLayout):
    def __init__(self,**kwargs):
        super(ROISelector, self).__init__(**kwargs)
        self.orientation = "vertical"
        self.ROIs = [] #store rectangles on the GUI
        self.ROIOut  = [] #pass them to opencv
        self.touches = [] #store touch events on the figure
        self.footer = BoxLayout()
        self.footer.size_hint = (1.,.1)
        self.btn_load_roi = Button(text = "load ROIs")
        self.btn_save_roi = Button(text = "save ROIs")
        self.btn_clear_roi = Button(text = "clear last ROI")
        self.btn_no_roi = Button(text = "do not use any ROI")
        self.footer.add_widget(self.btn_load_roi)
        self.footer.add_widget(self.btn_save_roi)
        self.footer.add_widget(self.btn_clear_roi)
        self.footer.add_widget(self.btn_no_roi)
        self.btn_save_roi.bind(on_press = self.save_ROI)
        self.btn_load_roi.bind(on_press = self.load_ROI)
        self.btn_no_roi.bind(on_press = self.no_ROI)
        self.btn_clear_roi.bind(on_press = self.delete_ROI)
        global CHOSEN_VIDEO
        CHOSEN_VIDEO.bind(chosen=self.do)

    def do(self, *args):
        if hasattr(CHOSEN_VIDEO.video, "video_path") and CHOSEN_VIDEO.video.video_path is not None:
            self.visualiser = VisualiseVideo()
            self.add_widget(self.visualiser)
            self.add_widget(self.footer)
            self.window = Window
            self.window.bind(on_resize=self.updateROIs)
            self.video_object = CHOSEN_VIDEO.video
            self.visualiser.visualise_video(self.video_object)
            if hasattr(CHOSEN_VIDEO, "old_video") and CHOSEN_VIDEO.old_video.ROI is not None:
                self.btn_load_roi.disabled = not hasattr(CHOSEN_VIDEO.old_video, "ROI")
            else:
                self.btn_load_roi.disabled = True

    def on_touch_down(self, touch):
        # print("touch down dispatch")
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
                rect = [self.touches[0], self.touches[-1]]
                sorted(rect, key=lambda x:x[1], reverse=True)
                rectS = np.diff(rect, axis=0)[0]
                with self.visualiser.display_layout.canvas:
                    Color(1, 1, 0,.5)
                    self.rect = Rectangle(pos=(rect[0][0], rect[0][1]), size=(rectS[0],rectS[1]))
                    self.ROIs.append(self.rect)
                    ratioH = self.visualiser.display_layout.height / self.visualiser.display_layout.texture.height
                    ratioW = self.visualiser.display_layout.width / self.visualiser.display_layout.texture.width
                    newRectP1 = (self.rect.pos[0] / ratioW, (self.rect.pos[0] + self.rect.size[0]) / ratioW)
                    newRectP2 = (self.rect.pos[1] / ratioH, (self.rect.pos[1] + self.rect.size[1]) / ratioH)
                    point1 = (int(newRectP1[0]), int(self.visualiser.frame.shape[1]-newRectP2[0]))
                    point2 = (int(newRectP1[1]), int(self.visualiser.frame.shape[1]-newRectP2[1]))
                    self.ROIOut.append([point1,point2])

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
        # print("saving ROI")
        if len(self.ROIOut) > 0:
            self.ROIcv2 = np.zeros_like(self.visualiser.frame,dtype='uint8')
            for p in self.ROIOut:
                # print("adding rectangles to ROI")
                # print("rect ", p)
                cv2.rectangle(self.ROIcv2, p[0], p[1], 255, -1)
        CHOSEN_VIDEO.video.ROI = self.ROIcv2
        CHOSEN_VIDEO.video.save()

    def no_ROI(self, *args):
        CHOSEN_VIDEO.video.ROI = np.ones_like(self.visualiser.frame ,dtype='uint8') * 255
        CHOSEN_VIDEO.apply_ROI = False
        CHOSEN_VIDEO.video.save()

    def load_ROI(self, *args):
        CHOSEN_VIDEO.video.ROI = CHOSEN_VIDEO.old_video.ROI

class BkgSubtraction(BoxLayout):
    def __init__(self, **kwargs):
        super(BkgSubtraction, self).__init__(**kwargs)
        self.bkg = None
        #set useful popups
        #saving:
        self.saving_popup = Popup(title='Saving',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.saving_popup.bind(on_open=self.save_bkg)
        #computing:
        self.computing_popup = Popup(title='Computing',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.computing_popup.bind(on_open=self.compute_bkg)
        global CHOSEN_VIDEO

    def subtract_bkg(self, *args):
        if hasattr(CHOSEN_VIDEO.old_video, "bkg") or hasattr(CHOSEN_VIDEO.video, "bkg"):
            if CHOSEN_VIDEO.old_video.bkg is not None:
                self.bkg = CHOSEN_VIDEO.old_video.bkg
            elif CHOSEN_VIDEO.video.bkg is not None:
                self.bkg = CHOSEN_VIDEO.video.bkg
        else:
            self.compute_bkg()

    def save_bkg(self, *args):
        CHOSEN_VIDEO.video.bkg = self.bkg
        CHOSEN_VIDEO.video.save()
        self.saving_popup.dismiss()

    def compute_bkg(self, *args):
        self.bkg = computeBkg(CHOSEN_VIDEO.video)
        self.save_bkg()
        self.computing_popup.dismiss()

class PreprocessingPreview(BoxLayout):
    def __init__(self, **kwargs):
        super(PreprocessingPreview, self).__init__(**kwargs)
        #get video path information and bind it
        self.visualiser = VisualiseVideo()
        global CHOSEN_VIDEO
        CHOSEN_VIDEO.bind(chosen=self.do)
        self.container_layout = BoxLayout(orientation = 'vertical', size_hint = (.3, 1.))
        self.add_widget(self.container_layout)
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
        #bkg sub switch
        self.bkg_subtractor_switch = Switch()
        #ROI label
        self.ROI_label = Label(text = 'apply ROI')
        self.ROI_label.text_size = self.ROI_label.size
        self.ROI_label.size = self.ROI_label.texture_size
        self.ROI_label.font_size = 16
        self.ROI_label.halign =  "center"
        self.ROI_label.valign = "middle"
        self.container_layout.add_widget(self.ROI_label)
        #ROI switch
        self.ROI_switch = Switch()
        self.container_layout.add_widget(self.ROI_switch)
        self.container_layout.add_widget(self.bkg_subtraction_label)
        self.container_layout.add_widget(self.bkg_subtractor_switch)
        self.count_scrollup = 0
        self.scale = 1
        self.ROI_popup_text = Label(text='It seems that the ROI you are trying to apply corresponds to the entire frame. Please, go to the ROI selection tab to select and save a ROI')
        self.ROI_popup = Popup(title='ROI warning',
            content=self.ROI_popup_text,
            size_hint=(.5,.5))
        self.ROI_popup_text.text_size = self.ROI_popup.size
        self.ROI_popup_text.size = self.ROI_popup_text.texture_size
        self.ROI_popup_text.font_size = 16
        self.ROI_popup_text.halign =  "center"
        self.ROI_popup_text.valign = "middle"
        self.saving_popup = Popup(title='Saving',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.ROI_popup_text.bind(size=lambda s, w: s.setter('text_size')(s, w))
        self.saving_popup.bind(on_open=self.save_preproc)

    def init_preproc_parameters(self):
        if hasattr(CHOSEN_VIDEO, "old_video") and CHOSEN_VIDEO.old_video._has_been_preprocessed == True:
            self.max_threshold = CHOSEN_VIDEO.old_video.max_threshold
            self.min_threshold = CHOSEN_VIDEO.old_video.min_threshold
            self.min_area = CHOSEN_VIDEO.old_video.min_area
            self.max_area = CHOSEN_VIDEO.old_video.max_area
        else:
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
        self.save_prec_params_btn = Button()
        self.save_prec_params_btn.text = "Load preprocessing params"
        self.save_prec_params_btn.bind(on_press = self.save_preproc_params)
        #create button to save parameter
        self.segment_video_btn = Button()
        self.segment_video_btn.text = "Segment video"
        # self.load_prec_params_btn.bind(on_press = self.laod_preproc_params)

    def do(self, *args):
        if hasattr(CHOSEN_VIDEO.video, "video_path") and CHOSEN_VIDEO.video.video_path is not None:
            self.init_preproc_parameters()
            self.ROI_switch.bind(active = self.apply_ROI)
            self.ROI_switch.active = CHOSEN_VIDEO.video.apply_ROI
            self.bkg_subtractor_switch.active = CHOSEN_VIDEO.video.subtract_bkg
            self.bkg_subtractor_switch.bind(active = self.apply_bkg_subtraction)
            self.bkg = CHOSEN_VIDEO.video.bkg
            self.ROI = CHOSEN_VIDEO.video.ROI if CHOSEN_VIDEO.video.ROI is not None else np.ones((CHOSEN_VIDEO.video._height, CHOSEN_VIDEO.video._width) ,dtype='uint8') * 255
            CHOSEN_VIDEO.video.ROI = self.ROI
            # print("ROI in do - preprocessing", self.ROI)
            self.init_segment_zero()

    def apply_ROI(self, instance, active):
        # print("applying ROI")
        CHOSEN_VIDEO.video.apply_ROI = active
        if active  == True:
            num_valid_pxs_in_ROI = len(sum(np.where(CHOSEN_VIDEO.video.ROI == 255)))
            num_pxs_in_frame = CHOSEN_VIDEO.video._height * CHOSEN_VIDEO.video._width
            self.ROI_is_trivial = num_pxs_in_frame == num_valid_pxs_in_ROI

            if CHOSEN_VIDEO.video.ROI is not None and not self.ROI_is_trivial:
                self.ROI = CHOSEN_VIDEO.video.ROI
            elif self.ROI_is_trivial:
                self.ROI_popup.open()
                instance.active = False
                CHOSEN_VIDEO.apply_ROI = False
        elif active == False:
            self.ROI = np.ones((CHOSEN_VIDEO.video._height, CHOSEN_VIDEO.video._width) ,dtype='uint8') * 255
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def apply_bkg_subtraction(self, instance, active):
        CHOSEN_VIDEO.video.subtract_bkg = active
        if CHOSEN_VIDEO.video.subtract_bkg == True:
            self.bkg_subtractor.subtract_bkg()
            self.bkg = self.bkg_subtractor.bkg
        else:
            self.bkg = None
        return active

    def add_widget_list(self):
        for w in self.w_list:
            self.container_layout.add_widget(w)

    def update_max_th_lbl(self,instance, value):
        self.max_threshold_lbl.text = "Max threshold:\n" +  str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def update_min_th_lbl(self,instance, value):
        self.min_threshold_lbl.text = "Min threshold:\n" + str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)
        # self.show_preprocessing(self.frame)

    def update_max_area_lbl(self,instance, value):
        self.max_area_lbl.text = "Max area:\n" + str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def update_min_area_lbl(self,instance, value):
        self.min_area_lbl.text = "Min area:\n" + str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def save_preproc_params(self):
        self.saving_popup.open()
        CHOSEN_VIDEO.video.max_threshold = self.max_threshold_slider.value
        CHOSEN_VIDEO.video.min_threshold = self.min_threshold_slider.value
        CHOSEN_VIDEO.video.min_area = self.min_area_slider.value
        CHOSEN_VIDEO.video.max_area = self.max_area_slider.value

    def save_preproc(self, *args):
        CHOSEN_VIDEO.video.save()
        self.saving_popup.dismiss()

    def init_segment_zero(self):
        # create instance of video shower
        self.visualiser = VisualiseVideo()
        self.add_widget(self.visualiser)
        self.visualiser.visualise_video(CHOSEN_VIDEO.video, func = self.show_preprocessing)
        self.currentSegment = 0
        #create layout for video and slider
        # self.button_layout = BoxLayout(orientation="horizontal", size_hint=(1.,.1))
        # self.button_layout.add_widget(self.load_prec_params_btn)
        # self.button_layout.add_widget(self.segment_video_btn)
        # self.video_layout_preprocessing.add_widget(self.button_layout)

    def show_preprocessing(self, frame):
        # pass frame to grayscale
        if len(frame.shape) > 2:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
        else:
            self.frame = frame
        # compute average intensity
        avIntensity = np.float32(np.mean(self.frame))
        # generate a averaged frame for segmentation
        self.av_frame = self.frame / avIntensity
        # threshold the frame according to the sliders' values
        print('this is the current background ', self.bkg)
        self.segmented_frame = segmentVideo(self.av_frame,
                                            int(self.min_threshold_slider.value),
                                            int(self.max_threshold_slider.value),
                                            self.bkg,
                                            self.ROI,
                                            self.bkg_subtractor_switch.active)
        #get information on the blobs find by thresholding
        boundingBoxes, miniFrames, _, _, _, goodContours, _ = blobExtractor(self.segmented_frame,
                                                                        self.frame,
                                                                        int(self.min_area_slider.value),
                                                                        int(self.max_area_slider.value))
        #draw the blobs on the original frame
        cv2.drawContours(self.frame, goodContours, -1, color=255, thickness = -1)
        #display the segmentation
        if self.count_scrollup != 0:
            self.dst = cv2.warpAffine(self.frame, self.M, (self.frame.shape[1], self.frame.shape[1]))
            buf1 = cv2.flip(self.dst,0)
        else:
            buf1 = cv2.flip(self.frame, 0)

        buf = buf1.tostring()
        textureFrame = Texture.create(size=(self.frame.shape[1], self.frame.shape[1]), colorfmt='luminance')
        textureFrame.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
        # display image from the texture
        self.visualiser.display_layout.texture = textureFrame

    def fromShowFrameToTexture(self, coords):
        """
        Maps coordinate in visualiser.display_layout (the image whose texture is the frame) to
        the coordinates of the original image
        """
        coords = np.asarray(coords)
        origFrameW = CHOSEN_VIDEO.video._width
        origFrameH = CHOSEN_VIDEO.video._height

        # actualFrameW, actualFrameH = self.visualiser.display_layout.size
        # self.y_offset = self.sliderBox.height + self.button_layout.height
        # self.x_offset = self.container_layout.width
        # coords[0] = coords[0] - self.x_offset
        # coords[1] = coords[1] - self.y_offset
        # wRatio = abs(origFrameW / actualFrameW)
        # hRatio = abs(origFrameH / actualFrameH)
        # ratios = np.asarray([wRatio, hRatio])
        # coords =  np.multiply(coords, ratios)
        # coords[1] = origFrameH - coords[1]
        return coords

    def on_touch_down(self, touch):
        self.touches = []
        # print( 'scrollup number ', self.count_scrollup)
        if self.parent is not None and self.visualiser.display_layout.collide_point(*touch.pos):
            # print( 'i think you are on the image')
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
                # frame = self.parent.frame
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

class Accumulator(BoxLayout):
    def __init__(self, **kwargs):
        super(Accumulator, self).__init__(**kwargs)
        global CHOSEN_VIDEO
        CHOSEN_VIDEO.bind(chosen=self.do)

    def do(self, *args):
        if hasattr(CHOSEN_VIDEO.video, "video_path") and CHOSEN_VIDEO.video.video_path is not None:
            if CHOSEN_VIDEO.video._has_been_assigned == True or  CHOSEN_VIDEO.old_video._has_been_assigned == True:
                return False
            else:
                return True

class Validator(BoxLayout):
    def __init__(self, **kwargs):
        super(Validator, self).__init__(**kwargs)
        global CHOSEN_VIDEO
        CHOSEN_VIDEO.bind(chosen=self.do)
        #it should not happen, but a warning just in case
        self.warning_popup = Popup(title = 'Warning',
                            content = Label(text = 'The video has not been tracked yet. Track it before performing validation.'),
                            size_hint = (.3,.3))
        self.warning_popup.bind(size=lambda s, w: s.setter('text_size')(s, w))
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
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

    def get_first_frame(self):
        if not hasattr(CHOSEN_VIDEO.video, 'first_frame_for_validation'):
            self.global_fragments = np.load(CHOSEN_VIDEO.old_video.global_fragments_path)
            max_distance_travelled_global_fragment = order_global_fragments_by_distance_travelled(self.global_fragments)[0]
            return max_distance_travelled_global_fragment.index_beginning_of_fragment
        else:
            return CHOSEN_VIDEO.video.first_frame_for_validation

    def do(self, *args):
        if hasattr(CHOSEN_VIDEO.video, "video_path") and CHOSEN_VIDEO.video.video_path is not None:
            # print("has been assigned ", CHOSEN_VIDEO.video._has_been_assigned)
            if CHOSEN_VIDEO.video._has_been_assigned == True:
                list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.old_video.blobs_path)
                self.blobs_in_video = list_of_blobs.blobs_in_video
            elif CHOSEN_VIDEO.old_video._has_been_assigned == True:
                CHOSEN_VIDEO.video = CHOSEN_VIDEO.old_video
                list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.old_video.blobs_path)
                self.blobs_in_video = list_of_blobs.blobs_in_video
            #init variables used for zooming
            self.count_scrollup = 0
            self.scale = 1
            #init elements in the self widget
            self.init_segmentZero()
        else:
            # print("no assignment done")
            self.warning_popup.open()

    def init_segmentZero(self):
        #create and add widget to visualise the tracked video
        self.visualiser = VisualiseVideo()
        self.add_widget(self.visualiser)
        #get colors to visualise trajectories and so on
        self.colors = get_spaced_colors_util(CHOSEN_VIDEO.video.number_of_animals)
        #create and add layout for buttons
        self.button_box = BoxLayout(orientation='vertical', size_hint=(.3,1.))
        self.add_widget(self.button_box)
        #create, add and bind button: go to next crossing
        self.next_cross_button = Button(id='crossing_btn', text='Next crossing', size_hint=(1,1))
        self.next_cross_button.bind(on_press=self.go_to_next_crossing)
        self.button_box.add_widget(self.next_cross_button)
        #create, add and bind button: go to previous crossing
        self.previous_cross_button = Button(id='crossing_btn', text='Previous crossing', size_hint=(1,1))
        self.previous_cross_button.bind(on_press=self.go_to_previous_crossing)
        self.button_box.add_widget(self.previous_cross_button)
        #create, add and bind button to go back to the first global fragments
        self.go_to_first_global_fragment_button = Button(id='back_to_first_gf_btn', text='First global fragment', size_hint=(1,1))
        self.go_to_first_global_fragment_button.bind(on_press = self.go_to_first_global_fragment)
        self.button_box.add_widget(self.go_to_first_global_fragment_button)
        #create, add and bind button: save groundtruth
        self.save_groundtruth_btn = Button(id='save_groundtruth_btn', text='Save updated identities',size_hint = (1,1))
        self.save_groundtruth_btn.bind(on_press=self.show_saving)
        self.save_groundtruth_btn.bind(on_release=self.save_groundtruth)
        self.save_groundtruth_btn.disabled = True
        # add button to the button layout
        self.button_box.add_widget(self.save_groundtruth_btn)
        # create button to compute accuracy with respect to the groundtruth entered by the user
        self.compute_accuracy_button = Button(id = "compute_accuracy_button", text = "Compute accuracy", size_hint  = (1.,1.))
        self.compute_accuracy_button.disabled = False
        self.compute_accuracy_button.bind(on_press = self.compute_accuracy_wrt_groundtruth)
        # add button to layout
        self.button_box.add_widget(self.compute_accuracy_button)
        #start visualising the video
        self.visualiser.visualise_video(CHOSEN_VIDEO.video, func = self.writeIds, frame_index_to_start = self.get_first_frame())

    def go_to_next_crossing(self,instance):
        non_crossing = True
        #get frame index from the slider initialised in visualiser
        frame_index = int(self.visualiser.video_slider.value)
        #for every subsequent frame check the blobs and stop if a crossing (or a jump) occurs
        while non_crossing == True:
            if frame_index < CHOSEN_VIDEO.video._num_frames:
                frame_index = frame_index + 1
                blobs_in_frame = self.blobs_in_video[frame_index]
                for blob in blobs_in_frame:
                    if not blob.is_a_fish_in_a_fragment:
                        non_crossing = False
                        self.visualiser.video_slider.value = frame_index
                        self.visualiser.visualise(frame_index, func = self.writeIds)
            else:
                break

    def go_to_previous_crossing(self,instance):
        non_crossing = True
        #get frame index from the slider initialised in visualiser
        frame_index = int(self.visualiser.video_slider.value)
        #for every subsequent frame check the blobs and stop if a crossing (or a jump) occurs
        while non_crossing == True:
            if frame_index > 0:
                frame_index = frame_index - 1
                blobs_in_frame = self.blobs_in_video[frame_index]
                for blob in blobs_in_frame:
                    if not blob.is_a_fish_in_a_fragment:
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
        self.visualiser.video_slider.value = frame_index
        self.visualiser.visualise(frame_index, func = self.writeIds)
        return True

    @staticmethod
    def getNearestCentroid(point, cents):
        """
        Finds the nearest neighbour in cents with respect to point (in 2D)
        """
        point = np.asarray(point)
        cents = np.asarray(cents)
        cents_x = cents[:,0]
        cents_y = cents[:,1]
        dist_x = cents_x - point[0]
        dist_y = cents_y - point[1]
        distances = dist_x**2 + dist_y**2
        return np.argmin(distances)

    @staticmethod
    def get_clicked_blob(point, contours):
        """
        Get contour in which point is contained
        """
        # print("point ", point)
        # print("contours ", contours)
        indices = [i for i, cnt in enumerate(contours) if cv2.pointPolygonTest(cnt, tuple(point), measureDist = False) >= 0]
        # print(indices)
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
        # centroids = np.asarray([getattr(blob, "centroid") for blob in blobs_in_frame])
        contours = [getattr(blob, "contour") for blob in blobs_in_frame]
        if self.scale != 1:
            #transforms the centroids to the visualised texture
            # centroids = [self.apply_affine_transform_on_point(self.M, centroid) for centroid in centroids]
            contours = [self.apply_affine_transform_on_contour(self.M, cnt) for cnt in contours]

        mouse_coords = self.fromShowFrameToTexture(mouse_coords)
        if self.scale != 1:
            mouse_coords = self.apply_inverse_affine_transform_on_point(self.M, mouse_coords)
        # centroid_ind = self.getNearestCentroid(mouse_coords, centroids) # compute the nearest centroid
        # print("contours shape, ", contours[0].shape)
        blob_ind = self.get_clicked_blob(mouse_coords, contours)
        # print("blob index ", blob_ind)
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
        if hasattr(CHOSEN_VIDEO.video, 'resolution_reduction'):
            if CHOSEN_VIDEO.video.resolution_reduction == 1:
                original_frame_width = CHOSEN_VIDEO.video._width
                original_frame_height = CHOSEN_VIDEO.video._height
            else:
                original_frame_width = int(CHOSEN_VIDEO.video._width * CHOSEN_VIDEO.video.resolution_reduction)
                original_frame_height = int(CHOSEN_VIDEO.video._height * CHOSEN_VIDEO.video.resolution_reduction)
        else:
            original_frame_width = int(CHOSEN_VIDEO.video._width)
            original_frame_height = int(CHOSEN_VIDEO.video._height)
        # print("--------------------------- width and height ", original_frame_width, original_frame_height)

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
        # print("frame shape in write Ids ", frame.shape)
        # cv2.putText(frame, str(self.visualiser.video_slider.value),(50,50), font, 1, [0, 0, 0], 3)
        print("****************************************************************")
        for blob in blobs_in_frame:
            print("8<-------------------------------------------------------------")

            if not blob.is_a_crossing:
                # print("______________________user generated id ", blob.user_generated_identity)
                int_centroid = blob.centroid.astype('int')
                if blob.user_generated_identity is None:
                    if hasattr(blob, "_identity_corrected_solving_duplication"):
                        if blob._identity_corrected_solving_duplication is not None:
                            cur_id = blob._identity_corrected_solving_duplication
                            cur_id_str = 'd-' + str(cur_id)
                        else:
                            cur_id = blob.identity
                            cur_id_str = str(cur_id)
                    else:
                        cur_id = blob.identity
                        cur_id_str = str(cur_id)
                else:
                    cur_id = blob.user_generated_identity
                    cur_id_str = 'u-' + str(cur_id)

                if type(cur_id) is 'int':
                    cv2.circle(frame, tuple(int_centroid), 2, self.colors[cur_id], -1)
                elif type(cur_id) is 'list':
                    cv2.circle(frame, tuple(int_centroid), 2, [255, 255, 255], -1)
                if blob._assigned_during_accumulation:
                    # we draw a circle in the centroid if the blob has been assigned during accumulation
                    cv2.putText(frame, 'a-' + cur_id_str,tuple(int_centroid), font, 1, self.colors[cur_id], 3)
                elif not blob._assigned_during_accumulation:
                    # we draw a cross in the centroid if the blob has been assigned during assignation
                    # cv2.putText(frame, 'x',tuple(int_centroid), font, 1,self.colors[cur_id], 1)
                    if blob.is_a_fish_in_a_fragment:
                        cv2.putText(frame, cur_id_str, tuple(int_centroid), font, 1, self.colors[cur_id], 2)
                    elif not blob.is_a_fish:
                        cv2.putText(frame, cur_id_str, tuple(int_centroid), font, 1, [255,255,255], 2)
                    elif blob.is_a_jump:
                        bounding_box = blob.bounding_box_in_frame_coordinates
                        cv2.putText(frame, cur_id_str, tuple(int_centroid), font, 1, self.colors[cur_id], 2)
                        cv2.rectangle(frame, bounding_box[0], bounding_box[1], (0, 255, 0) , 2)
                    elif blob.is_a_ghost_crossing:
                        bounding_box = blob.bounding_box_in_frame_coordinates
                        cv2.putText(frame, cur_id_str, tuple(int_centroid), font, 1, self.colors[cur_id], 2)
                        cv2.rectangle(frame, bounding_box[0], bounding_box[1], (255, 255, 255) , 2)
                    elif hasattr(blob , 'is_an_extreme_of_individual_fragment'):
                        cv2.putText(frame, cur_id_str, tuple(int_centroid), font, 1, self.colors[cur_id], 2)
                    else:
                        cv2.putText(frame, cur_id_str, tuple(int_centroid), font, 1, [0, 0, 0], 2)
            elif blob.is_a_crossing:
                # print("writing crossing ids")
                if blob.user_generated_identity is not None:
                    for centroid, identity in zip(blob._user_generated_centroids, blob._user_generated_identities):
                        centroid = centroid.astype('int')
                        cv2.putText(frame, str(identity), tuple(centroid), font, .5, self.colors[identity], 3)
                        cv2.circle(frame, tuple(centroid), 2, self.colors[identity], -1)
                else:
                    bounding_box = blob.bounding_box_in_frame_coordinates
                    cv2.rectangle(frame, bounding_box[0], bounding_box[1], (255, 0, 0) , 2)

                self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
                self._keyboard.bind(on_key_down=self._on_keyboard_down)

            print("\nidentity: ", blob._identity)
            if hasattr(blob, "_identity_corrected_solving_duplication"):
                print("corrected identity in duplications: ", blob._identity_corrected_solving_duplication)
        #     if hasattr(blob,"fragment_identifier"):
        #         print("fragment_identifier: ", blob.fragment_identifier)
        #     else:
        #         print("it does not have a fragment identifier")
        #     if hasattr(blob,"identities_before_crossing"):
        #         print("identity_before_crossing: ", blob.identities_before_crossing)
        #     if hasattr(blob,"identities_after_crossing"):
        #         print("identity_after_crossing: ", blob.identities_after_crossing)
        #     print("assigned during accumulation: ", blob.assigned_during_accumulation)
        #     if hasattr(blob,"frequencies_in_fragment"):
        #         print("frequencies in fragment: ", blob.frequencies_in_fragment)
        #     else:
        #         print("this blob does not have frequencies in fragment")
        #     print("P1_vector: ", blob.P1_vector)
            print("P2_vector: ", blob.P2_vector)
        #     print("is_a_fish: ", blob.is_a_fish)
        #     print("is_in_a_fragment: ", blob.is_in_a_fragment)
        #     print("is_a_fish_in_a_fragment: ", blob.is_a_fish_in_a_fragment)
        #     print("is_a_jump: ", blob.is_a_jump)
        #     print("is_a_ghost_crossing: ", blob.is_a_ghost_crossing)
        #     print("is_a_crossing: ", blob.is_a_crossing)
        #     print("next: ", blob.next)
        #     print("previous: ", blob.previous)
        # print("8<-------------------------------------------------------------")
        # Visualization of the process
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
        # print('********************** self.blob_to_modify.user_generated_identity, ', self.blob_to_modify.user_generated_identity)
        count_past_corrections = 1 #to take into account the modification already done in the current frame
        count_future_corrections = 0
        new_blob_identity = modified_blob.user_generated_identity
        if modified_blob.is_a_fish_in_a_fragment:
            current = modified_blob

            while current.next[0].is_a_fish_in_a_fragment:
                # print("propagating forward")
                current.next[0].user_generated_identity = current.user_generated_identity
                current = current.next[0]
                count_future_corrections += 1
                # print(count_future_corrections)

            current = modified_blob

            while current.previous[0].is_a_fish_in_a_fragment:
                # print("propagating backward")
                current.previous[0].user_generated_identity = current.user_generated_identity
                current = current.previous[0]
                count_past_corrections += 1
                # print(count_past_corrections)

            # print(self.count_user_generated_identities_dict)
            # print(new_blob_identity)
            # self.count_user_generated_identities_dict[new_blob_identity] = self.count_user_generated_identities_dict[new_blob_identity] + \
            #                                                             count_future_corrections + \
            #                                                             count_past_corrections
            # print("count_user_generated_identities_dict id, ", self.count_user_generated_identities_dict[new_blob_identity])
        #init and bind keyboard again
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def overwriteIdentity(self):
        # enable buttons to save corrected version and compute the accuracy
        self.save_groundtruth_btn.disabled = False
        self.compute_accuracy_button.disabled = False
        if not self.blob_to_modify.is_a_crossing:
            self.blob_to_modify.user_generated_identity = self.identity_update
            self.propagate_groundtruth_identity_in_individual_fragment()
        else:
            self.blob_to_modify._user_generated_centroids.append(self.user_generated_centroids)
            self.blob_to_modify._user_generated_identities.append(self.identity_update)
            # print("assigning ids and centroids to crossings:")
            # print(self.blob_to_modify._user_generated_identities)
            # print(self.blob_to_modify._user_generated_centroids)
        self.visualiser.visualise(trackbar_value = int(self.visualiser.video_slider.value), func=self.writeIds)

    def on_press_show_saving(selg, *args):
        self.show_saving()

    def save_groundtruth(self, *args):
        self.go_and_save()
        self.popup_saving.dismiss()

    def go_and_save(self):
        blobs_list = ListOfBlobs(blobs_in_video = self.blobs_in_video, path_to_save = CHOSEN_VIDEO.video.blobs_path)
        blobs_list.generate_cut_points(100)
        blobs_list.cut_in_chunks()
        blobs_list.save()
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
        self.identityInput.bind(on_text_validate=self.on_enter)
        self.popup.open()

    def show_blob_attributes(self, blob_to_explore):
        self.container = BoxLayout()
        self.blob_to_explore = blob_to_explore
        self.show_attributes_box = BoxLayout(orientation="vertical")
        #identity
        if blob_to_explore.user_generated_identity is not None:
            blob_to_explore_identity = blob_to_explore.user_generated_identity
        elif blob_to_explore._identity_corrected_solving_duplication is not None:
            blob_to_explore_identity = blob_to_explore._identity_corrected_solving_duplication
        else:
            blob_to_explore_identity = blob_to_explore.identity
        self.id_label = Label(text='Assigned identity: ' + str(blob_to_explore_identity))
        self.id_label.text_size = self.id_label.size
        self.id_label.texture_size = self.id_label.size
        #fragment identifier
        self.frag_id_label = Label(text='Fragment identifier: ' + str(blob_to_explore.fragment_identifier))
        self.frag_id_label.text_size = self.frag_id_label.size
        self.frag_id_label.texture_size = self.frag_id_label.size
        #is in a fragment
        self.in_a_fragment_label = Label(text='It is in an individual fragment: ' + str(blob_to_explore.is_in_a_fragment))
        self.in_a_fragment_label.text_size = self.in_a_fragment_label.size
        self.in_a_fragment_label.texture_size = self.in_a_fragment_label.size
        #is a fish
        self.fish_label = Label(text='It is a fish: ' + str(blob_to_explore.is_a_fish))
        self.fish_label.text_size = self.fish_label.size
        self.fish_label.texture_size = self.fish_label.size
        #is a ghost crossing
        self.ghost_crossing_label = Label(text='It is a ghost crossing: ' + str(blob_to_explore.is_a_ghost_crossing))
        self.ghost_crossing_label.text_size = self.ghost_crossing_label.size
        self.ghost_crossing_label.texture_size = self.ghost_crossing_label.size
        #is a jump
        self.jump_label = Label(text='It is a jump: ' + str(blob_to_explore.is_a_jump))
        self.jump_label.text_size = self.jump_label.size
        self.jump_label.texture_size = self.jump_label.size

        self.id_label.bind(size=lambda s, w: s.setter('text_size')(s, w))
        self.frag_id_label.bind(size=lambda s, w: s.setter('text_size')(s, w))
        self.in_a_fragment_label.bind(size=lambda s, w: s.setter('text_size')(s, w))
        self.ghost_crossing_label.bind(size=lambda s, w: s.setter('text_size')(s, w))
        self.jump_label.bind(size=lambda s, w: s.setter('text_size')(s, w))

        self.container.add_widget(self.show_attributes_box)
        self.show_attributes_box.add_widget(self.id_label)
        self.show_attributes_box.add_widget(self.frag_id_label)
        self.show_attributes_box.add_widget(self.in_a_fragment_label)
        self.show_attributes_box.add_widget(self.ghost_crossing_label)
        self.show_attributes_box.add_widget(self.jump_label)

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

    def frame_interval_popup(self):
        print("-----------------------------------------------------")
        self.popup_container = BoxLayout(orientation = "vertical")
        # self.popup_container_lbl = Label(text = "Input start and ending frame to compute accuracy against groundtruth")
        # self.popup_container_lbl.text_size = self.popup_container_lbl.size
        # self.popup_container_lbl.texture_size = self.popup_container_lbl.size
        # self.popup_container.add_widget(self.popup_container_lbl)
        # self.text_input_box = BoxLayout(orientation = "horizontal")
        #
        # self.start_frame_input = TextInput(text ='start', multiline=False)
        # self.end_frame_input = TextInput(text ='end', multiline=False)
        #
        # self.text_input_box.add_widget(self.start_frame_input)
        # self.text_input_box.add_widget(self.end_frame_input)
        # self.popup_container.add_widget(self.text_input_box)
        self.popup_start_end_groundtruth = Popup(title='Groundtruth Accuracy - Frame Interval',
                    content=self.popup_container,
                    size_hint=(.5,.5))
        # self.start_frame_input.bind(on_text_validate = self.on_enter_start_end)
        # self.end_frame_input.bind(on_text_validate = self.on_enter_start_end)
        self.popup_start_end_groundtruth.open()

    def on_enter_start_end(self,value):
        if self.start_frame_input.text != '':
            start = int(self.start_frame_input.text)
        else:
            start = 0

        if self.end_frame_input.text != '':
            end = int(self.end_frame_input.text)
        else:
            end = self.CHOSEN_VIDEO.video_object._num_frames
        self.groundtruth_interval = [start, end]
        self.popup_start_end_groundtruth.dismiss()


    def compute_accuracy_wrt_groundtruth(self, *args):
        # self.frame_interval_popup()
        start = int(raw_input('starting frame: '))
        end = int(raw_input('end frame:'))
        blobs_in_video = self.blobs_in_video[start:end+1]
        # blobs_in_video = self.blobs_in_video
        count_number_assignment_per_individual_assigned = {i: 0 for i in range(1,CHOSEN_VIDEO.video.number_of_animals+1)}
        count_number_assignment_per_individual_all = {i: 0 for i in range(1,CHOSEN_VIDEO.video.number_of_animals+1)}
        #create dictionary to store eventual corrections made by the user
        count_errors_identities_dict_assigned = {i:0 for i in range(1, CHOSEN_VIDEO.video.number_of_animals + 1)}
        count_errors_identities_dict_all = {i:0 for i in range(1, CHOSEN_VIDEO.video.number_of_animals + 1)}
        check_ground_truth = False
        frames_with_zeros = []
        for blobs_in_frame in blobs_in_video:
            for blob in blobs_in_frame:
                if hasattr(blob, "_identity_corrected_solving_duplication"):
                    if blob._identity_corrected_solving_duplication is None:
                        blob_identity = blob.identity
                    elif blob._identity_corrected_solving_duplication is not None:
                        blob_identity = blob._identity_corrected_solving_duplication
                else:
                    blob_identity = blob.identity
                if (blob.is_a_fish_in_a_fragment or\
                        blob.is_a_jump or\
                        blob.is_a_jumping_fragment or\
                        hasattr(blob,'is_an_extreme_of_individual_fragment')) and\
                        blob.user_generated_identity != -1: # we are not considering crossing or failures of the model area
                    if blob.user_generated_identity is not None and blob.user_generated_identity != blob.identity:
                        count_number_assignment_per_individual_all[blob.user_generated_identity] += 1
                        count_errors_identities_dict_all[blob.user_generated_identity] += 1
                        if blob.identity != 0:
                            count_number_assignment_per_individual_assigned[blob.user_generated_identity] += 1
                            count_errors_identities_dict_assigned[blob.user_generated_identity] += 1
                    elif blob.identity != 0:
                        count_number_assignment_per_individual_assigned[blob.identity] += 1
                        count_number_assignment_per_individual_all[blob.identity] += 1
                    elif (blob.identity == 0 or blob._identity_corrected_solving_duplication == 0) and blob.user_generated_identity is None:
                        print("frame number, ", blob.frame_number)
                        frames_with_zeros.append(blob.frame_number)
                        check_ground_truth = True

        if not check_ground_truth:
            self.individual_accuracy_assigned = {i : 1 - count_errors_identities_dict_assigned[i] / count_number_assignment_per_individual_assigned[i] for i in range(1, CHOSEN_VIDEO.video.number_of_animals + 1)}
            self.accuracy_assigned = np.mean(self.individual_accuracy_assigned.values())
            self.individual_accuracy = {i : 1 - count_errors_identities_dict_all[i] / count_number_assignment_per_individual_all[i] for i in range(1, CHOSEN_VIDEO.video.number_of_animals + 1)}
            self.accuracy = np.mean(self.individual_accuracy.values())
            # print("count_errors_identities_dict_assigned, ", count_errors_identities_dict_assigned)
            # print("count_errors_identities_dict_all, ", count_errors_identities_dict_all)
            # print("count_number_assignment_per_individual_assigned, ", count_number_assignment_per_individual_assigned)
            # print("count_number_assignment_per_individual_all, ", count_number_assignment_per_individual_all)
            # print("individual_accuracy_assigned, ", self.individual_accuracy_assigned)
            # print("accuracy_assigned, ", self.accuracy_assigned)
            # print("individual_accuracy, ", self.individual_accuracy)
            # print("accuracy, ", self.accuracy)

            self.plot_final_statistics()
            self.statistics_popup.open()
        else:
            print("there are fish with 0 identity in frame ", frames_with_zeros)

    def plot_final_statistics(self):
        content = BoxLayout()
        self.statistics_popup = Popup(title = "Statistics",
                                    content = content,
                                    size_hint = (.5, .5))

        fig, ax = plt.subplots(1)
        colors = get_spaced_colors_util(CHOSEN_VIDEO.video.number_of_animals, norm = True)

        width = .5
        plt.bar(self.individual_accuracy.keys(), self.individual_accuracy.values(), width, color=colors)
        plt.axhline(self.accuracy, color = 'k')
        ax.set_xlabel('individual')
        ax.set_ylabel('Individual accuracy')
        content.add_widget(FigureCanvasKivyAgg(fig))

class Root(TabbedPanel):

    def __init__(self, **kwargs):
        super(Root, self).__init__(**kwargs)
        self.bind(current_tab=self.content_changed_cb)

    def content_changed_cb(self, obj, value):
        print('CONTENT', value.content.id)
        print(type(value.content.id))

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
