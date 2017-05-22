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

import sys
sys.path.append('../')
sys.path.append('../utils')
import cv2
import numpy as np
from video import Video
from py_utils import getExistentFiles
from video_utils import computeBkg
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
        except:
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

    global CHOSEN_VIDEO
    CHOSEN_VIDEO = Chosen_Video()

    def on_enter(self,value):
        print(self.animal_type_input.text)
        print(self.animal_number_input.text)
        CHOSEN_VIDEO.video._animal_type = self.animal_type_input.text
        CHOSEN_VIDEO.video._number_of_animals = int(self.animal_number_input.text)
        self.popup.dismiss()

    def open(self, path, filename):
        print("opening video file")
        if filename:
            CHOSEN_VIDEO.set_chosen_item(filename[0])
            CHOSEN_VIDEO.video.create_session_folder()
            processes_list = ['bkg', 'ROI', 'preprocparams', 'preprocessing', 'pretraining', 'accumulation', 'training', 'assignment']
            #get existent files and paths to load them
            self.existentFiles, CHOSEN_VIDEO.old_video = getExistentFiles(CHOSEN_VIDEO.video, processes_list)
            print(self.existentFiles)
            if CHOSEN_VIDEO.old_video.animal_type is None and CHOSEN_VIDEO.old_video.number_of_animals is None:
                self.create_animal_type_and_number_popup()
                self.animal_type_input.bind(on_text_validate=self.on_enter)
                self.animal_number_input.bind(on_text_validate=self.on_enter)
                self.popup.open()
            else:
                CHOSEN_VIDEO.video._animal_type = CHOSEN_VIDEO.old_video.animal_type
                CHOSEN_VIDEO.video._number_of_animals = CHOSEN_VIDEO.old_video.number_of_animals

            self.video = CHOSEN_VIDEO.video
        return not hasattr(self, 'video')

    def create_animal_type_and_number_popup(self):
        self.popup_container = BoxLayout()
        self.animal_type_box = BoxLayout(orientation="vertical")
        self.animal_type_label = Label(text='What animal are you tracking? [fish/flies]:\n')
        self.animal_type_label.text_size = self.animal_type_label.size
        self.animal_type_label.texture_size = self.animal_type_label.size
        self.animal_type_box.add_widget(self.animal_type_label)
        self.animal_type_input = TextInput(text ='', multiline=False)
        self.animal_type_box.add_widget(self.animal_type_input)
        self.popup_container.add_widget(self.animal_type_box)

        self.animal_number_box = BoxLayout(orientation="vertical")
        self.animal_number_label = Label(text='How many animals are you going to track:\n')
        self.animal_number_label.text_size = self.animal_number_label.size
        self.animal_number_label.texture_size = self.animal_number_label.size
        self.animal_number_box.add_widget(self.animal_number_label)
        self.animal_number_input = TextInput(text ='', multiline=False)
        self.animal_number_box.add_widget(self.animal_number_input)
        self.popup_container.add_widget(self.animal_number_box)
        self.popup = Popup(title='Correcting identity',
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

    def visualise_video(self, video_object, func = None):
        self.video_object = video_object
        self.add_widget(self.display_layout)
        self.add_slider()
        self.add_widget(self.footer)
        self.cap = cv2.VideoCapture(self.video_object.video_path)
        self.visualise(0, func = func)

    def add_slider(self):
        self.video_slider = Slider(id='video_slider',
                                min=0,
                                max=self.video_object._num_frames,
                                step=1,
                                value=0,
                                size_hint=(1.,1.))
        self.video_slider.bind(value=self.get_value)
        self.footer.add_widget(self.video_slider)

    def visualise(self, trackbar_value, current_segment = 0, func = None):
        sNumber = self.video_object.in_which_episode(trackbar_value)
        print('seg number ', sNumber)
        print('trackbar_value ', trackbar_value)
        sFrame = trackbar_value

        if sNumber != current_segment: # we are changing segment
            print('Changing segment...')
            currentSegment = sNumber
            if self.video_object._paths_to_video_segments:
                self.cap = cv2.VideoCapture(self.video_object._paths_to_video_segments[sNumber])
        #Get frame from video file
        if self.video_object._paths_to_video_segments:
            start = self.video_object._episodes_start_end[sNumber][0]
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame - start)
        else:
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbar_value)
        ret, frame = self.cap.read()
        if func is None:
            func = self.simple_visualisation
        func(frame)

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
        self.visualise(value)

class ROISelector(BoxLayout):
    def __init__(self,**kwargs):
        super(ROISelector, self).__init__(**kwargs)
        self.orientation = "vertical"
        self.ROIs = [] #store rectangles on the GUI
        self.ROIOut  = [] #pass them to opencv
        self.touches = [] #store touch events on the figure
        self.visualiser = VisualiseVideo()

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
        if hasattr(CHOSEN_VIDEO, "video"):
            self.add_widget(self.visualiser)
            self.add_widget(self.footer)
            self.video_object = CHOSEN_VIDEO.video
            self.visualiser.visualise_video(self.video_object)
            self.ROIcv2 = np.zeros_like(self.visualiser.frame,dtype='uint8')
            if hasattr(CHOSEN_VIDEO, "old_video"):
                self.btn_load_roi.disabled = not hasattr(CHOSEN_VIDEO.old_video, "ROI")
            else:
                self.btn_load_roi.disabled = True

    def on_touch_down(self, touch):
        print("touch down dispatch")
        self.touches = []
        if self.visualiser.display_layout.collide_point(*touch.pos):
            self.touches.append(touch.pos)
        else:
            self.disable_touch_down_outside_collided_widget()

    def disable_touch_down_outside_collided_widget(self):
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
                    ratioH = self.visualiser.initImH / self.visualiser.display_layout.texture.height
                    ratioW = self.visualiser.initImW / self.visualiser.display_layout.texture.width
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

    def save_ROI(self, *args):
        if len(self.ROIOut) > 0:
            for p in self.ROIOut:
                cv2.rectangle(self.ROIcv2,p[0], p[1],255,-1)
        CHOSEN_VIDEO.video.ROI = self.ROIcv2
        CHOSEN_VIDEO.video.save()

    def no_ROI(self, *args):
        CHOSEN_VIDEO.video.ROI = np.ones_like(self.frame,dtype='uint8') * 255
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
        if hasattr(CHOSEN_VIDEO.old_video, "bkg"):
            self.bkg = CHOSEN_VIDEO.old_video.bkg
        else:
            self.compute_bkg()

    def save_bkg(self, *args):
        CHOSEN_VIDEO.video.bkg = self.bkg
        CHOSEN_VIDEO.video.save()
        self.saving_popup.dismiss()

    def compute_bkg(self, *args):
        self.bkg = computeBkg(CHOSEN_VIDEO.video)
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
        self.container_layout.add_widget(self.bkg_subtraction_label)
        #bkg sub switch
        self.bkg_subtractor_switch = Switch()
        self.container_layout.add_widget(self.bkg_subtractor_switch)

        self.saving_popup = Popup(title='Saving',
            content=Label(text='wait ...'),
            size_hint=(.3,.3))
        self.saving_popup.bind(on_open=self.save_preproc)

    def init_preproc_parameters(self):
        if CHOSEN_VIDEO.old_video._has_been_preprocessed == True:
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
        if hasattr(CHOSEN_VIDEO, "video"):
            self.init_preproc_parameters()
            self.ROI = CHOSEN_VIDEO.video.ROI
            self.bkg = self.check_bkg()
            self.bkg_subtractor_switch.bind(active = self.apply_bkg_subtraction)
            self.bkg_subtractor_switch.active = self.flag_bkg
            print 'active? ', self.bkg_subtractor_switch.active
            #bind the switch to the background loader / computer

            self.init_segment_zero()

    def apply_bkg_subtraction(self, instance, active):
        CHOSEN_VIDEO.get_bkg_path()
        print 'flag: ', self.flag_bkg
        print 'instance ', instance
        print 'active ', active
        CHOSEN_VIDEO.video.subtract_bkg = active
        if CHOSEN_VIDEO.video.subtract_bkg == True:
            self.bkg_subtractor.subtract_bkg()
        self.visualiser.visualise_video(self.videoSlider.value)
        return self.flag_bkg

    def add_widget_list(self):
        for w in self.w_list:
            self.container_layout.add_widget(w)

    def update_max_th_lbl(self,instance, value):
        self.max_threshold_lbl.text = "Max threshold:\n" +  str(int(value))
        self.show_preprocessing(self.videoSlider.value)

    def update_min_th_lbl(self,instance, value):
        self.min_threshold_lbl.text = "Min threshold:\n" + str(int(value))
        self.show_preprocessing(self.videoSlider.value)

    def update_max_area_lbl(self,instance, value):
        self.max_area_lbl.text = "Max area:\n" + str(int(value))
        self.show_preprocessing(self.videoSlider.value)

    def update_min_area_lbl(self,instance, value):
        self.min_area_lbl.text = "Min area:\n" + str(int(value))
        self.show_preprocessing(self.videoSlider.value)

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
        self.visualiser.visualise_video(self.CHOSEN_VIDEO.video, func = self.show_preprocessing)
        self.currentSegment = 0
        #create layout for video and slider
        self.video_layout_preprocessing = BoxLayout(orientation = 'vertical')
        self.add_widget(self.video_layout_preprocessing)

        #create image to store the video
        self.video_layout_preprocessing.add_widget(self.video_shower)
        self.video_layout_preprocessing.add_widget(self.sliderBox)
        self.button_layout = BoxLayout(orientation="horizontal", size_hint=(1.,.1))
        self.button_layout.add_widget(self.load_prec_params_btn)
        self.button_layout.add_widget(self.segment_video_btn)
        self.video_layout_preprocessing.add_widget(self.button_layout)

        self.show_preprocessing()

    def show_preprocessing(self, value = 0):
        # pass frame to grayscale
        self.frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
        # compute average intensity
        avIntensity = np.float32(np.mean(self.frame))
        # generate a averaged frame for segmentation
        self.av_frame = self.frame / avIntensity
        # threshold the frame according to the sliders' values
        print 'this is the current background ', self.bkg
        self.segmented_frame = segmentVideo(self.av_frame,
                                            int(self.min_threshold_slider.value),
                                            int(self.max_threshold_slider.value),
                                            self.bkg,
                                            self.ROI,
                                            self.bkg_subtractor_switch.active)
        #get information on the blobs find by thresholding
        boundingBoxes, miniFrames, _, _, _, goodContours, bkgSamples = blobExtractor(self.segmented_frame,
                                                                                    self.frame,
                                                                                    int(self.min_area_slider.value),
                                                                                    int(self.max_area_slider.value),
                                                                                    self.video_height,
                                                                                    self.video_width)
        #draw the blobs on the original frame
        cv2.drawContours(self.frame, goodContours, -1, color=255, thickness = -1)
        #display the segmentation
        if self.video_shower.count_scrollup != 0:
            self.dst = cv2.warpAffine(self.frame,self.video_shower.M,(self.video_width, self.video_height))
            buf1 = cv2.flip(self.dst,0)
        else:
            buf1 = cv2.flip(self.frame, 0)

        buf = buf1.tostring()
        textureFrame = Texture.create(size=(self.video_width, self.video_height), colorfmt='luminance')
        textureFrame.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
        # display image from the texture
        self.video_shower.show_frame_preprocessing.texture = textureFrame

    def on_slider_value_change(self,instance,value):
        self.sliderValue.text = 'frame ' + str(int(value))
        self.show_preprocessing(int(value))

    def fromShowFrameToTexture(self, coords):
        """
        Maps coordinate in show_frame_preprocessing (the image whose texture is the frame) to
        the coordinates of the original image
        """
        coords = np.asarray(coords)
        origFrameW = self.video_width
        origFrameH = self.video_height

        actualFrameW, actualFrameH = self.video_shower.show_frame_preprocessing.size
        self.y_offset = self.sliderBox.height + self.button_layout.height
        self.x_offset = self.container_layout.width
        coords[0] = coords[0] - self.x_offset
        coords[1] = coords[1] - self.y_offset
        wRatio = abs(origFrameW / actualFrameW)
        hRatio = abs(origFrameH / actualFrameH)
        ratios = np.asarray([wRatio, hRatio])
        coords =  np.multiply(coords, ratios)
        coords[1] = origFrameH - coords[1]
        return coords

    def on_touch_down(self, touch):
        self.touches = []
        print 'scrollup number ', self.count_scrollup
        if self.parent is not None and self.show_frame_preprocessing.collide_point(*touch.pos):
            print 'i think you are on the image'
            if touch.button == 'scrollup':
                self.count_scrollup += 1

                coords = self.parent.parent.fromShowFrameToTexture(touch.pos)
                rows,cols = self.parent.parent.frame.shape
                self.scale = 1.5 * self.count_scrollup
                self.M = cv2.getRotationMatrix2D((coords[0],coords[1]),0,self.scale)
                self.dst = cv2.warpAffine(self.parent.parent.frame,self.M,(cols,rows))
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]),
                                            colorfmt='luminance')
                textureFrame.blit_buffer(buf,
                                        colorfmt='luminance', bufferfmt='ubyte')
                self.show_frame_preprocessing.texture = textureFrame

            elif touch.button == 'scrolldown':
                # frame = self.parent.frame
                coords = self.parent.parent.fromShowFrameToTexture(touch.pos)
                rows,cols = self.parent.parent.frame.shape
                self.dst = self.parent.parent.frame
                buf1 = cv2.flip(self.dst, 0)
                buf = buf1.tostring()
                textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]),
                                            colorfmt='luminance')
                textureFrame.blit_buffer(buf,
                                        colorfmt='luminance',
                                        bufferfmt='ubyte')
                self.show_frame_preprocessing.texture = textureFrame
                self.count_scrollup = 0

        else:
            self.scale = 1

# class VideoShowerPreprocessing(BoxLayout):
#     def __init__(self, **kwargs):
#         super(VideoShowerPreprocessing, self).__init__(**kwargs)
#         self.id="video_layout_validation"
#         self.orientation = "vertical"
#         self.show_frame_preprocessing = Image(keep_ratio=False, allow_stretch=True, size_hint = (1.,1.))
#         self.add_widget(self.show_frame_preprocessing)
#         self.count_scrollup = 0
#         self.scale = 1
#
#     def on_touch_down(self, touch):
#         self.touches = []
#         print 'scrollup number ', self.count_scrollup
#         if self.parent is not None and self.show_frame_preprocessing.collide_point(*touch.pos):
#             print 'i think you are on the image'
#             if touch.button == 'scrollup':
#                 self.count_scrollup += 1
#
#                 coords = self.parent.parent.fromShowFrameToTexture(touch.pos)
#                 rows,cols = self.parent.parent.frame.shape
#                 self.scale = 1.5 * self.count_scrollup
#                 self.M = cv2.getRotationMatrix2D((coords[0],coords[1]),0,self.scale)
#                 self.dst = cv2.warpAffine(self.parent.parent.frame,self.M,(cols,rows))
#                 buf1 = cv2.flip(self.dst, 0)
#                 buf = buf1.tostring()
#                 textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='luminance')
#                 textureFrame.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
#                 self.show_frame_preprocessing.texture = textureFrame
#
#             elif touch.button == 'scrolldown':
#                 # frame = self.parent.frame
#                 coords = self.parent.parent.fromShowFrameToTexture(touch.pos)
#                 rows,cols = self.parent.parent.frame.shape
#                 self.dst = self.parent.parent.frame
#                 buf1 = cv2.flip(self.dst, 0)
#                 buf = buf1.tostring()
#                 textureFrame = Texture.create(size=(self.dst.shape[1], self.dst.shape[0]), colorfmt='luminance')
#                 textureFrame.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
#                 self.show_frame_preprocessing.texture = textureFrame
#                 self.count_scrollup = 0
#
#         else:
#             self.scale = 1

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
    Config.write()
    def build(self):
        return MainWindow()

if __name__ == '__main__':
    idtrackerdeepApp().run()
