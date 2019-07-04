# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

from __future__ import absolute_import, division, print_function
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.graphics import *
from kivy.uix.switch import Switch
from kivy.uix.slider import Slider
from idtrackerai.gui.visualise_video import VisualiseVideo
from idtrackerai.gui.bkg_subtraction import BkgSubtraction
from idtrackerai.gui.kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process
from functools import partial
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import numpy as np, time, cv2, matplotlib, matplotlib.pyplot as plt
from idtrackerai.preprocessing.segmentation import segment_frame, resegment
from idtrackerai.utils.segmentation_utils import blob_extractor
from scipy.stats import mode
from confapp import conf

from idtrackerai.preprocessing_preview_api import PreprocessingPreviewAPI

class PreprocessingPreview(PreprocessingPreviewAPI, BoxLayout):


    def __init__(self,
        chosen_video = None,
        deactivate_preprocessing = None,
        deactivate_tracking = None,
        **kwargs
    ):
        PreprocessingPreviewAPI.__init__(self, chosen_video)

        ## GUI ###################################################

        # To remove in the future
        global CHOSEN_VIDEO, DEACTIVATE_PREPROCESSING, DEACTIVATE_TRACKING
        CHOSEN_VIDEO             = chosen_video
        DEACTIVATE_PREPROCESSING = deactivate_preprocessing
        DEACTIVATE_TRACKING      = deactivate_tracking

        self.deactivate_preprocessing = deactivate_preprocessing
        self.deactivate_tracking      = deactivate_tracking

        BoxLayout.__init__(self, **kwargs)

        self.chosen_video.bind(chosen=self.do)
        self.container_layout = BoxLayout(orientation = 'vertical', size_hint = (.3, 1.))
        self.reduce_resolution_btn = Button(text = "Reduce resolution")
        self.bkg_subtractor = BkgSubtraction(orientation = 'vertical', chosen_video = self.chosen_video)
        self.bkg_subtraction_label = CustomLabel(font_size = 14, text = "background subtraction")
        self.bkg_subtractor_switch = Switch()
        self.check_segmentation_consistency_label = CustomLabel(font_size = 14,
                                                                text = 'check segmentation consistency')
        self.check_segmentation_consistency_switch = Switch()
        self.ROI_label = CustomLabel(font_size = 14, text = 'apply ROI')
        self.ROI_switch = Switch()
        self.container_layout.add_widget(self.reduce_resolution_btn)
        self.container_layout.add_widget(self.ROI_label)
        self.container_layout.add_widget(self.ROI_switch)
        self.container_layout.add_widget(self.bkg_subtraction_label)
        self.container_layout.add_widget(self.bkg_subtractor_switch)
        self.container_layout.add_widget(self.check_segmentation_consistency_label)
        self.container_layout.add_widget(self.check_segmentation_consistency_switch)
        self.count_scrollup = 0
        self.scale = 1
        self.has_been_executed = False
        self.ROI_popup_text = CustomLabel(font_size = 14, text='It seems that the ROI you are trying to apply corresponds to the entire frame. Please, go to the ROI selection tab to select and save a ROI')
        self.ROI_popup = Popup(title='ROI warning',
            content=self.ROI_popup_text,
            size_hint=(.5,.5))
        self.saving_popup = Popup(title='Saving',
            content=CustomLabel(text='wait ...'),
            size_hint=(.3,.3))

        self.help_button_preprocessing = HelpButton()
        self.help_button_preprocessing.size_hint = (1.,1.)
        self.help_button_preprocessing.create_help_popup("Preprocessing",\
                                                "The aim of this part of the process is to separate the animals from the background," +
                                                " by setting the following parameters." +
                                                "\n1) Apply ROI: Allows to consider only a region of interest on the frame. " +
                                                "Select it by using the table ROI selection." +
                                                "\n2) background subtraction: Perform background subtraction by computing " +
                                                "a model of the background on the fly."+
                                                "\n3) Check segmentation consiscenty: Raise a warning if there are more blobs than animals during the" +
                                                "segmentation process and ask to readjust the preprocessing parameters for those frames" +
                                                "\n4) Max\Min intensity: Set the maximum intensity used to separate "+
                                                "the animals from the background."+
                                                "\n5) Max\Min area: Filter the blob by area.")
        self.help_button_preprocessing.help_popup.size_hint = (.5,.6)
        ## GUI ###################################################



    def init_preproc_parameters(self):
        super().init_preproc_parameters()

        self.max_threshold_slider = Slider(id = 'max_threhsold', min = conf.MIN_THRESHOLD, max = conf.MAX_THRESHOLD, value = self.max_threshold, step = 1)
        self.max_threshold_lbl = CustomLabel(font_size = 14, id = 'max_threshold_lbl', text = "Max intensity: " + str(int(self.max_threshold_slider.value)))
        self.min_threshold_slider = Slider(id='min_threshold_slider', min = conf.MIN_THRESHOLD, max = conf.MAX_THRESHOLD, value = self.min_threshold, step = 1)
        self.min_threshold_lbl = CustomLabel(font_size = 14, id='min_threshold_lbl', text = "Min intensity:" + str(int(self.min_threshold_slider.value)))
        self.max_area_slider = Slider(id='max_area_slider', min = conf.MAX_AREA_LOWER, max = conf.MAX_AREA_UPPER, value = self.max_area, step = 1)
        self.max_area_lbl = CustomLabel(font_size = 14, id='max_area_lbl', text = "Max area:" + str(int(self.max_area_slider.value)))
        self.min_area_slider = Slider(id='min_area_slider', min = conf.MIN_AREA_LOWER, max = conf.MIN_AREA_UPPER, value = self.min_area, step = 1)
        self.min_area_lbl = CustomLabel(font_size = 14, id='min_area_lbl', text = "Min area:" + str(int(self.min_area_slider.value)))
        self.w_list = [ self.max_threshold_lbl, self.max_threshold_slider,
                        self.min_threshold_lbl, self.min_threshold_slider,
                        self.max_area_lbl, self.max_area_slider,
                        self.min_area_lbl, self.min_area_slider]
        self.add_widget_list()
        self.max_threshold_slider.bind(value=self.update_max_th_lbl)
        self.min_threshold_slider.bind(value=self.update_min_th_lbl)
        self.max_area_slider.bind(value=self.update_max_area_lbl)
        self.min_area_slider.bind(value=self.update_min_area_lbl)
        self.tracking_interval = Button(id = "tracking_interval", text = "Tracking interval", size_hint  = (1.,1.))
        self.tracking_interval.disabled = False
        self.tracking_interval.bind(on_press = self.get_tracking_interval)
        self.container_layout.add_widget(self.tracking_interval)
        self.segment_video_btn = Button(text = "Segment video")
        self.container_layout.add_widget(self.segment_video_btn)
        self.container_layout.add_widget(self.help_button_preprocessing)



    def get_tracking_interval(self, *args):
        self.create_frame_interval_popup()
        self.popup_tracking_interval.open()

    def create_frame_interval_popup(self):
        self.tracking_interval_container = BoxLayout(orientation = "vertical")
        self.tracking_interval_label = CustomLabel(text = "Insert the video intervals in frames separated by commas (e.g. 100-250, 300-500) on which the tracking will be performed")
        self.tracking_interval_container.add_widget(self.tracking_interval_label)
        self.tracking_interval_text_input = TextInput(text = '', multiline=False)
        self.tracking_interval_container.add_widget(self.tracking_interval_text_input)
        self.tracking_interval_text_input.bind(on_text_validate = self.on_enter_tracking_interval)

        self.popup_tracking_interval = Popup(title='Video tracking interval',
                    content=self.tracking_interval_container,
                    size_hint=(.4,.4))

    def on_enter_tracking_interval(self, value):
        start, end = self.tracking_interval_text_input.text.split('-')
        self.chosen_video.video._tracking_interval = (int(start), int(end))
        self.popup_tracking_interval.dismiss()

    def activate_ROI_switch(self, *args):
        if hasattr(self,'visualiser'):
            self.ROI_switch.active = True

    def deactivate_ROI_switch(self, *args):
        if hasattr(self,'visualiser'):
            self.ROI_switch.active = False

    def do(self, *args):
        if self.chosen_video.video is not None and self.chosen_video.video.video_path is not None:

            self.init_preview()

            self.create_resolution_reduction_popup()
            self.res_red_input.bind(on_text_validate = self.on_enter_res_red_coeff)
            self.reduce_resolution_btn.bind(on_press = self.open_resolution_reduction_popup)
            # self.ROI_switch.bind(active = self.apply_ROI)
            num_valid_pxs_in_ROI = len(sum(np.where(self.chosen_video.video.ROI == 255)))
            num_pxs_in_frame = self.chosen_video.video.height * self.chosen_video.video.width
            self.ROI_switch.active = not (num_pxs_in_frame == num_valid_pxs_in_ROI or num_valid_pxs_in_ROI == 0)
            self.bkg_subtractor_switch.active = False
            self.bkg_subtractor_switch.bind(active = self.apply_bkg_subtraction)
            self.create_number_of_animals_popup()
            self.num_of_animals_input.bind(on_text_validate = self.set_number_of_animals)
            self.num_of_animals_popup.bind(on_dismiss = self.show_segmenting_popup)
            self.create_computational_steps_popups()
            self.segmenting_popup.bind(on_open = self.compute_list_of_blobs)
            self.segmenting_popup.bind(on_dismiss = self.show_consistency_popup)
            self.consistency_popup.bind(on_open = self.check_segmentation_consistency)
            self.consistency_success_popup.bind(on_open = self.save_list_of_blobs_segmented)
            self.consistency_success_popup.bind(on_dismiss = self.model_area_and_crossing_detector)
            self.DCD_popup.bind(on_open = self.train_and_apply_crossing_detector)
            self.DCD_popup.bind(on_dismiss = self.plot_crossing_detection_statistics)
            self.has_been_executed = True
            self.segment_video_btn.bind(on_press = self.segment)
            self.bkg_subtractor.visualiser = self.visualiser
            self.bkg_subtractor.shower = self.show_preprocessing



    def compute_list_of_blobs(self, *args):
        super().compute_list_of_blobs(*args)
        self.segmenting_popup.dismiss()

    def segment(self, *args):
        super().segment(
            self.min_threshold_slider.value,
            self.max_threshold_slider.value,
            self.min_area_slider.value,
            self.max_area_slider.value
        )
        if not hasattr(self, 'number_of_animals'):
            self.num_of_animals_input.text = str(mode(self.number_of_detected_blobs)[0][0])
        else:
            self.num_of_animals_input.text = str(self.number_of_animals)
        self.num_of_animals_popup.open()



    def visualise_resegmentation(self, frame_number):
        def update_resegmentation_paramenters(instance, value):
            if instance.id == "max_threshold_slider":
                max_threshold_lbl.text = "Max intensity:\n" +  str(int(value))
                self.new_preprocessing_parameters['max_threshold'] = value
            elif instance.id == "min_threshold_slider":
                min_threshold_lbl.text = "Min intensity:\n" +  str(int(value))
                self.new_preprocessing_parameters['min_threshold'] = value
            elif instance.id == "max_area_slider":
                max_area_lbl.text = "Max area:\n" +  str(int(value))
                self.new_preprocessing_parameters['max_area'] = value
            elif instance.id == "min_area_slider":
                min_area_lbl.text = "Min area:\n" +  str(int(value))
                self.new_preprocessing_parameters['min_area'] = value
            resegmentation_visualiser.visualise(frame_number,
                                                func = partial(self.show_preprocessing,
                                                visualiser = resegmentation_visualiser,
                                                sliders = [min_threshold_slider, max_threshold_slider,
                                                        min_area_slider, max_area_slider],
                                                hide_video_slider = True))

        self.resegmentation_step_finished = False
        self.resegmentation_box = BoxLayout()
        resegmentation_controls_box = BoxLayout(orientation = "vertical", size_hint = (.3, 1.))
        max_threshold_slider = Slider(id = 'max_threshold_slider', min = conf.MIN_THRESHOLD, max = conf.MAX_THRESHOLD,
                        value = self.new_preprocessing_parameters['max_threshold'],
                        step = 1)
        max_threshold_lbl = CustomLabel(font_size = 14,
                text = "Max intensity: " + str(int(max_threshold_slider.value)))
        min_threshold_slider = Slider(id = 'min_threshold_slider', min = conf.MIN_THRESHOLD, max = conf.MAX_THRESHOLD,
                        value = self.new_preprocessing_parameters['min_threshold'],
                        step = 1)
        min_threshold_lbl = CustomLabel(font_size = 14,
                text = "Min intensity:" + str(int(min_threshold_slider.value)))
        max_area_slider = Slider(id = 'max_area_slider', min = conf.MAX_AREA_LOWER, max = conf.MAX_AREA_UPPER,
                            value = self.new_preprocessing_parameters['max_area'],
                            step = 1)
        max_area_lbl = CustomLabel(font_size = 14,
                        text = "Max area:" + str(int(max_area_slider.value)))
        min_area_slider = Slider(id = 'min_area_slider', min = conf.MIN_AREA_LOWER, max = conf.MIN_AREA_UPPER,
                            value = self.new_preprocessing_parameters['min_area'],
                            step = 1)
        min_area_lbl = CustomLabel(font_size = 14,
                            text = "Min area:" + str(int(min_area_slider.value)))
        resegment_btn = Button(text = "resegment")

        w_list = [ max_threshold_lbl, max_threshold_slider, min_threshold_lbl,
                    min_threshold_slider, max_area_lbl, max_area_slider,
                    min_area_lbl, min_area_slider, resegment_btn]
        max_threshold_slider.bind(value = update_resegmentation_paramenters)
        min_threshold_slider.bind(value = update_resegmentation_paramenters)
        max_area_slider.bind(value = update_resegmentation_paramenters)
        min_area_slider.bind(value = update_resegmentation_paramenters)
        resegment_btn.bind(on_release = self.resegment_and_update)

        [resegmentation_controls_box.add_widget(w) for w in w_list]
        resegmentation_visualiser = VisualiseVideo(chosen_video = self.chosen_video)
        self.create_areas_figure(visualiser = resegmentation_visualiser)
        resegmentation_visualiser.visualise_video(self.chosen_video.video,
                                                func = partial(self.show_preprocessing,
                                                visualiser = resegmentation_visualiser),
                                                frame_index_to_start = frame_number)
        resegmentation_visualiser.visualise(frame_number,
                                            func = partial(self.show_preprocessing,
                                            visualiser = resegmentation_visualiser,
                                            sliders = [min_threshold_slider, max_threshold_slider,
                                                    min_area_slider, max_area_slider],
                                            hide_video_slider = True))
        self.resegmentation_box.add_widget(resegmentation_controls_box)
        self.resegmentation_box.add_widget(resegmentation_visualiser)
        self.consistency_fail_popup_content.add_widget(self.resegmentation_box)

    def resegment_and_update(self, *args):
        for frame_number in self.frames_with_more_blobs_than_animals:
            self.chosen_video.video._maximum_number_of_blobs = resegment(self.chosen_video.video,
                                                frame_number,
                                                self.chosen_video.list_of_blobs,
                                                self.new_preprocessing_parameters)
            if self.chosen_video.video._maximum_number_of_blobs <= self.chosen_video.video.number_of_animals:
                self.chosen_video.video._resegmentation_parameters.append((frame_number, self.new_preprocessing_parameters))
        self.frames_with_more_blobs_than_animals, self.chosen_video.video._maximum_number_of_blobs = self.chosen_video.list_of_blobs.check_maximal_number_of_blob(self.chosen_video.video.number_of_animals, return_maximum_number_of_blobs = True)
        self.resegmentation_step_finished = True

    def resegmentation(self, *args):
        if len(self.frames_with_more_blobs_than_animals) > 0 and self.resegmentation_step_finished == True:
            if hasattr(self, 'resegmentation_box'):
                self.consistency_fail_popup_content.remove_widget(self.resegmentation_box)
            self.new_preprocessing_parameters = {'min_threshold': self.chosen_video.video.min_threshold,
                                        'max_threshold': self.chosen_video.video.max_threshold,
                                        'min_area': self.chosen_video.video.min_area,
                                        'max_area': self.chosen_video.video.max_area}
            self.visualise_resegmentation(self.frames_with_more_blobs_than_animals[0])
        elif len(self.frames_with_more_blobs_than_animals) == 0:
            Clock.unschedule(self.resegmentation)
            self.consistency_popup.dismiss()
            self.consistency_success_popup.open()

    def check_segmentation_consistency(self, *args):
        super().check_segmentation_consistency()

        if self.check_segmentation_consistency_switch.active and \
          len(self.frames_with_more_blobs_than_animals) > 0:
            self.resegmentation_step_finished = True
            self.consistency_popup.dismiss()
            self.consistency_fail_popup.open()
            Clock.schedule_interval(self.resegmentation, 1)
        else:
            self.consistency_popup.dismiss()
            self.consistency_success_popup.open()


    def save_list_of_blobs_segmented(self, *args):
        super().save_list_of_blobs_segmented()
        self.consistency_success_popup.dismiss()

    def model_area_and_crossing_detector(self, *args):
        super().model_area_and_crossing_detector()
        self.DCD_popup.open()


    def train_and_apply_crossing_detector(self, *args):
        super().train_and_apply_crossing_detector()
        self.DCD_popup.dismiss()

    def plot_crossing_detection_statistics(self, *args):
        content = BoxLayout(orientation = "vertical")
        self.crossing_label = CustomLabel(font_size = 14, text = "The deep crossing detector has been trained succesfully "+
                                        "and used to discriminate crossing and individual images. "+
                                        "In the figure the loss, accuracy and accuracy per class, respectively. "+
                                        "The video is currenly being fragmented. The 'Go to the tracking tab' button will"+
                                        " activate at the end of the process.", size_hint = (1., .2))
        if self.chosen_video.video.number_of_animals != 1 and CHOSEN_VIDEO.video._there_are_crossings and not self.crossing_detector_trainer.model_diverged:
            matplotlib.rcParams.update({'font.size': 8,
                                        'axes.labelsize': 8,
                                        'xtick.labelsize' : 8,
                                        'ytick.labelsize' : 8,
                                        'legend.fontsize': 8})
            fig, ax_arr = plt.subplots(3)
            fig.set_facecolor((.188, .188, .188))
            fig.subplots_adjust(left=0.1, bottom=0.15, right=.9, top=.95, wspace=None, hspace=1)
            self.fig.set_facecolor((.188, .188, .188))
            [(ax.set_facecolor((.188, .188, .188)), ax.tick_params(color='white', labelcolor='white'), ax.xaxis.label.set_color('white'), ax.yaxis.label.set_color('white')) for ax in ax_arr]
            [spine.set_edgecolor('white') for ax in ax_arr for spine in ax.spines.values()]
            self.crossing_detector_trainer.store_training_accuracy_and_loss_data.plot(ax_arr, color = 'r', plot_now = False, legend_font_color = "white")
            self.crossing_detector_trainer.store_validation_accuracy_and_loss_data.plot(ax_arr, color ='b', plot_now = False, legend_font_color = "white")
            self.go_to_tracking_button = Button(text = "Go to the tracking tab", size_hint = (1.,.1))
            self.go_to_tracking_button.disabled = True
            crossing_detector_accuracy = FigureCanvasKivyAgg(fig)
            content.add_widget(self.crossing_label)
            content.add_widget(self.go_to_tracking_button)
            content.add_widget(crossing_detector_accuracy)
        elif self.chosen_video.video.number_of_animals == 1:
            content.add_widget(CustomLabel(text = "Tracking a single animal...", size_hint = (1.,.1)))
            self.go_to_tracking_button = Button(text = "Go to the tracking tab", size_hint = (1.,.1))
            self.go_to_tracking_button.disabled = True
            content.add_widget(self.go_to_tracking_button)
        elif not CHOSEN_VIDEO.video._there_are_crossings:
            content.add_widget(CustomLabel(text = "There were not enough crossings to train the crossing detector. \nCrossings and individuals will be discriminated only by area."))
            self.go_to_tracking_button = Button(text = "Go to the tracking tab", size_hint = (1.,.1))
            self.go_to_tracking_button.disabled = True
            # self.disappointed = Image(source = os.path.join(os.path.dirname(__file__), 'single_animal.png'))
            # content.add_widget(self.disappointed)
            content.add_widget(self.go_to_tracking_button)
        self.crossing_detector_accuracy_popup = Popup(title = 'Crossing/individual images discrimination',
                            content = content,
                            size_hint = (1., 1.))
        self.crossing_detector_accuracy_popup.bind(on_open = self.generate_list_of_fragments_and_global_fragments)
        if hasattr(self, 'go_to_tracking_button'):
            self.go_to_tracking_button.bind(on_release = self.crossing_detector_accuracy_popup.dismiss)
        self.chosen_video.video._crossing_detector_time =\
            time.time() - self.chosen_video.video.crossing_detector_time
        self.crossing_detector_accuracy_popup.open()

    def generate_list_of_fragments_and_global_fragments(self, *args):
        super().generate_list_of_fragments_and_global_fragments()
        self.deactivate_tracking.setter(False)

    def open_resolution_reduction_popup(self, *args):
        self.res_red_popup.open()

    def on_enter_res_red_coeff(self, *args):
        self.chosen_video.video.resolution_reduction = float(self.res_red_input.text)
        self.resolution_reduction = self.chosen_video.video.resolution_reduction
        self.res_red_popup.dismiss()
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def apply_ROI(self, instance, active):
        self.chosen_video.video._apply_ROI = active
        if active  == True:
            num_valid_pxs_in_ROI = len(sum(np.where(self.chosen_video.video.ROI == 255)))
            num_pxs_in_frame = self.chosen_video.video.height * self.chosen_video.video.width
            self.ROI_is_trivial = (num_pxs_in_frame == num_valid_pxs_in_ROI or num_valid_pxs_in_ROI == 0)
            if self.chosen_video.video.ROI is not None and not self.ROI_is_trivial:
                self.ROI = self.chosen_video.video.ROI
            elif self.ROI_is_trivial:
                self.ROI_popup.open()
                instance.active = False
                self.chosen_video.apply_ROI = False
        elif active == False:
            self.ROI = np.ones((self.chosen_video.video.height, self.chosen_video.video.width) ,dtype='uint8') * 255
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def update_bkg_and_ROI_in_CHOSEN_VIDEO(self):
        self.chosen_video.video.resolution_reduction = self.chosen_video.video.resolution_reduction

    def apply_bkg_subtraction(self, instance, active):
        self.areas_plotted = False
        self.chosen_video.video._subtract_bkg = active
        if self.chosen_video.video.subtract_bkg == True:
            if self.chosen_video.video.bkg is not None:
                self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)
            elif self.chosen_video.old_video is not None and self.chosen_video.old_video.original_bkg is not None:
                self.chosen_video.video._original_bkg = self.chosen_video.old_video.original_bkg
                self.update_bkg_and_ROI_in_self.chosen_video()
                self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)
            elif self.chosen_video.video.original_bkg is None:

                self.bkg_subtractor.computing_popup.open()

        else:
            self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def add_widget_list(self):
        for w in self.w_list:
            self.container_layout.add_widget(w)

    def update_max_th_lbl(self,instance, value):
        self.max_threshold_lbl.text = "Max intensity:\n" +  str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def update_min_th_lbl(self,instance, value):
        self.min_threshold_lbl.text = "Min intensity:\n" + str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def update_max_area_lbl(self,instance, value):
        self.max_area_lbl.text = "Max area:\n" + str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def update_min_area_lbl(self,instance, value):
        self.min_area_lbl.text = "Min area:\n" + str(int(value))
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def init_segment_zero(self):
        super().init_segment_zero()

        self.visualiser = VisualiseVideo(chosen_video = self.chosen_video)
        self.add_widget(self.container_layout)
        self.add_widget(self.visualiser)
        self.create_areas_figure()
        self.visualiser.visualise_video(self.chosen_video.video, func = self.show_preprocessing)
        self.ROI_switch.bind(active = self.apply_ROI)

    @staticmethod
    def set_matplotlib_params(font_size = 8):
        matplotlib.rcParams.update({'font.size': font_size, 'axes.labelsize': font_size,
                                    'xtick.labelsize' : font_size, 'ytick.labelsize' : font_size,
                                    'legend.fontsize': font_size})

    def create_areas_figure(self, visualiser = None):
        self.set_matplotlib_params()
        self.areas_box = BoxLayout(orientation = "vertical", size_hint = (1.,.3))
        self.areas_label_text = "Areas:"
        self.areas_label = CustomLabel(font_size = 14, text = self.areas_label_text, size_hint = (1.,.15))
        self.fig, self.ax = plt.subplots(1)
        self.fig.subplots_adjust(left=.1, bottom=.3, right=.9, top=.9)
        self.fig.set_facecolor((.188, .188, .188))
        self.ax.set_xlabel('blob')
        self.ax.set_ylabel('area')
        self.ax.set_facecolor((.188, .188, .188))
        self.ax.tick_params(color='white', labelcolor='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        [spine.set_edgecolor('white') for spine in self.ax.spines.values()]
        self.area_bars = FigureCanvasKivyAgg(self.fig)
        self.area_bars_width = .5
        self.areas_box.add_widget(self.areas_label)
        self.areas_box.add_widget(self.area_bars)
        if visualiser is None:
            visualiser = self.visualiser

    def plot_areas_figure(self, areas, visualiser):
        self.ax.clear()
        self.ax.bar(range(len(areas)), areas, self.area_bars_width)
        if len(areas) > 0:
            min_area = np.min(areas)
            self.ax.axhline(min_area, color = 'w', linewidth = .3)
            self.areas_label.text = str(len(areas)) + " blobs detected. Minimum area: " + str(min_area)

        self.area_bars.draw()
        has_child = False

        for c in list(visualiser.children):
            if isinstance(c, FigureCanvasKivyAgg):
                has_child = True

        if not has_child:
            visualiser.add_widget(self.areas_box)

    def show_preprocessing(self, frame, visualiser = None, sliders = None, hide_video_slider = False):
        if visualiser is None:
            visualiser = self.visualiser

        min_threshold_slider = self.min_threshold_slider if sliders is None else sliders[0]
        max_threshold_slider = self.max_threshold_slider if sliders is None else sliders[1]
        min_area_slider = self.min_area_slider if sliders is None else sliders[2]
        max_area_slider = self.max_area_slider if sliders is None else sliders[3]
        if hide_video_slider:
            visualiser.video_slider.disabled = True
        if len(frame.shape) > 2:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY )
        else:
            self.frame = frame
        if hasattr(self, 'area_bars'):
            visualiser.remove_widget(self.areas_box)
        if not hasattr(self.chosen_video.video, 'number_of_channels'):
            #if frame.shape[2] == 1 or (np.any(frame[:,:,1] == frame[:,:,2] ) and np.any(frame[:,:, 0] == frame[:,:,1])):
            self.chosen_video.video._number_of_channels = 1
            #else:
            #    raise NotImplementedError("Colour videos has still to be integrated")
        if hasattr(self.chosen_video.video, 'resolution_reduction') and self.chosen_video.video.resolution_reduction != 1:
            if self.bkg_subtractor_switch.active and self.chosen_video.video.bkg.shape != self.frame.shape:
                bkg = cv2.resize(self.chosen_video.video.bkg, None, fx = self.chosen_video.video.resolution_reduction, fy = self.chosen_video.video.resolution_reduction,
                                 interpolation=cv2.INTER_AREA)
            else:
                bkg = self.chosen_video.video.bkg
            if self.ROI.shape != self.frame.shape:
                ROI = cv2.resize(self.ROI, None, fx = self.chosen_video.video.resolution_reduction, fy = self.chosen_video.video.resolution_reduction,
                                 interpolation=cv2.INTER_AREA)
            else:
                ROI = self.ROI
        else:
            bkg = self.chosen_video.video.bkg
            ROI = self.ROI
        avIntensity = np.float32(np.mean(self.frame))
        self.av_frame = self.frame / avIntensity
        self.segmented_frame = segment_frame(self.av_frame,
                                            int(min_threshold_slider.value),
                                            int(max_threshold_slider.value),
                                            bkg,
                                            ROI,
                                            self.bkg_subtractor_switch.active)
        boundingBoxes, miniFrames, _, areas, _, goodContours, _ = blob_extractor(self.segmented_frame,
                                                                        self.frame,
                                                                        int(min_area_slider.value),
                                                                        int(max_area_slider.value))
        if hasattr(self, "number_of_detected_blobs"):
            self.number_of_detected_blobs.append(len(areas))


        self.plot_areas_figure(areas, visualiser)
        cv2.drawContours(self.frame, goodContours, -1, color=255, thickness = -1)
        # self.frame = cv2.bitwise_and(self.frame, self.frame, mask = self.ROI)
        alpha = .05
        self.frame = cv2.addWeighted(ROI, alpha, self.frame, 1 - alpha, 0)
        if self.count_scrollup != 0:
            self.dst = cv2.warpAffine(self.frame, self.M, (self.frame.shape[1], self.frame.shape[1]))
            buf1 = cv2.flip(self.dst,0)
        else:
            buf1 = cv2.flip(self.frame, 0)
        buf = buf1.tostring()
        textureFrame = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='luminance')
        textureFrame.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
        visualiser.display_layout.texture = textureFrame

    def fromShowFrameToTexture(self, coords):
        """Maps coordinate in visualiser.display_layout (the image whose texture is the frame) to
        the coordinates of the original image
        """
        coords = np.asarray(coords)
        origFrameW = self.chosen_video.video.width
        origFrameH = self.chosen_video.video.height
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

    def create_computational_steps_popups(self):
        self.segmenting_label = CustomLabel(text='Blobs are being extracted from each frame. This operation can take several minutes.')
        self.segmenting_popup_content = BoxLayout(orientation = "vertical")
        self.segmenting_popup_content.add_widget(self.segmenting_label)
        self.segmenting_popup = Popup(title='Segmentation',
            content = self.segmenting_popup_content,
            size_hint = (.3,.3))

        self.consistency_label = CustomLabel(text='Checking the consistency of the segmentation: A number of blobs in single frame higher than the number of animals to track is not admitted.')
        self.consistency_popup_content = BoxLayout(orientation = "vertical")
        self.consistency_popup_content.add_widget(self.consistency_label)
        self.consistency_popup = Popup(title='Consistency',
            content = self.consistency_popup_content,
            size_hint = (.3,.3))

        self.consistency_success_label = CustomLabel(text='The segmentation is consistent: Saving the list of blobs.')
        self.consistency_success_popup_content = BoxLayout(orientation = "vertical")
        self.consistency_success_popup_content.add_widget(self.consistency_success_label)
        self.consistency_success_popup = Popup(title='Success',
            content = self.consistency_success_popup_content,
            size_hint = (.3,.3))

        self.consistency_fail_label = CustomLabel(text='Some frame contain more blobs than animals. Please specify the parameters to be used in those frames',
                                                    size_hint = (1.,.1))
        self.consistency_fail_popup_content = BoxLayout(orientation = "vertical")
        self.consistency_fail_popup_content.add_widget(self.consistency_fail_label)
        self.consistency_fail_popup = Popup(title='Resegment',
            content = self.consistency_fail_popup_content,
            size_hint = (.9,.9))

        self.DCD_label = CustomLabel(text='Discriminating individual and crossing images: Applying model area and deep crossing detector.')
        self.DCD_popup_content = BoxLayout(orientation = "vertical")
        self.DCD_popup_content.add_widget(self.DCD_label)
        self.DCD_popup = Popup(title='Crossing detection',
            content = self.DCD_popup_content,
            size_hint = (.3,.3))

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

    def create_number_of_animals_popup(self):
        self.num_of_animals_container = BoxLayout()
        self.num_of_animals = BoxLayout(orientation="vertical")
        self.num_of_animals_label = CustomLabel(text='Type the number of animals to be tracked.')
        self.num_of_animals.add_widget(self.num_of_animals_label)
        self.num_of_animals_input = TextInput(text = '', multiline = False)
        self.num_of_animals.add_widget(self.num_of_animals_input)
        self.num_of_animals_container.add_widget(self.num_of_animals)
        self.num_of_animals_popup = Popup(title = 'Number of animals',
                            content = self.num_of_animals_container,
                            size_hint = (.3, .3))

    def set_number_of_animals(self, *args):
        self.chosen_video.video._number_of_animals = int(self.num_of_animals_input.text)
        self.num_of_animals_popup.dismiss()

    def show_segmenting_popup(self, *args):
        self.segmenting_popup.open()

    def show_consistency_popup(self, *args):
        self.consistency_popup.open()
