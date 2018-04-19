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
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function
import kivy
from kivy.core.window import Window
from kivy.clock import Clock
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
from kivy.uix.progressbar import ProgressBar
from idtrackerai.gui.visualise_video import VisualiseVideo
from idtrackerai.gui.bkg_subtraction import BkgSubtraction
from idtrackerai.gui.kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process
from functools import partial
import matplotlib
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from scipy.stats import mode
import cv2
from idtrackerai.video import Video
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import ListOfFragments, create_list_of_fragments
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments, create_list_of_global_fragments
from idtrackerai.preprocessing.segmentation import segment_frame, segment, resegment
from idtrackerai.utils.video_utils import blob_extractor
from idtrackerai.crossing_detector import detect_crossings

class PreprocessingPreview(BoxLayout):
    def __init__(self, chosen_video = None,
                deactivate_preprocessing = None,
                deactivate_tracking = None,
                **kwargs):
        super(PreprocessingPreview, self).__init__(**kwargs)
        global CHOSEN_VIDEO, DEACTIVATE_PREPROCESSING, DEACTIVATE_TRACKING
        CHOSEN_VIDEO = chosen_video
        DEACTIVATE_PREPROCESSING = deactivate_preprocessing
        DEACTIVATE_TRACKING = deactivate_tracking
        CHOSEN_VIDEO.bind(chosen=self.do)
        self.container_layout = BoxLayout(orientation = 'vertical', size_hint = (.3, 1.))
        self.reduce_resolution_btn = Button(text = "Reduce resolution")
        self.bkg_subtractor = BkgSubtraction(orientation = 'vertical', chosen_video = CHOSEN_VIDEO)
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

    def init_preproc_parameters(self):
        if CHOSEN_VIDEO.old_video is not None and CHOSEN_VIDEO.old_video._has_been_preprocessed == True:
            self.max_threshold = CHOSEN_VIDEO.old_video.max_threshold
            self.min_threshold = CHOSEN_VIDEO.old_video.min_threshold
            self.min_area = CHOSEN_VIDEO.old_video.min_area
            self.max_area = CHOSEN_VIDEO.old_video.max_area
            self.resolution_reduction = CHOSEN_VIDEO.old_video.resolution_reduction
            self.number_of_animals = CHOSEN_VIDEO.old_video.number_of_animals
            CHOSEN_VIDEO.video.resolution_reduction = CHOSEN_VIDEO.old_video.resolution_reduction
        else:
            self.max_threshold = 135
            self.min_threshold = 0
            self.min_area = 150
            self.max_area = 60000
            CHOSEN_VIDEO.video.resolution_reduction = 1.
            self.resolution_reduction = CHOSEN_VIDEO.video.resolution_reduction
            if CHOSEN_VIDEO.video._original_ROI is None:
                CHOSEN_VIDEO.video._original_ROI = np.ones( (CHOSEN_VIDEO.video.height, CHOSEN_VIDEO.video.width), dtype='uint8') * 255
        self.max_threshold_slider = Slider(id = 'max_threhsold', min = 0, max = 255, value = self.max_threshold, step = 1)
        self.max_threshold_lbl = CustomLabel(font_size = 14, id = 'max_threshold_lbl', text = "Max intensity: " + str(int(self.max_threshold_slider.value)))
        self.min_threshold_slider = Slider(id='min_threshold_slider', min = 0, max = 255, value = self.min_threshold, step = 1)
        self.min_threshold_lbl = CustomLabel(font_size = 14, id='min_threshold_lbl', text = "Min intensity:" + str(int(self.min_threshold_slider.value)))
        self.max_area_slider = Slider(id='max_area_slider', min = 0, max = 60000, value = self.max_area, step = 1)
        self.max_area_lbl = CustomLabel(font_size = 14, id='max_area_lbl', text = "Max area:" + str(int(self.max_area_slider.value)))
        self.min_area_slider = Slider(id='min_area_slider', min = 0, max = 1000, value = self.min_area, step = 1)
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
        self.segment_video_btn = Button(text = "Segment video")
        self.container_layout.add_widget(self.segment_video_btn)
        self.container_layout.add_widget(self.help_button_preprocessing)

    def activate_ROI_switch(self, *args):
        if hasattr(self,'visualiser'):
            self.ROI_switch.active = True

    def deactivate_ROI_switch(self, *args):
        if hasattr(self,'visualiser'):
            self.ROI_switch.active = False

    def do(self, *args):
        if CHOSEN_VIDEO.video is not None and CHOSEN_VIDEO.video.video_path is not None:
            self.init_preproc_parameters()
            self.create_resolution_reduction_popup()
            self.res_red_input.bind(on_text_validate = self.on_enter_res_red_coeff)
            self.reduce_resolution_btn.bind(on_press = self.open_resolution_reduction_popup)
            # self.ROI_switch.bind(active = self.apply_ROI)
            num_valid_pxs_in_ROI = len(sum(np.where(CHOSEN_VIDEO.video.ROI == 255)))
            num_pxs_in_frame = CHOSEN_VIDEO.video.height * CHOSEN_VIDEO.video.width
            self.ROI_switch.active = not (num_pxs_in_frame == num_valid_pxs_in_ROI or num_valid_pxs_in_ROI == 0)
            self.bkg_subtractor_switch.active = False
            self.bkg_subtractor_switch.bind(active = self.apply_bkg_subtraction)
            CHOSEN_VIDEO.video.resolution_reduction = CHOSEN_VIDEO.video.resolution_reduction
            self.bkg = CHOSEN_VIDEO.video.bkg
            self.ROI = CHOSEN_VIDEO.video.ROI if CHOSEN_VIDEO.video.ROI is not None else np.ones((CHOSEN_VIDEO.video.height, CHOSEN_VIDEO.video.width) ,dtype='uint8') * 255
            self.create_number_of_animals_popup()
            self.num_of_animals_input.bind(on_text_validate = self.set_number_of_animals)
            self.num_of_animals_popup.bind(on_dismiss = self.show_segmenting_popup)
            self.create_computational_steps_popups()
            self.segmenting_popup.bind(on_open = self.compute_list_of_blobs)
            self.segmenting_popup.bind(on_dismiss = self.show_consistency_popup)
            self.consistency_popup.bind(on_open = self.check_segmentation_consistency)
            self.consistency_success_popup.bind(on_open = self.save_list_of_blobs)
            self.consistency_success_popup.bind(on_dismiss = self.model_area_and_crossing_detector)
            self.DCD_popup.bind(on_open = self.train_and_apply_crossing_detector)
            self.DCD_popup.bind(on_dismiss = self.plot_crossing_detection_statistics)
            self.init_segment_zero()
            self.has_been_executed = True
            self.segment_video_btn.bind(on_press = self.segment)
            self.bkg_subtractor.visualiser = self.visualiser
            self.bkg_subtractor.shower = self.show_preprocessing

    def compute_list_of_blobs(self, *args):
        self.blobs = segment(CHOSEN_VIDEO.video)
        self.segmenting_popup.dismiss()

    def segment(self, *args):
        CHOSEN_VIDEO.video._max_threshold = self.max_threshold_slider.value
        CHOSEN_VIDEO.video._min_threshold = self.min_threshold_slider.value
        CHOSEN_VIDEO.video._min_area = self.min_area_slider.value
        CHOSEN_VIDEO.video._max_area = self.max_area_slider.value
        CHOSEN_VIDEO.video.resolution_reduction = self.resolution_reduction
        CHOSEN_VIDEO.video.save()
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
        max_threshold_slider = Slider(id = 'max_threshold_slider', min = 0, max = 255,
                        value = self.new_preprocessing_parameters['max_threshold'],
                        step = 1)
        max_threshold_lbl = CustomLabel(font_size = 14,
                text = "Max intensity: " + str(int(max_threshold_slider.value)))
        min_threshold_slider = Slider(id = 'min_threshold_slider', min = 0, max = 255,
                        value = self.new_preprocessing_parameters['min_threshold'],
                        step = 1)
        min_threshold_lbl = CustomLabel(font_size = 14,
                text = "Min intensity:" + str(int(min_threshold_slider.value)))
        max_area_slider = Slider(id = 'max_area_slider', min = 0, max = 60000,
                            value = self.new_preprocessing_parameters['max_area'],
                            step = 1)
        max_area_lbl = CustomLabel(font_size = 14,
                        text = "Max area:" + str(int(max_area_slider.value)))
        min_area_slider = Slider(id = 'min_area_slider', min = 0, max = 1000,
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
        resegmentation_visualiser = VisualiseVideo(chosen_video = CHOSEN_VIDEO)
        self.create_areas_figure(visualiser = resegmentation_visualiser)
        resegmentation_visualiser.visualise_video(CHOSEN_VIDEO.video,
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
            CHOSEN_VIDEO.video._maximum_number_of_blobs = resegment(CHOSEN_VIDEO.video,
                                                frame_number,
                                                CHOSEN_VIDEO.list_of_blobs,
                                                self.new_preprocessing_parameters)
            if CHOSEN_VIDEO.video._maximum_number_of_blobs <= CHOSEN_VIDEO.video.number_of_animals:
                CHOSEN_VIDEO.video._resegmentation_parameters.append((frame_number, self.new_preprocessing_parameters))
        self.frames_with_more_blobs_than_animals, CHOSEN_VIDEO.video._maximum_number_of_blobs = CHOSEN_VIDEO.list_of_blobs.check_maximal_number_of_blob(CHOSEN_VIDEO.video.number_of_animals, return_maximum_number_of_blobs = True)
        self.resegmentation_step_finished = True

    def resegmentation(self, *args):
        if len(self.frames_with_more_blobs_than_animals) > 0 and self.resegmentation_step_finished == True:
            if hasattr(self, 'resegmentation_box'):
                self.consistency_fail_popup_content.remove_widget(self.resegmentation_box)
            self.new_preprocessing_parameters = {'min_threshold': CHOSEN_VIDEO.video.min_threshold,
                                        'max_threshold': CHOSEN_VIDEO.video.max_threshold,
                                        'min_area': CHOSEN_VIDEO.video.min_area,
                                        'max_area': CHOSEN_VIDEO.video.max_area}
            self.visualise_resegmentation(self.frames_with_more_blobs_than_animals[0])
        elif len(self.frames_with_more_blobs_than_animals) == 0:
            Clock.unschedule(self.resegmentation)
            self.consistency_popup.dismiss()
            self.consistency_success_popup.open()

    def check_segmentation_consistency(self, *args):
        CHOSEN_VIDEO.list_of_blobs = ListOfBlobs(blobs_in_video = self.blobs)
        CHOSEN_VIDEO.video.create_preprocessing_folder()
        self.frames_with_more_blobs_than_animals, CHOSEN_VIDEO.video._maximum_number_of_blobs = CHOSEN_VIDEO.list_of_blobs.check_maximal_number_of_blob(CHOSEN_VIDEO.video.number_of_animals, return_maximum_number_of_blobs = True)
        if len(self.frames_with_more_blobs_than_animals) > 0 and (self.check_segmentation_consistency_switch.active or CHOSEN_VIDEO.video.number_of_animals == 1):
            self.resegmentation_step_finished = True
            self.consistency_popup.dismiss()
            self.consistency_fail_popup.open()
            Clock.schedule_interval(self.resegmentation, 1)
        else:
            self.consistency_popup.dismiss()
            self.consistency_success_popup.open()

    def save_list_of_blobs(self, *args):
        CHOSEN_VIDEO.video._has_been_segmented = True
        if len(CHOSEN_VIDEO.list_of_blobs.blobs_in_video[-1]) == 0:
            CHOSEN_VIDEO.list_of_blobs.blobs_in_video = CHOSEN_VIDEO.list_of_blobs.blobs_in_video[:-1]
            CHOSEN_VIDEO.list_of_blobs.number_of_frames = len(CHOSEN_VIDEO.list_of_blobs.blobs_in_video)
            CHOSEN_VIDEO.video._number_of_frames = CHOSEN_VIDEO.list_of_blobs.number_of_frames
        CHOSEN_VIDEO.video.save()
        CHOSEN_VIDEO.list_of_blobs.save(CHOSEN_VIDEO.video,
                                CHOSEN_VIDEO.video.blobs_path_segmented,
                                number_of_chunks = CHOSEN_VIDEO.video.number_of_frames)
        self.consistency_success_popup.dismiss()

    def model_area_and_crossing_detector(self, *args):
        CHOSEN_VIDEO.video._model_area, CHOSEN_VIDEO.video._median_body_length = CHOSEN_VIDEO.list_of_blobs.compute_model_area_and_body_length(CHOSEN_VIDEO.video.number_of_animals)
        CHOSEN_VIDEO.video.compute_identification_image_size(CHOSEN_VIDEO.video.median_body_length)
        if not CHOSEN_VIDEO.list_of_blobs.blobs_are_connected:
            CHOSEN_VIDEO.list_of_blobs.compute_overlapping_between_subsequent_frames()
        self.DCD_popup.open()

    def train_and_apply_crossing_detector(self, *args):
        self.crossing_detector_trainer = detect_crossings(CHOSEN_VIDEO.list_of_blobs, CHOSEN_VIDEO.video,
                        CHOSEN_VIDEO.video.model_area, use_network = True,
                        return_store_objects = True, plot_flag = False)
        self.DCD_popup.dismiss()

    def plot_crossing_detection_statistics(self, *args):
        content = BoxLayout(orientation = "vertical")
        self.crossing_label = CustomLabel(font_size = 14, text = "The deep crossing detector has been trained succesfully "+
                                        "and used to discriminate crossing and individual images. "+
                                        "In the figure the loss, accuracy and accuracy per class, respectively. "+
                                        "The video is currenly being fragmented. The 'Go to the tracking tab' button will"+
                                        " activate at the end of the process.", size_hint = (1., .2))
        if CHOSEN_VIDEO.video.number_of_animals != 1 and not self.crossing_detector_trainer.model_diverged:
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
        elif CHOSEN_VIDEO.video.number_of_animals == 1:
            content.add_widget(CustomLabel(text = "Ok, tracking a single animal...", size_hint = (1.,.1)))
            self.go_to_tracking_button = Button(text = "Go to the tracking tab", size_hint = (1.,.1))
            self.go_to_tracking_button.disabled = True
            self.disappointed = Image(source = os.path.join(os.path.dirname(__file__), 'single_animal.png'))
            content.add_widget(self.disappointed)
            content.add_widget(self.go_to_tracking_button)
        else:
            content.add_widget(CustomLabel(text = "The model diverged, crossing and individuals will be discriminated only by area."))
        self.crossing_detector_accuracy_popup = Popup(title = 'Crossing/individual images discrimination',
                            content = content,
                            size_hint = (1., 1.))
        self.crossing_detector_accuracy_popup.bind(on_open = self.generate_list_of_fragments_and_global_fragments)
        if hasattr(self, 'go_to_tracking_button'):
            self.go_to_tracking_button.bind(on_release = self.crossing_detector_accuracy_popup.dismiss)
        self.crossing_detector_accuracy_popup.open()

    def generate_list_of_fragments_and_global_fragments(self, *args):
        CHOSEN_VIDEO.list_of_blobs.compute_overlapping_between_subsequent_frames()
        CHOSEN_VIDEO.list_of_blobs.compute_fragment_identifier_and_blob_index(max(CHOSEN_VIDEO.video.number_of_animals, CHOSEN_VIDEO.video.maximum_number_of_blobs))
        CHOSEN_VIDEO.list_of_blobs.compute_crossing_fragment_identifier()
        fragments = create_list_of_fragments(CHOSEN_VIDEO.list_of_blobs.blobs_in_video,
                                            CHOSEN_VIDEO.video.number_of_animals)
        self.list_of_fragments = ListOfFragments(fragments)
        CHOSEN_VIDEO.video._fragment_identifier_to_index = self.list_of_fragments.get_fragment_identifier_to_index_list()
        if CHOSEN_VIDEO.video.number_of_animals != 1:
            global_fragments = create_list_of_global_fragments(CHOSEN_VIDEO.list_of_blobs.blobs_in_video,
                                                                self.list_of_fragments.fragments,
                                                                CHOSEN_VIDEO.video.number_of_animals)
            self.list_of_global_fragments = ListOfGlobalFragments(global_fragments)
            CHOSEN_VIDEO.video.number_of_global_fragments = self.list_of_global_fragments.number_of_global_fragments
            self.list_of_global_fragments.filter_candidates_global_fragments_for_accumulation()
            CHOSEN_VIDEO.video.number_of_global_fragments_candidates_for_accumulation = self.list_of_global_fragments.number_of_global_fragments
            #XXX I skip the fit of the gamma ...
            self.list_of_global_fragments.relink_fragments_to_global_fragments(self.list_of_fragments.fragments)
            CHOSEN_VIDEO.video._number_of_unique_images_in_global_fragments = self.list_of_fragments.compute_total_number_of_images_in_global_fragments()
            self.list_of_global_fragments.compute_maximum_number_of_images()
            CHOSEN_VIDEO.video._maximum_number_of_images_in_global_fragments = self.list_of_global_fragments.maximum_number_of_images
            self.list_of_fragments.get_accumulable_individual_fragments_identifiers(self.list_of_global_fragments)
            self.list_of_fragments.get_not_accumulable_individual_fragments_identifiers(self.list_of_global_fragments)
            self.list_of_fragments.set_fragments_as_accumulable_or_not_accumulable()
        else:
            CHOSEN_VIDEO.video._number_of_unique_images_in_global_fragments = None
            CHOSEN_VIDEO.video._maximum_number_of_images_in_global_fragments = None
        CHOSEN_VIDEO.video._has_been_preprocessed = True
        CHOSEN_VIDEO.list_of_blobs.save(CHOSEN_VIDEO.video, CHOSEN_VIDEO.video.blobs_path, number_of_chunks = CHOSEN_VIDEO.video.number_of_frames)
        self.list_of_fragments.save(CHOSEN_VIDEO.video.fragments_path)
        if CHOSEN_VIDEO.video.number_of_animals != 1:
            self.list_of_global_fragments.save(CHOSEN_VIDEO.video.global_fragments_path, self.list_of_fragments.fragments)
            CHOSEN_VIDEO.list_of_global_fragments = self.list_of_global_fragments
        CHOSEN_VIDEO.video.save()
        CHOSEN_VIDEO.list_of_fragments = self.list_of_fragments
        DEACTIVATE_TRACKING.setter(False)

    def open_resolution_reduction_popup(self, *args):
        self.res_red_popup.open()

    def on_enter_res_red_coeff(self, *args):
        CHOSEN_VIDEO.video.resolution_reduction = float(self.res_red_input.text)
        self.resolution_reduction = CHOSEN_VIDEO.video.resolution_reduction
        self.res_red_popup.dismiss()
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def apply_ROI(self, instance, active):
        CHOSEN_VIDEO.video._apply_ROI = active
        if active  == True:
            num_valid_pxs_in_ROI = len(sum(np.where(CHOSEN_VIDEO.video.ROI == 255)))
            num_pxs_in_frame = CHOSEN_VIDEO.video.height * CHOSEN_VIDEO.video.width
            self.ROI_is_trivial = (num_pxs_in_frame == num_valid_pxs_in_ROI or num_valid_pxs_in_ROI == 0)
            if CHOSEN_VIDEO.video.ROI is not None and not self.ROI_is_trivial:
                self.ROI = CHOSEN_VIDEO.video.ROI
            elif self.ROI_is_trivial:
                self.ROI_popup.open()
                instance.active = False
                CHOSEN_VIDEO.apply_ROI = False
        elif active == False:
            self.ROI = np.ones((CHOSEN_VIDEO.video.height, CHOSEN_VIDEO.video.width) ,dtype='uint8') * 255
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)

    def update_bkg_and_ROI_in_CHOSEN_VIDEO(self):
        CHOSEN_VIDEO.video.resolution_reduction = CHOSEN_VIDEO.video.resolution_reduction

    def apply_bkg_subtraction(self, instance, active):
        self.areas_plotted = False
        CHOSEN_VIDEO.video._subtract_bkg = active
        if CHOSEN_VIDEO.video.subtract_bkg == True:
            if CHOSEN_VIDEO.video.bkg is not None:
                self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)
            elif CHOSEN_VIDEO.old_video is not None and CHOSEN_VIDEO.old_video.original_bkg is not None:
                CHOSEN_VIDEO.video._original_bkg = CHOSEN_VIDEO.old_video.original_bkg
                self.update_bkg_and_ROI_in_CHOSEN_VIDEO()
                self.visualiser.visualise(self.visualiser.video_slider.value, func = self.show_preprocessing)
            elif CHOSEN_VIDEO.video.original_bkg is None:

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
        self.visualiser = VisualiseVideo(chosen_video = CHOSEN_VIDEO)
        self.add_widget(self.container_layout)
        self.add_widget(self.visualiser)
        self.currentSegment = 0
        self.areas_plotted = False
        self.number_of_detected_blobs = [0]
        self.create_areas_figure()
        self.visualiser.visualise_video(CHOSEN_VIDEO.video, func = self.show_preprocessing)
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
        if not hasattr(CHOSEN_VIDEO.video, 'number_of_channels'):
            if frame.shape[2] == 1 or (np.any(frame[:,:,1] == frame[:,:,2] ) and np.any(frame[:,:, 0] == frame[:,:,1])):
                CHOSEN_VIDEO.video._number_of_channels = 1
            else:
                raise NotImplementedError("Colour videos has still to be integrated")
        avIntensity = np.float32(np.mean(self.frame))
        self.av_frame = self.frame / avIntensity
        self.segmented_frame = segment_frame(self.av_frame,
                                            int(min_threshold_slider.value),
                                            int(max_threshold_slider.value),
                                            CHOSEN_VIDEO.video.bkg,
                                            self.ROI,
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
        self.frame = cv2.addWeighted(self.ROI, alpha, self.frame, 1 - alpha, 0)
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
        CHOSEN_VIDEO.video._number_of_animals = int(self.num_of_animals_input.text)
        self.num_of_animals_popup.dismiss()

    def show_segmenting_popup(self, *args):
        self.segmenting_popup.open()

    def show_consistency_popup(self, *args):
        self.consistency_popup.open()
