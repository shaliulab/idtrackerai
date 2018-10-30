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
# (2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (R-F.,F. and B.,M. contributed equally to this work.)


from __future__ import absolute_import, division, print_function
import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
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
import matplotlib
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import copy
import numpy as np
import time
from scipy.stats import mode
import cv2
from idtrackerai.preprocessing.segmentation import segment_frame, segment
from idtrackerai.utils.video_utils import blob_extractor
from idtrackerai.video import Video
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import ListOfFragments, create_list_of_fragments
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments
from idtrackerai.crossing_detector import detect_crossings
from idtrackerai.list_of_global_fragments import create_list_of_global_fragments
from idtrackerai.accumulation_manager import AccumulationManager
from idtrackerai.accumulator import perform_one_accumulation_step
from idtrackerai.network.identification_model.network_params import NetworkParams
from idtrackerai.trainer import train
from idtrackerai.assigner import assigner
from idtrackerai.postprocessing.compute_velocity_model import compute_model_velocity
from idtrackerai.postprocessing.correct_impossible_velocity_jumps import correct_impossible_velocity_jumps
from idtrackerai.postprocessing.assign_them_all import close_trajectories_gaps
from idtrackerai.postprocessing.get_trajectories import produce_output_dict
from idtrackerai.pre_trainer import pre_train_global_fragment
from idtrackerai.network.identification_model.store_accuracy_and_loss import Store_Accuracy_and_Loss
from idtrackerai.network.identification_model.id_CNN import ConvNetwork
from idtrackerai.constants import  BATCH_SIZE_IDCNN, THRESHOLD_ACCEPTABLE_ACCUMULATION, VEL_PERCENTILE, THRESHOLD_EARLY_STOP_ACCUMULATION, MAX_RATIO_OF_PRETRAINED_IMAGES, MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS
from idtrackerai.postprocessing.identify_non_assigned_with_interpolation import assign_zeros_with_interpolation_identities

from .tracker_api import TrackerAPI

class Tracker(TrackerAPI, BoxLayout):

    def __init__(self,
        chosen_video = None,
        deactivate_tracking = None,
        deactivate_validation = None,
        **kwargs
    ):
        TrackerAPI.__init__(self, chosen_video)

        ## GUI ###################################################

        # To remove in the future
        global CHOSEN_VIDEO, DEACTIVATE_TRACKING, DEACTIVATE_VALIDATION
        CHOSEN_VIDEO          = chosen_video
        DEACTIVATE_TRACKING   = deactivate_tracking
        DEACTIVATE_VALIDATION = deactivate_validation

        self.deactivate_tracking   = deactivate_tracking
        self.deactivate_validation = deactivate_validation

        BoxLayout.__init__(self, **kwargs)

        self.has_been_executed = False
        self.control_panel = BoxLayout(orientation = "vertical", size_hint = (.26,1.))
        self.add_widget(self.control_panel)
        self.help_button_tracker = HelpButton()
        self.help_button_tracker.size_hint = (1.,.4)
        self.help_button_tracker.create_help_popup(
            "Tracking",\
            "Press the button 'Start protocol cascade' to start tracking the video. " +
            "In the 'Advanced idCNN controls' you will be able to change the identification model " +
            "hyperparameters. " +
            "Activate the 'Save tensorboard summaries' switch to save the training and validation " +
            "losses as well as a subset of the images at the output of every layer of the model. " +
            "The files with the summaries will be saved in your session folder in the corresponding 'accumulation' folder. " +
            "If you are restoring the session from a different point of the tracking " +
            "press the upper botton which will indicate the process that will be computed."
        )
        ## GUI ###################################################


    def __init_tracking_gui_handler(self, status=None):

        if not self.has_been_executed:
            self.create_main_layout()
            self.control_panel.add_widget(self.help_button_tracker)
            self.has_been_executed = True

        if status==0: # status_post_processing

            self.start_tracking_button.bind(on_release = self.update_and_show_happy_ending_popup)
            self.start_tracking_button.text = "Show estimated\naccuracy"

        elif 1<=status<2: # status_residual_identification

            if status==1.1:
                self.start_tracking_button.bind(on_release = self.update_and_show_happy_ending_popup)
                self.start_tracking_button.text = "Show estimated\naccuracy"

            elif status==1.2:
                self.start_tracking_button.bind(on_release = self.start_from_post_processing)
                self.start_tracking_button.text = "Start\npost-processing"

        elif status==2: # status_protocol3_accumulation

            self.start_tracking_button.bind(on_release = self.start_from_identification)
            self.start_tracking_button.text = "Start\nresidual identification"

        elif status==3: # status_protocol3_pretraining

            self.create_one_shot_accumulation_popup()
            self.start_tracking_button.bind(on_release = self.accumulate)
            self.start_tracking_button.text = "Start\naccumulation\n(protocol 3)"

        elif status==4: # status_protocols1_and_2

            self.create_one_shot_accumulation_popup()
            self.start_tracking_button.bind(on_release = self.accumulate)
            self.start_tracking_button.text = "Start\nidentification\nor\nprotocol 3"

        elif status==5: # status_protocols1_and_2_not_def

            Logger.info("Starting protocol cascade")
            self.start_tracking_button.bind(on_release = self.protocol1)
            self.track_wo_identities_button.bind(on_release = self.track_wo_identities)





    def do(self):
        if self.chosen_video.video.number_of_animals == 1:
            self.create_main_layout()
            self.start_tracking_button.bind(on_release = self.track_single_animal)
            self.start_tracking_button.text = "Get animal\ntrajectory"
            self.start_tracking_button.size_hint = (.2,.3)
        elif self.chosen_video.list_of_global_fragments.number_of_global_fragments == 1:
            self.create_main_layout()
            self.start_tracking_button.bind(on_release = self.track_single_global_fragment_video)
            self.start_tracking_button.text = "Only one global\nfragment\nwas found.\nThere is not\nneed the\nidentification CNN.\nGet animals\ntrajectories"
            self.start_tracking_button.size_hint = (.2,.3)
        else:
            self.init_tracking( gui_handler=self.__init_tracking_gui_handler )


    def track_single_animal(self, *args):
        super().track_single_animal()
        self.trajectories_popup.open()

    def track_single_global_fragment_video(self, *args):
        super().track_single_global_fragment_video()
        self.trajectories_popup.open()

    def track_wo_identities(self, *args):
        self.chosen_video.video.accumulation_trial = 0
        self.chosen_video.video._first_frame_first_global_fragment = [0]
        self.chosen_video.video._track_wo_identities = True
        self.trajectories_popup.open()



    def protocol1(self, *args):
        super().protocol1(create_popup=self.create_one_shot_accumulation_popup)



    def one_shot_accumulation(self, *args):
        super().one_shot_accumulation(save_summaries = self.generate_tensorboard_switch.active)

        self.accumulation_counter_value.text = str(self.accumulation_manager.counter + 1)
        if self.accumulation_manager.counter == 1:
            self.create_tracking_figures_axes()
        self.percentage_accumulated_images_value.text = str(self.accumulation_manager.ratio_accumulated_images)
        self.protocol_value.text = '2' if self.chosen_video.video.accumulation_trial == 0 else '3'
        self.store_training_accuracy_and_loss_data.plot_global_fragments(self.ax_arr,
                                                                    self.chosen_video.video,
                                                                    self.accumulation_manager.list_of_fragments.fragments,
                                                                    black = False,
                                                                    canvas_from_GUI = self.tracking_fig_canvas)
        self.store_validation_accuracy_and_loss_data.plot(self.ax_arr,
                                                    color ='b',
                                                    canvas_from_GUI = self.tracking_fig_canvas,
                                                    index = self.accumulation_manager.counter - 1,
                                                    legend_font_color = 'w')
        self.store_training_accuracy_and_loss_data.plot(self.ax_arr,
                                                    color = 'r',
                                                    canvas_from_GUI = self.tracking_fig_canvas,
                                                    index = self.accumulation_manager.counter - 1,
                                                    legend_font_color = 'w')



    def __accumulate_handler(self, status):

        if status==1:
            self.identification_popup.open()
        elif status==2.1:
            self.one_shot_accumulation_popup.dismiss()
        elif status==2.2:
            self.identification_popup.open()
        elif status==2.3:
            self.create_pretraining_popup()
        elif status==3.1:
            Clock.unschedule(self.accumulate)
        elif status==3.2:
            self.one_shot_accumulation_popup.dismiss()
        elif status==3.3:
            self.identification_popup.open()



    def accumulate(self, *args):
        super().accumulate(gui_handler=self.__accumulate_handler)



    def save_and_update_accumulation_parameters_in_parachute(self):
        Logger.warning("self.accumulation_manager.ratio_accumulated_images %.4f" %self.accumulation_manager.ratio_accumulated_images)
        self.chosen_video.video._ratio_accumulated_images = self.accumulation_manager.ratio_accumulated_images
        self.chosen_video.video._percentage_of_accumulated_images.append(self.chosen_video.video.ratio_accumulated_images)
        self.chosen_video.list_of_fragments.save_light_list(self.chosen_video.video._accumulation_folder)

    def accumulation_loop(self):
        Logger.warning('------------Calling accumulation loop')
        if hasattr(self, 'one_shot_accumulation_popup'):
            delattr(self, 'one_shot_accumulation_popup')
            self.create_one_shot_accumulation_popup()
        super().accumulation_loop()
        self.one_shot_accumulation_popup.open()
        Clock.schedule_interval(self.accumulate, 2)

    def protocol3(self):
        Logger.debug("------------------------> protocol3")
        self.init_pretraining_variables()
        number_of_images_in_global_fragments = self.chosen_video.video.number_of_unique_images_in_global_fragments
        if self.chosen_video.old_video and self.chosen_video.old_video.first_accumulation_finished == True:
            self.chosen_video.list_of_global_fragments.reset(roll_back_to = 'fragmentation')
            self.chosen_video.list_of_fragments.reset(roll_back_to = 'fragmentation')
        Logger.info("Starting pretraining. Checkpoints will be stored in %s" %self.chosen_video.video.pretraining_folder)
        if self.chosen_video.video.tracking_with_knowledge_transfer:
            Logger.info("Performing knowledge transfer from %s" %self.chosen_video.video.knowledge_transfer_model_folder)
            self.pretrain_network_params.knowledge_transfer_folder = self.chosen_video.video.knowledge_transfer_model_folder
        Logger.info("Start pretraining")
        self.pretraining_step_finished = True
        self.pretraining_loop()

    def accumulation_parachute_init(self, iteration_number):
        Logger.debug("------------------------> accumulation_parachute_init")
        Logger.info("Starting accumulation %i" %iteration_number)
        self.one_shot_accumulation_popup.dismiss()
        delete = not self.chosen_video.processes_to_restore['protocol3_accumulation'] if 'protocol3_accumulation' in self.chosen_video.processes_to_restore.keys() else True
        self.chosen_video.video.create_accumulation_folder(iteration_number = iteration_number, delete = delete)
        self.chosen_video.video.accumulation_trial = iteration_number
        self.chosen_video.list_of_fragments.reset(roll_back_to = 'fragmentation')
        self.chosen_video.list_of_global_fragments.reset(roll_back_to = 'fragmentation')
        Logger.info("We will restore the network from a previous pretraining: %s" %self.chosen_video.video.pretraining_folder)
        self.accumulation_network_params.save_folder = self.chosen_video.video.accumulation_folder
        self.accumulation_network_params.restore_folder = self.chosen_video.video.pretraining_folder
        self.accumulation_network_params.scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax']
        Logger.info("Initialising accumulation network")
        self.net = ConvNetwork(self.accumulation_network_params)
        self.net.restore()
        self.net.reinitialize_softmax_and_fully_connected()
        Logger.info("Initialising accumulation manager")
        self.chosen_video.video._first_frame_first_global_fragment.append(self.chosen_video.list_of_global_fragments.set_first_global_fragment_for_accumulation(self.chosen_video.video, accumulation_trial = iteration_number - 1))
        Logger.warning('first_frame_first_global_fragment ' + str(self.chosen_video.video.first_frame_first_global_fragment))
        if self.chosen_video.video.identity_transfer and self.chosen_video.video.number_of_animals < self.chosen_video.video.knowledge_transfer_info_dict['number_of_animals']:
            tf.reset_default_graph()
            self.accumulation_network_params.number_of_animals = self.chosen_video.video.number_of_animals
            self.accumulation_network_params.restore_folder = self.chosen_video.video.pretraining_folder
            self.net = ConvNetwork(self.accumulation_network_params)
            self.net.restore()
            self.net.reinitialize_softmax_and_fully_connected()
        self.chosen_video.list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(self.chosen_video.video, accumulation_trial = iteration_number - 1)
        self.accumulation_manager = AccumulationManager(self.chosen_video.video,
                                                    self.chosen_video.list_of_fragments, self.chosen_video.list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        Logger.info("Start accumulation")
        self.global_step = 0

    def save_after_first_accumulation(self):
        """Set flags and save data"""
        Logger.info("Saving first accumulation paramters")
        Clock.unschedule(self.accumulate)
        if not self.restoring_first_accumulation:
            self.chosen_video.video._first_accumulation_finished = True
            self.chosen_video.video._ratio_accumulated_images = self.accumulation_manager.ratio_accumulated_images
            self.chosen_video.video._percentage_of_accumulated_images = [self.chosen_video.video.ratio_accumulated_images]
            self.chosen_video.video._accumulation_network_params = self.accumulation_network_params
            self.chosen_video.video.save()
            self.chosen_video.list_of_fragments.save(self.chosen_video.video.fragments_path)
            self.chosen_video.list_of_global_fragments.save(self.chosen_video.video.global_fragments_path, self.chosen_video.list_of_fragments.fragments)
            self.chosen_video.list_of_fragments.save_light_list(self.chosen_video.video._accumulation_folder)



    def save_after_second_accumulation(self):
        Logger.info("Saving second accumulation parameters")
        self.save_and_update_accumulation_parameters_in_parachute()
        self.chosen_video.video.accumulation_trial = np.argmax(self.chosen_video.video.percentage_of_accumulated_images)
        self.chosen_video.video._ratio_accumulated_images = self.chosen_video.video.percentage_of_accumulated_images[self.chosen_video.video.accumulation_trial]
        accumulation_folder_name = 'accumulation_' + str(self.chosen_video.video.accumulation_trial)
        self.chosen_video.video._accumulation_folder = os.path.join(self.chosen_video.video.session_folder, accumulation_folder_name)
        self.chosen_video.list_of_fragments.load_light_list(self.chosen_video.video._accumulation_folder)
        self.chosen_video.video._second_accumulation_finished = True
        Logger.info("Saving global fragments")
        self.chosen_video.list_of_fragments.save(self.chosen_video.video.fragments_path)
        self.chosen_video.list_of_global_fragments.save(self.chosen_video.video.global_fragments_path, self.chosen_video.list_of_fragments.fragments)
        Logger.info("Restoring networks to best second accumulation")
        self.accumulation_network_params.restore_folder = self.chosen_video.video._accumulation_folder
        self.net = ConvNetwork(self.accumulation_network_params)
        self.net.restore()
        self.chosen_video.video._accumulation_network_params = self.accumulation_network_params
        self.chosen_video.video.save()




    def init_pretraining_variables(self):
        self.init_pretraining_net()
        self.pretraining_global_step = 0
        self.net = ConvNetwork(self.pretrain_network_params)
        self.ratio_of_pretrained_images = 0
        if self.chosen_video.video.tracking_with_knowledge_transfer:
            self.net.restore()
        self.store_training_accuracy_and_loss_data_pretrain = Store_Accuracy_and_Loss(self.net,
                                                                                    name = 'training',
                                                                                    scope = 'pretraining')
        self.store_validation_accuracy_and_loss_data_pretrain = Store_Accuracy_and_Loss(self.net,
                                                                                    name = 'validation',
                                                                                    scope = 'pretraining')
        self.create_pretraining_figure()

    def pretraining_loop(self):
        Logger.debug("------------------------> pretraining_loop")
        self.chosen_video.list_of_fragments.reset(roll_back_to = 'fragmentation')
        self.chosen_video.list_of_global_fragments.order_by_distance_travelled()
        self.pretraining_popup.bind(on_open = self.one_shot_pretraining)
        self.pretraining_popup.open()
        Clock.schedule_interval(self.continue_pretraining, 2)

    def continue_pretraining(self, *args):
        Logger.debug("------------------------> continue_pretraining")
        if self.pretraining_step_finished and self.ratio_of_pretrained_images < MAX_RATIO_OF_PRETRAINED_IMAGES:
            self.one_shot_pretraining()
        elif self.ratio_of_pretrained_images > MAX_RATIO_OF_PRETRAINED_IMAGES:
            self.chosen_video.video._has_been_pretrained = True
            Clock.unschedule(self.continue_pretraining)
            Logger.warning('Calling accumulate from continue_pretraining')
            Logger.debug('****** saving protocol3 pretraining time')
            self.chosen_video.video._protocol3_pretraining_time = time.time()-self.chosen_video.video.protocol3_pretraining_time
            self.accumulate()

    def one_shot_pretraining(self, *args):
        Logger.debug("------------------------> one_shot_pretraining")
        self.pretraining_step_finished = False
        self.pretraining_global_fragment = self.chosen_video.list_of_global_fragments.global_fragments[self.pretraining_counter]
        self.net,\
        self.ratio_of_pretrained_images,\
        pretraining_global_step,\
        self.store_training_accuracy_and_loss_data_pretrain,\
        self.store_validation_accuracy_and_loss_data_pretrain,\
        self.chosen_video.list_of_fragments = pre_train_global_fragment(self.net,
                                                    self.pretraining_global_fragment,
                                                    self.chosen_video.list_of_fragments,
                                                    self.pretraining_global_step,
                                                    True, True,
                                                    self.generate_tensorboard_switch.active,
                                                    self.store_training_accuracy_and_loss_data_pretrain,
                                                    self.store_validation_accuracy_and_loss_data_pretrain,
                                                    print_flag = False,
                                                    plot_flag = False,
                                                    batch_size = BATCH_SIZE_IDCNN,
                                                    canvas_from_GUI = self.pretrain_fig_canvas)
        self.pretraining_counter += 1
        self.pretraining_counter_value.text = str(self.pretraining_counter)
        self.percentage_pretrained_images_value.text = str(self.ratio_of_pretrained_images)
        self.store_training_accuracy_and_loss_data_pretrain.plot_global_fragments(self.pretrain_ax_arr,
                                                                    self.chosen_video.video,
                                                                    self.chosen_video.list_of_fragments.fragments,
                                                                    black = False,
                                                                    canvas_from_GUI = self.pretrain_fig_canvas)
        self.store_validation_accuracy_and_loss_data_pretrain.plot(self.pretrain_ax_arr,
                                                    color ='b',
                                                    canvas_from_GUI = self.pretrain_fig_canvas,
                                                    index = self.pretraining_global_step,
                                                    legend_font_color = 'w')
        self.store_training_accuracy_and_loss_data_pretrain.plot(self.pretrain_ax_arr,
                                                    color = 'r',
                                                    canvas_from_GUI = self.pretrain_fig_canvas,
                                                    index = self.pretraining_global_step,
                                                    legend_font_color = 'w')
        self.pretraining_step_finished = True

    def identify(self, *args):
        self.chosen_video.video._identify_time = time.time()
        Logger.warning("In identify")
        self.chosen_video.list_of_fragments.reset(roll_back_to = 'accumulation')
        Logger.warning("Calling assigner")
        assigner(self.chosen_video.list_of_fragments, self.chosen_video.video, self.net)
        self.chosen_video.video._has_been_assigned = True
        self.chosen_video.video.save()
        self.identification_popup.dismiss()
        self.impossible_jumps_popup.open()

    def postprocess_impossible_jumps(self, *args):
        super().postprocess_impossible_jumps(*args)
        self.impossible_jumps_popup.dismiss()

    def update_list_of_blobs(self, *args):
        super().update_list_of_blobs(*args)
        self.trajectories_popup.open()

    def create_trajectories(self, *args):
        super().create_trajectories(*args,
            trajectories_popup_dismiss=self.trajectories_popup.dismiss,
            interpolate_crossings_popup_open=self.interpolate_crossings_popup.open,
            update_and_show_happy_ending_popup=self.update_and_show_happy_ending_popup
        )

    def interpolate_crossings(self, *args):
        super().interpolate_crossings(*args)
        self.interpolate_crossings_popup.dismiss()
        self.trajectories_wo_gaps_popup.open()

    def create_trajectories_wo_gaps(self, *args):
        super().create_trajectories_wo_gaps(*args)
        self.trajectories_wo_gaps_popup.dismiss()

    def update_and_show_happy_ending_popup(self, *args):
        super().update_and_show_happy_ending_popup(*args)
        self.create_happy_ending_popup(self.chosen_video.video.overall_P2)
        self.this_is_the_end_popup.open()
        self.deactivate_validation.setter(False)



    def start_from_identification(self, *args):
        self.identification_popup.open()

    def start_from_impossible_jumps(self, *args):
        self.impossible_jumps_popup.open()

    def start_from_post_processing(self, *args):
        self.trajectories_popup.open()

    def start_from_crossings_solved(self, *args):
        self.interpolate_crossings_popup.open()

    def create_main_layout(self):
        self.start_tracking_button = Button(text = "Start protocol cascade")
        self.control_panel.add_widget(self.start_tracking_button)
        if self.chosen_video.video.number_of_animals != 1 and self.chosen_video.list_of_global_fragments.number_of_global_fragments != 1:
            self.advanced_controls_button = Button(text = "Advanced idCNN\ncontrols")
            self.control_panel.add_widget(self.advanced_controls_button)
            self.track_wo_identities_button = Button(text = "Track without identities")
            self.control_panel.add_widget(self.track_wo_identities_button)
            self.generate_tensorboard_label = CustomLabel(font_size = 16,
                                                        text = "Save tensorboard summaries",
                                                        size_hint = (1.,.5))
            self.generate_tensorboard_switch = Switch(size_hint = (1.,.15))
            self.control_panel.add_widget(self.generate_tensorboard_label)
            self.control_panel.add_widget(self.generate_tensorboard_switch)
            self.create_network_params_labels()
            self.generate_tensorboard_switch.active = False
            self.create_display_network_parameters()
            self.create_advanced_controls_popup()
            self.create_identification_popup()
            self.create_impossible_jumps_popup()
            self.create_trajectories_popup()
            self.create_interpolate_during_crossings_popup()
            self.create_trajectories_wo_gaps_popup()
            if self.chosen_video.video.number_of_channels > 3:
                self.color_tracking_label = CustomLabel(font_size = 16,
                                                        text = "Enable color-tracking")
                self.color_tracking_switch = Switch()
                self.control_panel.add_widget(self.color_tracking_label)
                self.control_panel.add_widget(self.color_tracking_switch)
                self.color_tracking_switch.active = False
            self.advanced_controls_button.bind(on_press = self.show_advanced_controls)
        else:
            self.create_trajectories_popup()

    def show_advanced_controls(self, *args):
        self.advanced_controls_popup.open()

    def create_advanced_controls_popup(self):
        self.container = BoxLayout(orientation = "vertical")
        self.parameters_grid = GridLayout(cols = 2)
        self.disclaimer_box = BoxLayout(size_hint = (1,.3))
        self.disclaimer = CustomLabel(font_size = 14, text = "Modify the identification network parameters only if you fully understand the feature that you are changing. "+
                                                    "After modifying each parameter press return. Click outside of the popup to go back to the main window")
        self.disclaimer_box.add_widget(self.disclaimer)
        self.container.add_widget(self.disclaimer_box)
        self.container.add_widget(self.parameters_grid)
        self.mod_cnn_model_label = CustomLabel(font_size = 14, text = "CNN model (add your model in the cnn_architectures.py module): ")
        self.mod_cnn_model_text_input = TextInput(text = self.str_model, multiline=False)
        self.mod_learning_rate_label = CustomLabel(font_size = 14, text = "Learning rate")
        self.mod_learning_rate_text_input = TextInput(text = self.str_lr, multiline=False)
        self.mod_keep_prob_label = CustomLabel(font_size = 14, text = "Dropout ratio. If 1.0, no dropout is performed (for fully connected layers excluding softmax): ")
        self.mod_keep_prob_text_input = TextInput(text = self.str_kp, multiline=False)
        self.mod_optimiser_label = CustomLabel(font_size = 14, text = "Optimiser. Acceptable optimisers: SGD and Adam ")
        self.mod_optimiser_text_input = TextInput(text = self.str_optimiser, multiline=False)
        self.mod_scopes_layers_to_optimize_label = CustomLabel(font_size = 14, text = "Layers to train. Either all or fully")
        self.mod_scopes_layers_to_optimize_text_input = TextInput(text = self.str_layers_to_train, multiline=False)
        self.mod_save_folder_label = CustomLabel(font_size = 14, text = "Save folder [path where the model will be saved]: ")
        self.mod_save_folder_text_input = TextInput(text = self.save_folder, multiline=False)
        self.mod_knowledge_transfer_folder_label = CustomLabel(font_size = 14, text = "Knowledge transfer folder [path to load convolutional weights from a pre-trained model]: ")
        self.mod_knowledge_transfer_folder_text_input = TextInput(text = self.knowledge_transfer_folder, multiline=False)
        # self.mod_kt_conv_layers_to_discard_label = CustomLabel(font_size = 14, text = "Convolutional layers to discard from the transfered network. (e.g: conv3, conv2)")
        # self.mod_kt_conv_layers_to_discard_text_input = TextInput(text = self.kt_conv_layers_to_discard, multiline=False)
        items_to_add = [self.mod_cnn_model_label, self.mod_cnn_model_text_input,
                        self.mod_learning_rate_label, self.mod_learning_rate_text_input,
                        self.mod_keep_prob_label, self.mod_keep_prob_text_input,
                        self.mod_optimiser_label, self.mod_optimiser_text_input,
                        self.mod_scopes_layers_to_optimize_label,
                        self.mod_scopes_layers_to_optimize_text_input,
                        self.mod_save_folder_label,
                        self.mod_save_folder_text_input,
                        self.mod_knowledge_transfer_folder_label,
                        self.mod_knowledge_transfer_folder_text_input]
                        # self.mod_kt_conv_layers_to_discard_label,
                        # self.mod_kt_conv_layers_to_discard_text_input]
        [self.parameters_grid.add_widget(item) for item in items_to_add]
        self.advanced_controls_popup = Popup(title='Advanced identification network controls',
                                             content=self.container,
                                             size_hint=(.9,.9))
        self.bind_network_controls()

    def bind_network_controls(self):
        self.mod_cnn_model_text_input.bind(on_text_validate = self.on_enter_mod_cnn_model_text_input)
        self.mod_learning_rate_text_input.bind(on_text_validate = self.on_enter_mod_learning_rate_text_input)
        self.mod_keep_prob_text_input.bind(on_text_validate = self.on_enter_mod_keep_prob_text_input)
        self.mod_optimiser_text_input.bind(on_text_validate = self.on_enter_mod_optimiser_text_input)
        self.mod_scopes_layers_to_optimize_text_input.bind(on_text_validate = self.on_enter_mod_scopes_layers_to_optimize_text_input)
        self.mod_save_folder_text_input.bind(on_text_validate = self.on_enter_mod_save_folder_text_input)
        self.mod_knowledge_transfer_folder_text_input.bind(on_text_validate = self.on_enter_mod_knowledge_transfer_folder_text_input)
        # self.mod_kt_conv_layers_to_discard_text_input.bind(on_text_validate = self.on_enter_mod_kt_conv_layers_to_discard_text_input)

    def on_enter_mod_cnn_model_text_input(self, *args):
        self.accumulation_network_params._cnn_model = int(self.mod_cnn_model_text_input.text)
        self.cnn_model_value.text = self.mod_cnn_model_text_input.text

    def on_enter_mod_learning_rate_text_input(self, *args):
        self.accumulation_network_params.learning_rate = float(self.mod_learning_rate_text_input.text)
        self.learning_rate_value.text = self.mod_learning_rate_text_input.text

    def on_enter_mod_keep_prob_text_input(self, *args):
        self.accumulation_network_params.keep_prob = float(self.mod_keep_prob_text_input.text)
        self.keep_prob_value.text = self.mod_keep_prob_text_input.text


    def on_enter_mod_optimiser_text_input(self, *args):
        if  self.mod_optimiser_text_input.text == "SGD":
            use_adam_optimiser = False
        elif self.mod_optimiser_text_input.text.lower() == "adam":
            use_adam_optimiser = True
        self.accumulation_network_params.use_adam_optimiser = use_adam_optimiser
        self.optimiser_value.text = self.mod_optimiser_text_input.text

    def on_enter_mod_scopes_layers_to_optimize_text_input(self, *args):
        if self.mod_scopes_layers_to_optimize_text_input.text == "all":
            scopes_layers_to_optimize = None
        elif self.mod_scopes_layers_to_optimize_text_input.text == "fully":
            scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax']
        self.accumulation_network_params.scopes_layers_to_optimize = scopes_layers_to_optimize
        self.scopes_layers_to_optimize_value.text = self.mod_scopes_layers_to_optimize_text_input.text

    def on_enter_mod_save_folder_text_input(self, *args):
        self.accumulation_network_params._save_folder = self.mod_save_folder_text_input.text
        self.save_folder_value.text = self.mod_save_folder_text_input.text

    def on_enter_mod_knowledge_transfer_folder_text_input(self, *args):
        self.accumulation_network_params._knowledge_transfer_folder = self.mod_knowledge_transfer_folder_text_input.text
        self.knowledge_transfer_folder_value.text = self.mod_knowledge_transfer_folder_text_input.text
        if os.path.isdir(self.accumulation_network_params.knowledge_transfer_folder):
            self.chosen_video.video._tracking_with_knowledge_transfer = True
            self.chosen_video.video._knowledge_transfer_model_folder = self.accumulation_network_params._knowledge_transfer_folder

    # def on_enter_mod_kt_conv_layers_to_discard_text_input(self, *args):
    #     print("******",self.mod_kt_conv_layers_to_discard_text_input.text)
    #     if self.mod_kt_conv_layers_to_discard_text_input.text == 'None':
    #         print("is None")
    #         self.accumulation_network_params._kt_conv_layers_to_discard = None
    #     else:
    #         print("is not None")
    #         self.accumulation_network_params._kt_conv_layers_to_discard = self.mod_kt_conv_layers_to_discard_text_input.text
    #         self.chosen_video.video._kt_conv_layers_to_discard = self.accumulation_network_params._kt_conv_layers_to_discard


    def network_params_to_string(self):
        self.str_model = str(self.accumulation_network_params.cnn_model)
        self.str_lr = str(self.accumulation_network_params.learning_rate)
        self.str_kp = str(self.accumulation_network_params.keep_prob)
        self.str_optimiser = "SGD" if not self.accumulation_network_params.use_adam_optimiser else "Adam"
        self.str_layers_to_train = "all" if self.accumulation_network_params.scopes_layers_to_optimize is None else str(self.accumulation_network_params.scopes_layers_to_optimize)
        self.restore_folder = self.accumulation_network_params.restore_folder if self.accumulation_network_params.restore_folder is not None else 'None'
        self.save_folder = self.accumulation_network_params.save_folder if self.accumulation_network_params.save_folder is not None else 'None'
        self.knowledge_transfer_folder = self.accumulation_network_params.knowledge_transfer_folder if self.accumulation_network_params.knowledge_transfer_folder is not None else 'None'
        # self.kt_conv_layers_to_discard = self.accumulation_network_params.kt_conv_layers_to_discard if self.accumulation_network_params.kt_conv_layers_to_discard is not None else 'None'

    def create_network_params_labels(self):
        self.cnn_model_label = CustomLabel(font_size = 14, text = "CNN model: ", halign = "left")
        self.learning_rate_label = CustomLabel(font_size = 14, text = "learning_rate: ", halign = "left")
        self.keep_prob_label = CustomLabel(font_size = 14, text = "Dropout ratio: ", halign = "left")
        self.optimiser_label = CustomLabel(font_size = 14, text = "Optimiser: ", halign = "left")
        self.scopes_layers_to_optimize_label = CustomLabel(font_size = 14, text = "Layers to train: ", halign = "left")
        self.restore_folder_label = CustomLabel(font_size = 14, text = "Restore Folder: ", halign = "left")
        self.save_folder_label = CustomLabel(font_size = 14, text = "Save folder : ", halign = "left")
        self.knowledge_transfer_folder_label = CustomLabel(font_size = 14, text = "Knowledge transfer folder: ", halign = "left")
        self.image_size_label = CustomLabel(font_size = 14, text = "Image size: ", halign = "left")

    def get_network_parameters(self):
        self.network_params_to_string()
        self.cnn_model_value = CustomLabel(font_size = 14, text = self.str_model, halign = "left")
        self.learning_rate_value = CustomLabel(font_size = 14, text = self.str_lr, halign = "left")
        self.keep_prob_value = CustomLabel(font_size = 14, text = self.str_kp, halign = "left")
        self.optimiser_value = CustomLabel(font_size = 14, text = self.str_optimiser, halign = "left")
        self.scopes_layers_to_optimize_value = CustomLabel(font_size = 14, text = self.str_layers_to_train, halign = "left")
        self.restore_folder_value = CustomLabel(font_size = 14, text = self.restore_folder, halign = "left")
        self.save_folder_value = CustomLabel(font_size = 14, text = self.save_folder, halign = "left")
        self.knowledge_transfer_folder_value = CustomLabel(font_size = 14, text = self.knowledge_transfer_folder, halign = "left")
        self.image_size_value = CustomLabel(font_size = 14, text = str(self.accumulation_network_params.image_size), halign = "left")

    def create_display_network_parameters(self):
        self.get_network_parameters()
        self.network_parameters_box = BoxLayout(orientation = "vertical")
        self.network_parameters_box_title = CustomLabel(font_size = 20, text = "identificaiton network parameters:", size_hint = (1.,.1), halign = "left")
        self.network_parameters_grid = GridLayout(cols = 2)
        network_parameters_labels = [self.cnn_model_label, self.cnn_model_value,
                                    self.learning_rate_label,
                                    self.learning_rate_value,
                                    self.keep_prob_label, self.keep_prob_value,
                                    self.optimiser_label, self.optimiser_value,
                                    self.scopes_layers_to_optimize_label,
                                    self.scopes_layers_to_optimize_value,
                                    self.restore_folder_label,
                                    self.restore_folder_value,
                                    self.save_folder_label, self.save_folder_value,
                                    self.knowledge_transfer_folder_label,
                                    self.knowledge_transfer_folder_value,
                                    self.image_size_label, self.image_size_value]
        title_and_labels_grid = [self.network_parameters_box_title, self.network_parameters_grid]
        [self.network_parameters_box.add_widget(widget) for widget in title_and_labels_grid]
        [self.network_parameters_grid.add_widget(label) for label in network_parameters_labels]
        self.add_widget(BoxLayout(size_hint = (.0125,1.)))
        self.add_widget(self.network_parameters_box)
        self.add_widget(BoxLayout(size_hint = (.0125,1.)))
        self.network_parameters_grid.height = self.network_parameters_grid.minimum_height
        self.network_parameters_grid.width = self.network_parameters_grid.minimum_width

    def create_pretraining_popup(self, *args):
        self.pretraining_popup_container = BoxLayout(orientation = "vertical")
        self.pretraining_popup_data_container = GridLayout(cols = 2, size_hint = (1., .3))
        self.pretraining_counter = 0
        self.pretraining_counter_label = CustomLabel(text = "iteration: ", size_hint = (1., .2))
        self.pretraining_counter_value = CustomLabel(text = str(self.pretraining_counter + 1), size_hint = (1., .2))
        self.percentage_pretrained_images_label = CustomLabel(text = "percentage of pretrained images: ", size_hint = (1., .2))
        self.percentage_pretrained_images_value = CustomLabel(text = "Training: Wait for the data ...", size_hint = (1., .2))
        self.pretraining_popup_data_container.add_widget(self.pretraining_counter_label)
        self.pretraining_popup_data_container.add_widget(self.pretraining_counter_value)
        self.pretraining_popup_data_container.add_widget(self.percentage_pretrained_images_label)
        self.pretraining_popup_data_container.add_widget(self.percentage_pretrained_images_value)
        self.pretraining_image_box = BoxLayout()
        self.pretraining_popup_container.add_widget(self.pretraining_popup_data_container)
        self.pretraining_popup_container.add_widget(self.pretraining_image_box)
        self.pretraining_popup = Popup(title='Protocol3 (Pre-training): Learning features from the entire video ...',
            content= self.pretraining_popup_container,
            size_hint=(.8, .8))

    def create_one_shot_accumulation_popup(self, *args):
        self.one_shot_accumulation_popup_container = BoxLayout(orientation = "vertical")
        self.one_shot_accumulation_popup_data_container = GridLayout(cols = 2, size_hint = (1., .3))
        self.protocol_label = CustomLabel(text = "protocol : ", size_hint = (1., .2))
        self.protocol_value = CustomLabel(text = str(1), size_hint = (1., .2))
        self.accumulation_counter_label = CustomLabel(text = "iteration: ", size_hint = (1., .2))
        self.accumulation_counter_value = CustomLabel(text = str(self.accumulation_manager.counter + 1), size_hint = (1., .2))
        self.percentage_accumulated_images_label = CustomLabel(text = "percentage of accumulated images: ", size_hint = (1., .2))
        self.percentage_accumulated_images_value = CustomLabel(text = "Training: Wait for the data ...", size_hint = (1., .2))
        self.one_shot_accumulation_popup_data_container.add_widget(self.protocol_label)
        self.one_shot_accumulation_popup_data_container.add_widget(self.protocol_value)
        self.one_shot_accumulation_popup_data_container.add_widget(self.accumulation_counter_label)
        self.one_shot_accumulation_popup_data_container.add_widget(self.accumulation_counter_value)
        self.one_shot_accumulation_popup_data_container.add_widget(self.percentage_accumulated_images_label)
        self.one_shot_accumulation_popup_data_container.add_widget(self.percentage_accumulated_images_value)
        self.accumulation_image_box = BoxLayout()
        self.one_shot_accumulation_popup_container.add_widget(self.one_shot_accumulation_popup_data_container)
        self.one_shot_accumulation_popup_container.add_widget(self.accumulation_image_box)
        self.one_shot_accumulation_popup = Popup(title='Deep fingerprint protocols cascade',
            content= self.one_shot_accumulation_popup_container,
            size_hint=(.8, .8))

    def create_identification_popup(self):
        self.identification_label = CustomLabel(text = "Identifying the animals in frames not used for training.")
        self.identification_popup = Popup(title='Identification',
            content= self.identification_label,
            size_hint=(.4, .4))
        self.identification_popup.bind(on_open = self.identify)

    def create_impossible_jumps_popup(self):
        self.impossible_jumps_label = CustomLabel(text = "Detecting and correcting impossible velocity jumps in the animals trajectories.")
        self.impossible_jumps_popup = Popup(title='Postprocessing',
            content= self.impossible_jumps_label,
            size_hint=(.4, .4))
        self.impossible_jumps_popup.bind(on_open = self.postprocess_impossible_jumps)
        self.impossible_jumps_popup.bind(on_dismiss = self.update_list_of_blobs)

    def create_trajectories_popup(self):
        self.trajectories_label = CustomLabel(text = "Creating and saving trajectories file.")
        self.trajectories_popup = Popup(title='Outputting trajectories',
            content= self.trajectories_label,
            size_hint=(.4, .4))
        self.trajectories_popup.bind(on_open = self.create_trajectories)

    def create_interpolate_during_crossings_popup(self):
        self.interpolate_crossings_label = CustomLabel(text = "Identifying animals during crossings.")
        self.interpolate_crossings_popup = Popup(title='Crossing id interpolation',
            content= self.interpolate_crossings_label,
            size_hint=(.4, .4))
        self.interpolate_crossings_popup.bind(on_open = self.interpolate_crossings)

    def create_trajectories_wo_gaps_popup(self):
        self.trajectories_wo_gaps_label = CustomLabel(text = "Creating and saving trajectories file.")
        self.trajectories_wo_gaps_popup = Popup(title='Outputting trajectories',
            content= self.trajectories_wo_gaps_label,
            size_hint=(.4, .4))
        self.trajectories_wo_gaps_popup.bind(on_open = self.create_trajectories_wo_gaps)
        self.trajectories_wo_gaps_popup.bind(on_dismiss = self.update_and_show_happy_ending_popup)

    def create_happy_ending_popup(self, overall_P2 = None):
        self.this_is_the_end_grid = GridLayout(cols = 2)
        self.this_is_the_end_label = CustomLabel(text = "The video has been tracked with estimated accuracy:")
        self.output_information = CustomLabel(text = "The output files of the tracking including the trajectories" +
                                            " can be found in the folder: ")
        self.session_folder_info = CustomLabel(text = self.chosen_video.video.session_folder)
        if isinstance(overall_P2, float):
            overall_P2 = round(overall_P2 * 100, 2)
            if overall_P2 > 98:
                self.safe = True
            else:
                self.safe = False
        self.this_is_the_end_value = CustomLabel(text = str(overall_P2) + '%')
        self.quit_button = Button(text = "Quit")
        self.quit_button.bind(on_release = self.quit_app)
        go_to_validation_button_text = "Validate the tracking" if self.safe else "Validate the tracking\n(recommended)"
        self.go_to_validation_button = Button(text = go_to_validation_button_text)
        end_widgets = [self.this_is_the_end_label, self.this_is_the_end_value,
                        self.output_information ,self.session_folder_info,
                        self.quit_button, self.go_to_validation_button]
        [self.this_is_the_end_grid.add_widget(w) for w in end_widgets]
        self.this_is_the_end_popup = Popup(title = "Process finished",
            content = self.this_is_the_end_grid,
            size_hint = (.8,.8))

    def quit_app(self,  *args):
        Logger.critical("Good bye")
        App.get_running_app().stop()
        Window.close()

    @staticmethod
    def set_matplotlib_params(font_size = 8):
        matplotlib.rcParams.update({'font.size': font_size, 'axes.labelsize': font_size,
                                    'xtick.labelsize' : font_size, 'ytick.labelsize' : font_size,
                                    'legend.fontsize': font_size})

    def create_tracking_figures_axes(self):
        if hasattr(self, 'fig'):
            self.remove_widget(self.tracking_fig_canvas)
            self.fig.clear()
        self.set_matplotlib_params()
        self.fig, self.ax_arr = plt.subplots(3)
        self.fig.set_facecolor((.188, .188, .188))
        [(ax.set_facecolor((.188, .188, .188)), ax.tick_params(color='white', labelcolor='white'), ax.xaxis.label.set_color('white'), ax.yaxis.label.set_color('white')) for ax in self.ax_arr]
        [spine.set_edgecolor('white') for ax in self.ax_arr for spine in ax.spines.values()]
        self.fig.canvas.set_window_title('Accumulation ' + str(self.chosen_video.video.accumulation_trial))
        self.fig.subplots_adjust(left = .1, bottom = .2, right = .9, top = .9, wspace = None, hspace = 1.)
        self.tracking_main_figure = FigureCanvasKivyAgg(self.fig)
        self.tracking_fig_canvas = self.fig.canvas
        self.accumulation_image_box.add_widget(self.tracking_fig_canvas)

    def create_pretraining_figure(self):
        self.set_matplotlib_params()
        self.pretrain_fig, self.pretrain_ax_arr = plt.subplots(3)
        self.pretrain_fig.canvas.set_window_title('Pretraining')
        self.pretrain_fig.set_facecolor((.188, .188, .188))
        [(ax.set_facecolor((.188, .188, .188)), ax.tick_params(color='white', labelcolor='white'), ax.xaxis.label.set_color('white'), ax.yaxis.label.set_color('white')) for ax in self.pretrain_ax_arr]
        [spine.set_edgecolor('white') for ax in self.pretrain_ax_arr for spine in ax.spines.values()]
        self.tracking_main_figure = FigureCanvasKivyAgg(self.pretrain_fig)
        self.pretrain_fig_canvas = self.pretrain_fig.canvas
        self.pretrain_fig.subplots_adjust(left = .1, bottom = .2, right = .9, top = .9, wspace = None, hspace = 1.)
        self.epoch_index_to_plot = 0
        self.pretraining_image_box.add_widget(self.pretrain_fig_canvas)
