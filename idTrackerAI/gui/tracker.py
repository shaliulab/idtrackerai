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
sys.path.append('../postprocessing')
sys.path.append('../network')
sys.path.append('../network/crossings_detector_model')
sys.path.append('../network/identification_model')
sys.path.append('../plots')
import copy
from segmentation import segment_frame, segment
from video_utils import blob_extractor
import numpy as np
from scipy.stats import mode
import cv2
from video import Video
from list_of_blobs import ListOfBlobs
from list_of_fragments import ListOfFragments
from list_of_global_fragments import ListOfGlobalFragments
from crossing_detector import detect_crossings
from list_of_fragments import create_list_of_fragments
from list_of_global_fragments import create_list_of_global_fragments
from accumulation_manager import AccumulationManager
from accumulator import perform_one_accumulation_step
from network_params import NetworkParams
from trainer import train
from assigner import assigner
from compute_velocity_model import compute_model_velocity
from correct_impossible_velocity_jumps import correct_impossible_velocity_jumps
from assign_them_all import close_trajectories_gaps
from get_trajectories import produce_output_dict
from pre_trainer import pre_train_global_fragment
from store_accuracy_and_loss import Store_Accuracy_and_Loss
from id_CNN import ConvNetwork
from constants import BATCH_SIZE_IDCNN
from constants import THRESHOLD_ACCEPTABLE_ACCUMULATION, VEL_PERCENTILE, THRESHOLD_EARLY_STOP_ACCUMULATION, MAX_RATIO_OF_PRETRAINED_IMAGES, MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS

class Tracker(BoxLayout):
    def __init__(self, chosen_video = None,
                deactivate_tracking = None,
                deactivate_validation = None,
                **kwargs):
        super(Tracker, self).__init__(**kwargs)
        global CHOSEN_VIDEO, DEACTIVATE_TRACKING, DEACTIVATE_VALIDATION
        CHOSEN_VIDEO = chosen_video
        DEACTIVATE_TRACKING = deactivate_tracking
        DEACTIVATE_VALIDATION = deactivate_validation
        self.control_panel = BoxLayout(orientation = "vertical", size_hint = (.26,1.))
        self.add_widget(self.control_panel)
        self.help_button_tracker = HelpButton()
        self.help_button_tracker.size_hint = (1.,.4)
        self.help_button_tracker.create_help_popup("Tracking",\
                                                "A message to help people.")

    def do(self):
        CHOSEN_VIDEO.video.accumulation_trial = 0
        delete = not CHOSEN_VIDEO.processes_to_restore['protocols1_and_2'] if 'protocols1_and_2' in CHOSEN_VIDEO.processes_to_restore.keys() else True
        CHOSEN_VIDEO.video.create_accumulation_folder(iteration_number = 0, delete = delete)
        self.number_of_animals = CHOSEN_VIDEO.video.number_of_animals if not CHOSEN_VIDEO.video.identity_transfer\
                                                                    else CHOSEN_VIDEO.video.knowledge_transfer_info_dict['number_of_animals']
        self.restoring_first_accumulation = False
        self.init_accumulation_network()
        self.create_main_layout()
        if 'post_processing' in CHOSEN_VIDEO.processes_to_restore and CHOSEN_VIDEO.processes_to_restore['post_processing']:
            self.restore_trajectories()
            self.restore_crossings_solved()
            self.restore_trajectories_wo_gaps()
            self.start_tracking_button.bind(on_release = self.update_and_show_happy_ending_popup)
            self.start_tracking_button.text = "Show estimated\naccuracy"
        elif 'residual_identification' in CHOSEN_VIDEO.processes_to_restore and CHOSEN_VIDEO.processes_to_restore['residual_identification']:
            Logger.info("Restoring residual identification")
            self.restore_identification()
            CHOSEN_VIDEO.video._has_been_assigned = True
            self.start_tracking_button.bind(on_release = self.start_from_post_processing)
            self.start_tracking_button.text = "Start\npost-processing"
        elif 'protocol3_accumulation' in CHOSEN_VIDEO.processes_to_restore and CHOSEN_VIDEO.processes_to_restore['protocol3_accumulation']:
            Logger.info("Restoring second accumulation")
            self.restore_second_accumulation()
            CHOSEN_VIDEO.video._first_frame_first_global_fragment = [CHOSEN_VIDEO.video.percentage_of_accumulated_images]
            Logger.info("Starting identification")
            self.start_tracking_button.bind(on_release = self.start_from_identification)
            self.start_tracking_button.text = "Start\nresidual identification"
        elif 'protocol3_pretraining' in CHOSEN_VIDEO.processes_to_restore and CHOSEN_VIDEO.processes_to_restore['protocol3_pretraining']:
            Logger.info("Restoring pretraining")
            Logger.info("Initialising pretraining network")
            self.init_pretraining_net()
            Logger.info("Restoring pretraining")
            self.accumulation_step_finished = True
            self.restore_first_accumulation()
            self.restore_pretraining()
            self.accumulation_manager.ratio_accumulated_images = CHOSEN_VIDEO.video.percentage_of_accumulated_images[0]
            CHOSEN_VIDEO.video._first_frame_first_global_fragment = [self.accumulation_manager.ratio_accumulated_images]
            self.create_one_shot_accumulation_popup()
            Logger.info("Start accumulation parachute")
            self.start_tracking_button.bind(on_release = self.accumulate)
            self.start_tracking_button.text = "Start\naccumulation (protocol 3)"
        elif 'protocols1_and_2' in CHOSEN_VIDEO.processes_to_restore and CHOSEN_VIDEO.processes_to_restore['protocols1_and_2']:
            Logger.info("Restoring protocol 1")
            self.restoring_first_accumulation = True
            self.restore_first_accumulation()
            self.accumulation_manager.ratio_accumulated_images = CHOSEN_VIDEO.video.percentage_of_accumulated_images[0]
            CHOSEN_VIDEO.video._first_frame_first_global_fragment = [CHOSEN_VIDEO.video.percentage_of_accumulated_images[0]]
            self.accumulation_step_finished = True
            self.create_one_shot_accumulation_popup()
            self.start_tracking_button.bind(on_release = self.accumulate)
            self.start_tracking_button.text = "Start\nidentification"
        elif 'protocols1_and_2' not in CHOSEN_VIDEO.processes_to_restore or not CHOSEN_VIDEO.processes_to_restore['protocols1_and_2']:
            Logger.info("Starting protocol cascade")
            self.start_tracking_button.bind(on_release = self.protocol1)
        self.control_panel.add_widget(self.help_button_tracker)

    def init_accumulation_network(self):
        self.accumulation_network_params = NetworkParams(self.number_of_animals,
                                    learning_rate = 0.005,
                                    keep_prob = 1.0,
                                    scopes_layers_to_optimize = None,
                                    save_folder = CHOSEN_VIDEO.video.accumulation_folder,
                                    image_size = CHOSEN_VIDEO.video.identification_image_size,
                                    video_path = CHOSEN_VIDEO.video.video_path)

    def protocol1(self, *args):
        CHOSEN_VIDEO.list_of_fragments.reset(roll_back_to = 'fragmentation')
        CHOSEN_VIDEO.list_of_global_fragments.reset(roll_back_to = 'fragmentation')
        if CHOSEN_VIDEO.video.tracking_with_knowledge_transfer:
            Logger.debug('Setting layers to optimize for knowledge_transfer')
            self.accumulation_network_params.scopes_layers_to_optimize = None
        self.net = ConvNetwork(self.accumulation_network_params)
        if CHOSEN_VIDEO.video.tracking_with_knowledge_transfer:
            Logger.debug('Restoring for knowledge transfer')
            self.net.restore()
        CHOSEN_VIDEO.video._first_frame_first_global_fragment.append(CHOSEN_VIDEO.list_of_global_fragments.set_first_global_fragment_for_accumulation(CHOSEN_VIDEO.video, accumulation_trial = 0))
        if CHOSEN_VIDEO.video.identity_transfer and\
            CHOSEN_VIDEO.video.number_of_animals < CHOSEN_VIDEO.video.knowledge_transfer_info_dict['number_of_animals']:
            tf.reset_default_graph()
            self.accumulation_network_params.number_of_animals = CHOSEN_VIDEO.video.number_of_animals
            self.accumulation_network_params._restore_folder = None
            self.accumulation_network_params.knowledge_transfer_folder = CHOSEN_VIDEO.video.knowledge_transfer_model_folder
            self.net = ConvNetwork(self.accumulation_network_params)
            self.net.restore()
        CHOSEN_VIDEO.list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(CHOSEN_VIDEO.video, accumulation_trial = 0)
        self.accumulation_manager = AccumulationManager(CHOSEN_VIDEO.video, CHOSEN_VIDEO.list_of_fragments,
                                                    CHOSEN_VIDEO.list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        self.global_step = 0
        self.create_one_shot_accumulation_popup()
        self.accumulation_step_finished = True
        self.accumulation_loop()

    def one_shot_accumulation(self, *args):
        self.accumulation_step_finished = False
        self.accumulation_manager.ratio_of_accumulated_images,\
        store_validation_accuracy_and_loss_data,\
        store_training_accuracy_and_loss_data = perform_one_accumulation_step(self.accumulation_manager,
                                                                                CHOSEN_VIDEO.video,
                                                                                self.global_step,
                                                                                self.net,
                                                                                CHOSEN_VIDEO.video.identity_transfer,
                                                                                save_summaries = self.generate_tensorboard_switch.active,
                                                                                GUI_axes = None,
                                                                                net_properties = None,
                                                                                plot_flag = False)
        self.accumulation_counter_value.text = str(self.accumulation_manager.counter + 1)
        print('*****************************', self.accumulation_manager.counter)
        if not hasattr(self, 'ax_arr'):
            print("------------ Creating axes")
            self.create_tracking_figures_axes()
        self.percentage_accumulated_images_value.text = str(self.accumulation_manager.ratio_of_accumulated_images)
        self.protocol_value.text = '2' if CHOSEN_VIDEO.video.accumulation_trial == 0 else '3'
        store_training_accuracy_and_loss_data.plot_global_fragments(self.ax_arr,
                                                                    CHOSEN_VIDEO.video,
                                                                    self.accumulation_manager.list_of_fragments.fragments,
                                                                    black = False,
                                                                    canvas_from_GUI = self.tracking_fig_canvas)
        store_validation_accuracy_and_loss_data.plot(self.ax_arr,
                                                    color ='b',
                                                    canvas_from_GUI = self.tracking_fig_canvas,
                                                    index = self.accumulation_manager.counter - 1,
                                                    legend_font_color = 'w')
        store_training_accuracy_and_loss_data.plot(self.ax_arr,
                                                    color = 'r',
                                                    canvas_from_GUI = self.tracking_fig_canvas,
                                                    index = self.accumulation_manager.counter - 1,
                                                    legend_font_color = 'w')
        self.accumulation_step_finished = True

    def accumulate(self, *args):
        Logger.info("------------------------> Calling accumulate")
        if self.accumulation_step_finished and self.accumulation_manager.continue_accumulation:
            Logger.info("--------------------> Performing accumulation")
            self.one_shot_accumulation()
        elif not self.accumulation_manager.continue_accumulation\
            and not CHOSEN_VIDEO.video.first_accumulation_finished\
            and self.accumulation_manager.ratio_accumulated_images > THRESHOLD_EARLY_STOP_ACCUMULATION:
            Logger.info("Protocol 1 successful")
            self.save_after_first_accumulation()
            self.identification_popup.open()
        elif not self.accumulation_manager.continue_accumulation\
            and not CHOSEN_VIDEO.video.has_been_pretrained:
            self.save_after_first_accumulation()
            if self.accumulation_manager.ratio_accumulated_images > THRESHOLD_ACCEPTABLE_ACCUMULATION:
                Logger.info("Protocol 2 successful")
                self.identification_popup.open()
            elif self.accumulation_manager.ratio_accumulated_images < THRESHOLD_ACCEPTABLE_ACCUMULATION:
                Logger.info("Protocol 2 failed -> Start protocol 3")
                self.create_pretraining_popup()
                self.protocol3()
        elif CHOSEN_VIDEO.video.has_been_pretrained\
            and CHOSEN_VIDEO.video.accumulation_trial < MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS\
            and self.accumulation_manager.ratio_accumulated_images < THRESHOLD_ACCEPTABLE_ACCUMULATION :
            Logger.info("Accumulation in protocol 3 is not successful. Opening parachute ...")
            CHOSEN_VIDEO.video._ratio_accumulated_images = self.accumulation_manager.ratio_accumulated_images
            CHOSEN_VIDEO.video._percentage_of_accumulated_images.append(CHOSEN_VIDEO.video.ratio_accumulated_images)
            CHOSEN_VIDEO.video.accumulation_trial += 1
            self.accumulation_parachute_init(CHOSEN_VIDEO.video.accumulation_trial)
            self.accumulation_loop()
        elif CHOSEN_VIDEO.video.has_been_pretrained and\
            (self.accumulation_manager.ratio_accumulated_images >= THRESHOLD_ACCEPTABLE_ACCUMULATION\
            or CHOSEN_VIDEO.video.accumulation_trial >= MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS):
            Logger.info("Accumulation after protocol 3 has been successful")
            self.save_after_second_accumulation()
            self.identification_popup.open()

    def accumulation_loop(self):
        if hasattr(self, 'one_shot_accumulation_popup'):
            delattr(self, 'one_shot_accumulation_popup')
            self.create_one_shot_accumulation_popup()
        CHOSEN_VIDEO.video.init_accumulation_statistics_attributes()
        self.accumulation_manager.threshold_early_stop_accumulation = THRESHOLD_EARLY_STOP_ACCUMULATION
        self.one_shot_accumulation_popup.bind(on_open = self.one_shot_accumulation)
        self.one_shot_accumulation_popup.open()
        Clock.schedule_interval(self.accumulate, 2)

    def protocol3(self):
        self.init_pretraining_variables()
        number_of_images_in_global_fragments = CHOSEN_VIDEO.video.number_of_unique_images_in_global_fragments
        if CHOSEN_VIDEO.old_video and CHOSEN_VIDEO.old_video.first_accumulation_finished == True:
            CHOSEN_VIDEO.list_of_global_fragments.reset(roll_back_to = 'fragmentation')
            CHOSEN_VIDEO.list_of_fragments.reset(roll_back_to = 'fragmentation')
        Logger.info("Starting pretraining. Checkpoints will be stored in %s" %CHOSEN_VIDEO.video.pretraining_folder)
        if CHOSEN_VIDEO.video.tracking_with_knowledge_transfer:
            Logger.info("Performing knowledge transfer from %s" %CHOSEN_VIDEO.video.knowledge_transfer_model_folder)
            self.pretrain_network_params.knowledge_transfer_folder = CHOSEN_VIDEO.video.knowledge_transfer_model_folder
        Logger.info("Start pretraining")
        self.pretraining_step_finished = True
        self.pretraining_loop()

    def accumulation_parachute_init(self, iteration_number):
        Logger.info("Starting accumulation %i" %iteration_number)
        self.one_shot_accumulation_popup.dismiss()
        delete = not CHOSEN_VIDEO.processes_to_restore['protocol3_accumulation'] if 'protocol3_accumulation' in CHOSEN_VIDEO.processes_to_restore.keys() else True
        CHOSEN_VIDEO.video.create_accumulation_folder(iteration_number = iteration_number, delete = delete)
        CHOSEN_VIDEO.video.accumulation_trial = iteration_number
        CHOSEN_VIDEO.list_of_fragments.reset(roll_back_to = 'fragmentation')
        CHOSEN_VIDEO.list_of_global_fragments.reset(roll_back_to = 'fragmentation')
        Logger.info("We will restore the network from a previous pretraining: %s" %CHOSEN_VIDEO.video.pretraining_folder)
        self.accumulation_network_params.save_folder = CHOSEN_VIDEO.video.accumulation_folder
        self.accumulation_network_params.restore_folder = CHOSEN_VIDEO.video.pretraining_folder
        self.accumulation_network_params.scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax']
        Logger.info("Initialising accumulation network")
        self.net = ConvNetwork(self.accumulation_network_params)
        self.net.restore()
        self.net.reinitialize_softmax_and_fully_connected()
        Logger.info("Initialising accumulation manager")
        CHOSEN_VIDEO.video._first_frame_first_global_fragment.append(CHOSEN_VIDEO.list_of_global_fragments.set_first_global_fragment_for_accumulation(CHOSEN_VIDEO.video, accumulation_trial = iteration_number - 1))
        if CHOSEN_VIDEO.video.identity_transfer and CHOSEN_VIDEO.video.number_of_animals < CHOSEN_VIDEO.video.knowledge_transfer_info_dict['number_of_animals']:
            tf.reset_default_graph()
            self.accumulation_network_params.number_of_animals = CHOSEN_VIDEO.video.number_of_animals
            self.accumulation_network_params.restore_folder = CHOSEN_VIDEO.video.pretraining_folder
            self.net = ConvNetwork(self.accumulation_network_params)
            self.net.restore()
            self.net.reinitialize_softmax_and_fully_connected()
        CHOSEN_VIDEO.list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(CHOSEN_VIDEO.video, accumulation_trial = iteration_number - 1)
        self.accumulation_manager = AccumulationManager(CHOSEN_VIDEO.video,
                                                    CHOSEN_VIDEO.list_of_fragments, CHOSEN_VIDEO.list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        Logger.info("Start accumulation")
        self.global_step = 0

    def save_after_first_accumulation(self):
        """Set flags and save data"""
        Logger.info("Saving first accumulation paramters")
        Clock.unschedule(self.accumulate)
        if not self.restoring_first_accumulation:
            CHOSEN_VIDEO.video._first_accumulation_finished = True
            CHOSEN_VIDEO.video._ratio_accumulated_images = self.accumulation_manager.ratio_accumulated_images
            CHOSEN_VIDEO.video._percentage_of_accumulated_images = [CHOSEN_VIDEO.video.ratio_accumulated_images]
            CHOSEN_VIDEO.video.save()
            CHOSEN_VIDEO.list_of_fragments.save(CHOSEN_VIDEO.video.fragments_path)
            CHOSEN_VIDEO.list_of_global_fragments.save(CHOSEN_VIDEO.video.global_fragments_path, CHOSEN_VIDEO.list_of_fragments.fragments)
            CHOSEN_VIDEO.list_of_fragments.save_light_list(CHOSEN_VIDEO.video._accumulation_folder)

    def save_after_second_accumulation(self):
        Logger.info("Saving second accumulation parameters")
        Clock.unschedule(self.accumulate)
        CHOSEN_VIDEO.video.accumulation_trial = np.argmax(CHOSEN_VIDEO.video.percentage_of_accumulated_images)
        CHOSEN_VIDEO.video._first_frame_first_global_fragment = CHOSEN_VIDEO.video.first_frame_first_global_fragment[CHOSEN_VIDEO.video.accumulation_trial]
        CHOSEN_VIDEO.video._ratio_accumulated_images = CHOSEN_VIDEO.video.percentage_of_accumulated_images[CHOSEN_VIDEO.video.accumulation_trial]
        accumulation_folder_name = 'accumulation_' + str(CHOSEN_VIDEO.video.accumulation_trial)
        CHOSEN_VIDEO.video._accumulation_folder = os.path.join(CHOSEN_VIDEO.video.session_folder, accumulation_folder_name)
        CHOSEN_VIDEO.list_of_fragments.save_light_list(CHOSEN_VIDEO.video._accumulation_folder)
        # CHOSEN_VIDEO.list_of_fragments.load_light_list(CHOSEN_VIDEO.video._accumulation_folder)
        CHOSEN_VIDEO.video._second_accumulation_finished = True
        Logger.info("Saving global fragments")
        CHOSEN_VIDEO.list_of_fragments.save(CHOSEN_VIDEO.video.fragments_path)
        CHOSEN_VIDEO.list_of_global_fragments.save(CHOSEN_VIDEO.video.global_fragments_path, CHOSEN_VIDEO.list_of_fragments.fragments)
        CHOSEN_VIDEO.video.save()

    def init_pretraining_net(self):
        delete = not CHOSEN_VIDEO.processes_to_restore['protocol3_pretraining'] if 'protocol3_pretraining' in CHOSEN_VIDEO.processes_to_restore.keys() else True
        CHOSEN_VIDEO.video.create_pretraining_folder(delete = delete)
        self.pretrain_network_params = NetworkParams(CHOSEN_VIDEO.video.number_of_animals,
                                                learning_rate = 0.01,
                                                keep_prob = 1.0,
                                                use_adam_optimiser = False,
                                                scopes_layers_to_optimize = None,
                                                save_folder = CHOSEN_VIDEO.video.pretraining_folder,
                                                image_size = CHOSEN_VIDEO.video.identification_image_size,
                                                video_path = CHOSEN_VIDEO.video.video_path)

    def init_pretraining_variables(self):
        self.init_pretraining_net()
        self.pretraining_global_step = 0
        self.net = ConvNetwork(self.pretrain_network_params)
        self.ratio_of_pretrained_images = 0
        if CHOSEN_VIDEO.video.tracking_with_knowledge_transfer:
            self.net.restore()
        self.store_training_accuracy_and_loss_data_pretrain = Store_Accuracy_and_Loss(self.net,
                                                                                    name = 'training',
                                                                                    scope = 'pretraining')
        self.store_validation_accuracy_and_loss_data_pretrain = Store_Accuracy_and_Loss(self.net,
                                                                                    name = 'validation',
                                                                                    scope = 'pretraining')
        self.create_pretraining_figure()

    def pretraining_loop(self):
        CHOSEN_VIDEO.list_of_fragments.reset(roll_back_to = 'fragmentation')
        CHOSEN_VIDEO.list_of_global_fragments.order_by_distance_travelled()
        self.pretraining_popup.bind(on_open = self.one_shot_pretraining)
        self.pretraining_popup.open()
        Clock.schedule_interval(self.continue_pretraining, 2)

    def continue_pretraining(self, *args):
        if self.pretraining_step_finished and self.ratio_of_pretrained_images < MAX_RATIO_OF_PRETRAINED_IMAGES:
            self.one_shot_pretraining()
        elif self.ratio_of_pretrained_images > MAX_RATIO_OF_PRETRAINED_IMAGES:
            CHOSEN_VIDEO.video._has_been_pretrained = True
            Clock.unschedule(self.continue_pretraining)
            self.accumulate()

    def one_shot_pretraining(self, *args):
        self.pretraining_step_finished = False
        self.pretraining_global_fragment = CHOSEN_VIDEO.list_of_global_fragments.global_fragments[self.pretraining_counter]
        self.net,\
        self.ratio_of_pretrained_images,\
        pretraining_global_step,\
        self.store_training_accuracy_and_loss_data_pretrain,\
        self.store_validation_accuracy_and_loss_data_pretrain,\
        CHOSEN_VIDEO.list_of_fragments = pre_train_global_fragment(self.net,
                                                    self.pretraining_global_fragment,
                                                    CHOSEN_VIDEO.list_of_fragments,
                                                    self.pretraining_global_step,
                                                    True, True,
                                                    self.generate_tensorboard_switch.active,
                                                    self.store_training_accuracy_and_loss_data_pretrain,
                                                    self.store_validation_accuracy_and_loss_data_pretrain,
                                                    print_flag = False,
                                                    plot_flag = False,
                                                    batch_size = self.batch_size,
                                                    canvas_from_GUI = self.pretrain_fig_canvas)
        self.pretraining_counter += 1
        self.pretraining_counter_value.text = str(self.pretraining_counter)
        self.percentage_pretrained_images_value.text = str(self.ratio_of_pretrained_images)
        self.store_training_accuracy_and_loss_data_pretrain.plot_global_fragments(self.pretrain_ax_arr,
                                                                    CHOSEN_VIDEO.video,
                                                                    CHOSEN_VIDEO.list_of_fragments.fragments,
                                                                    black = False,
                                                                    canvas_from_GUI = self.pretrain_fig_canvas)
        self.store_validation_accuracy_and_loss_data_pretrain.plot(self.pretrain_ax_arr,
                                                    color ='b',
                                                    canvas_from_GUI = self.pretrain_fig_canvas,
                                                    legend_font_color = 'w')
        self.store_training_accuracy_and_loss_data_pretrain.plot(self.pretrain_ax_arr,
                                                    color = 'r',
                                                    canvas_from_GUI = self.pretrain_fig_canvas,
                                                    legend_font_color = 'w')
        self.pretraining_step_finished = True

    def identify(self, *args):
        if isinstance(CHOSEN_VIDEO.video.first_frame_first_global_fragment, list):
            CHOSEN_VIDEO.video._first_frame_first_global_fragment = CHOSEN_VIDEO.video.first_frame_first_global_fragment[CHOSEN_VIDEO.video.accumulation_trial]
        CHOSEN_VIDEO.list_of_fragments.reset(roll_back_to = 'accumulation')
        assigner(CHOSEN_VIDEO.list_of_fragments, CHOSEN_VIDEO.video, self.net)
        CHOSEN_VIDEO.video._has_been_assigned = True
        CHOSEN_VIDEO.video.save()
        self.identification_popup.dismiss()
        self.impossible_jumps_popup.open()

    def postprocess_impossible_jumps(self, *args):
        if not hasattr(CHOSEN_VIDEO.video, 'velocity_threshold') and hasattr(CHOSEN_VIDEO.old_video,'velocity_threshold'):
            CHOSEN_VIDEO.video.velocity_threshold = CHOSEN_VIDEO.old_video.velocity_threshold
        elif not hasattr(CHOSEN_VIDEO.old_video, 'velocity_threshold'):
            CHOSEN_VIDEO.video.velocity_threshold = compute_model_velocity(
                                                                CHOSEN_VIDEO.list_of_fragments.fragments,
                                                                CHOSEN_VIDEO.video.number_of_animals,
                                                                percentile = VEL_PERCENTILE)
        correct_impossible_velocity_jumps(CHOSEN_VIDEO.video, CHOSEN_VIDEO.list_of_fragments)
        CHOSEN_VIDEO.list_of_fragments.save(CHOSEN_VIDEO.video.fragments_path)
        CHOSEN_VIDEO.video.save()
        self.impossible_jumps_popup.dismiss()

    def update_list_of_blobs(self, *args):
        CHOSEN_VIDEO.video.individual_fragments_stats = CHOSEN_VIDEO.list_of_fragments.get_stats(CHOSEN_VIDEO.list_of_global_fragments)
        CHOSEN_VIDEO.video.compute_overall_P2(CHOSEN_VIDEO.list_of_fragments.fragments)
        CHOSEN_VIDEO.list_of_fragments.save_light_list(CHOSEN_VIDEO.video._accumulation_folder)
        CHOSEN_VIDEO.video.save()
        if not hasattr(CHOSEN_VIDEO, 'list_of_blobs'):
            CHOSEN_VIDEO.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video, CHOSEN_VIDEO.old_video.blobs_path)
        CHOSEN_VIDEO.list_of_blobs.update_from_list_of_fragments(CHOSEN_VIDEO.list_of_fragments.fragments,
                                                    CHOSEN_VIDEO.video.fragment_identifier_to_index)
        # if False:
        #     CHOSEN_VIDEO.list_of_blobs.compute_nose_and_head_coordinates()
        CHOSEN_VIDEO.list_of_blobs.save(CHOSEN_VIDEO.video,
                                        CHOSEN_VIDEO.video.blobs_path,
                                        number_of_chunks = CHOSEN_VIDEO.video.number_of_frames)
        self.trajectories_popup.open()

    def create_trajectories(self, *args):
        if 'post_processing' not in CHOSEN_VIDEO.processes_to_restore or not CHOSEN_VIDEO.processes_to_restore['post_processing']:
            CHOSEN_VIDEO.video.create_trajectories_folder()
            trajectories_file = os.path.join(CHOSEN_VIDEO.video.trajectories_folder, 'trajectories.npy')
            trajectories = produce_output_dict(CHOSEN_VIDEO.list_of_blobs.blobs_in_video, CHOSEN_VIDEO.video)
            np.save(trajectories_file, trajectories)
            Logger.info("Saving trajectories")
        CHOSEN_VIDEO.video._has_trajectories = True
        CHOSEN_VIDEO.video.save()
        self.trajectories_popup.dismiss()
        self.interpolate_crossings_popup.open()

    def interpolate_crossings(self, *args):
        CHOSEN_VIDEO.list_of_blobs_no_gaps = copy.deepcopy(CHOSEN_VIDEO.list_of_blobs)
        if not hasattr(CHOSEN_VIDEO.list_of_blobs_no_gaps.blobs_in_video[0][0], '_was_a_crossing'):
            Logger.debug("adding attribute was_a_crossing to every blob")
            [setattr(blob, '_was_a_crossing', False) for blobs_in_frame in
                CHOSEN_VIDEO.list_of_blobs_no_gaps.blobs_in_video for blob in blobs_in_frame]
        CHOSEN_VIDEO.video._has_crossings_solved = False
        CHOSEN_VIDEO.list_of_blobs_no_gaps = close_trajectories_gaps(CHOSEN_VIDEO.video, CHOSEN_VIDEO.list_of_blobs_no_gaps, CHOSEN_VIDEO.list_of_fragments)
        CHOSEN_VIDEO.video.blobs_no_gaps_path = os.path.join(os.path.split(CHOSEN_VIDEO.video.blobs_path)[0], 'blobs_collection_no_gaps.npy')
        CHOSEN_VIDEO.list_of_blobs_no_gaps.save(CHOSEN_VIDEO.video, path_to_save = CHOSEN_VIDEO.video.blobs_no_gaps_path, number_of_chunks = CHOSEN_VIDEO.video.number_of_frames)
        CHOSEN_VIDEO.video._has_crossings_solved = True
        CHOSEN_VIDEO.video.save()
        self.interpolate_crossings_popup.dismiss()
        self.trajectories_wo_gaps_popup.open()

    def create_trajectories_wo_gaps(self, *args):
        CHOSEN_VIDEO.video.create_trajectories_wo_gaps_folder()
        Logger.info("Generating trajectories. The trajectories files are stored in %s" %CHOSEN_VIDEO.video.trajectories_wo_gaps_folder)
        trajectories_wo_gaps_file = os.path.join(CHOSEN_VIDEO.video.trajectories_wo_gaps_folder, 'trajectories_wo_gaps.npy')
        trajectories_wo_gaps = produce_output_dict(CHOSEN_VIDEO.list_of_blobs_no_gaps.blobs_in_video, CHOSEN_VIDEO.video)
        np.save(trajectories_wo_gaps_file, trajectories_wo_gaps)
        Logger.info("Saving trajectories")
        CHOSEN_VIDEO.video._has_trajectories_wo_gaps = True
        CHOSEN_VIDEO.video.save()
        self.trajectories_wo_gaps_popup.dismiss()

    def update_and_show_happy_ending_popup(self, *args):
        if not hasattr(CHOSEN_VIDEO.video, 'overall_P2'):
            CHOSEN_VIDEO.video.compute_overall_P2(CHOSEN_VIDEO.list_of_fragments.fragments)
        self.create_happy_ending_popup(CHOSEN_VIDEO.video.overall_P2)
        CHOSEN_VIDEO.video.save()
        self.this_is_the_end_popup.open()
        DEACTIVATE_VALIDATION.setter(False)

    def restore_video_attributes(self):
        list_of_attributes = ['accumulation_folder',
                    'second_accumulation_finished',
                    'number_of_accumulated_global_fragments',
                    'number_of_non_certain_global_fragments',
                    'number_of_randomly_assigned_global_fragments',
                    'number_of_nonconsistent_global_fragments',
                    'number_of_nonunique_global_fragments',
                    'number_of_acceptable_global_fragments',
                    'validation_accuracy', 'validation_individual_accuracies',
                    'training_accuracy', 'training_individual_accuracies',
                    'percentage_of_accumulated_images', 'accumulation_trial',
                    'ratio_accumulated_images', 'first_accumulation_finished',
                    'identity_transfer', 'accumulation_statistics',
                    'first_frame_first_global_fragment', 'pretraining_folder',
                    'has_been_pretrained', 'has_been_assigned',
                    'has_crossings_solved','has_trajectories',
                    'has_trajectories_wo_gaps']
        is_property = [True, True, False, False, False, False, False, False,
                        False, False, False, False, True, False, True, True,
                        True, False, True, True, True, True, True, True, True]
        CHOSEN_VIDEO.video.copy_attributes_between_two_video_objects(CHOSEN_VIDEO.old_video, list_of_attributes, is_property = is_property)

    def restore_first_accumulation(self):
        self.restore_video_attributes()
        CHOSEN_VIDEO.video._ratio_accumulated_images = CHOSEN_VIDEO.video.percentage_of_accumulated_images[0]
        self.accumulation_network_params.restore_folder = CHOSEN_VIDEO.video._accumulation_folder
        self.accumulation_manager = AccumulationManager(CHOSEN_VIDEO.video, CHOSEN_VIDEO.list_of_fragments,
                                                    CHOSEN_VIDEO.list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        self.net = ConvNetwork(self.accumulation_network_params)
        self.net.restore()
        Logger.info("Saving video")
        CHOSEN_VIDEO.video.save()
        CHOSEN_VIDEO.list_of_fragments.save_light_list(CHOSEN_VIDEO.video._accumulation_folder)

    def restore_pretraining(self):
        Logger.info("Restoring pretrained network")
        self.restore_video_attributes()
        self.pretrain_network_params.restore_folder = CHOSEN_VIDEO.video.pretraining_folder
        self.net = ConvNetwork(self.pretrain_network_params)
        self.net.restore()
        self.accumulation_manager = AccumulationManager(CHOSEN_VIDEO.video, CHOSEN_VIDEO.list_of_fragments,
                                                    CHOSEN_VIDEO.list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        CHOSEN_VIDEO.video.save()

    def restore_second_accumulation(self):
        self.restore_video_attributes()
        Logger.info("Restoring trained network")
        self.accumulation_network_params.restore_folder = CHOSEN_VIDEO.video._accumulation_folder
        CHOSEN_VIDEO.list_of_fragments.load_light_list(CHOSEN_VIDEO.video._accumulation_folder)
        self.net = ConvNetwork(self.accumulation_network_params)
        self.net.restore()
        CHOSEN_VIDEO.video.save()

    def restore_identification(self):
        self.restore_video_attributes()
        CHOSEN_VIDEO.list_of_fragments.load_light_list(CHOSEN_VIDEO.video._accumulation_folder)
        CHOSEN_VIDEO.video.save()

    def restore_trajectories(self):
        self.restore_video_attributes()
        CHOSEN_VIDEO.video.save()

    def restore_crossings_solved(self):
        self.restore_video_attributes()
        CHOSEN_VIDEO.video.copy_attributes_between_two_video_objects(CHOSEN_VIDEO.old_video, ['blobs_no_gaps_path'], [False])
        CHOSEN_VIDEO.list_of_blobs_no_gaps = ListOfBlobs.load(CHOSEN_VIDEO.video, CHOSEN_VIDEO.video.blobs_no_gaps_path)
        CHOSEN_VIDEO.video.save()

    def restore_trajectories_wo_gaps(self):
        self.restore_video_attributes()
        CHOSEN_VIDEO.video.save()

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
        self.advanced_controls_button = Button(text = "Advanced idCNN\ncontrols")
        self.control_panel.add_widget(self.advanced_controls_button)
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
        if CHOSEN_VIDEO.video.number_of_channels > 3:
            self.color_tracking_label = CustomLabel(font_size = 16,
                                                    text = "Enable color-tracking")
            self.color_tracking_switch = Switch()
            self.control_panel.add_widget(self.color_tracking_label)
            self.control_panel.add_widget(self.color_tracking_switch)
            self.color_tracking_switch.active = False
        self.advanced_controls_button.bind(on_press = self.show_advanced_controls)

    def show_advanced_controls(self, *args):
        self.advanced_controls_popup.open()

    def create_advanced_controls_popup(self):
        self.container = BoxLayout(orientation = "vertical")
        self.parameters_grid = GridLayout(cols = 2)
        self.disclaimer_box = BoxLayout(size_hint = (1.,.2))
        self.disclaimer = CustomLabel(font_size = 14, text = "Modify the idCNN parameters only if you fully understand the feature that you are changing. After modifying each parameter press return.")
        self.disclaimer_box.add_widget(self.disclaimer)
        self.container.add_widget(self.disclaimer_box)
        self.container.add_widget(self.parameters_grid)
        self.mod_cnn_model_label = CustomLabel(font_size = 14, text = "CNN model: ")
        self.mod_cnn_model_text_input = TextInput(text = self.str_model, multiline=False)
        self.mod_learning_rate_label = CustomLabel(font_size = 14, text = "Learning rate")
        self.mod_learning_rate_text_input = TextInput(text = self.str_lr, multiline=False)
        self.mod_keep_prob_label = CustomLabel(font_size = 14, text = "Dropout ratio. If 1.0, no dropout is performed (for fully connected layers excluding softmax): ")
        self.mod_keep_prob_text_input = TextInput(text = self.str_kp, multiline=False)
        self.mod_batch_size_label = CustomLabel(font_size = 14, text = "Batch size (at current state it does nothing!!!!):")
        self.mod_batch_size_text_input = TextInput(text = self.str_batch_size, multiline=False)
        self.mod_optimiser_label = CustomLabel(font_size = 14, text = "Optimiser. Acceptable optimisers: SGD and Adam ")
        self.mod_optimiser_text_input = TextInput(text = self.str_optimiser, multiline=False)
        self.mod_scopes_layers_to_optimize_label = CustomLabel(font_size = 14, text = "Layers to train. Either all or fully")
        self.mod_scopes_layers_to_optimize_text_input = TextInput(text = self.str_layers_to_train, multiline=False)
        self.mod_save_folder_label = CustomLabel(font_size = 14, text = "Save folder [path where the model will be saved]: ")
        self.mod_save_folder_text_input = TextInput(text = self.save_folder, multiline=False)
        self.mod_knowledge_transfer_folder_label = CustomLabel(font_size = 14, text = "Knowledge transfer folder [path to load convolutional weights from a pre-trained model]: ")
        self.mod_knowledge_transfer_folder_text_input = TextInput(text = self.knowledge_transfer_folder, multiline=False)
        items_to_add = [self.mod_cnn_model_label, self.mod_cnn_model_text_input,
                        self.mod_learning_rate_label, self.mod_learning_rate_text_input,
                        self.mod_keep_prob_label, self.mod_keep_prob_text_input,
                        self.mod_batch_size_label, self.mod_batch_size_text_input,
                        self.mod_optimiser_label, self.mod_optimiser_text_input,
                        self.mod_scopes_layers_to_optimize_label,
                        self.mod_scopes_layers_to_optimize_text_input,
                        self.mod_save_folder_label,
                        self.mod_save_folder_text_input,
                        self.mod_knowledge_transfer_folder_label,
                        self.mod_knowledge_transfer_folder_text_input]
        [self.parameters_grid.add_widget(item) for item in items_to_add]
        self.advanced_controls_popup = Popup(title='Advanced idCNN controls',
            content=self.container,
            size_hint=(.8,.66))
        self.bind_network_controls()

    def bind_network_controls(self):
        self.mod_cnn_model_text_input.bind(on_text_validate = self.on_enter_mod_cnn_model_text_input)
        self.mod_learning_rate_text_input.bind(on_text_validate = self.on_enter_mod_learning_rate_text_input)
        self.mod_keep_prob_text_input.bind(on_text_validate = self.on_enter_mod_keep_prob_text_input)
        self.mod_batch_size_text_input.bind(on_text_validate = self.on_enter_mod_batch_size_text_input)
        self.mod_optimiser_text_input.bind(on_text_validate = self.on_enter_mod_optimiser_text_input)
        self.mod_scopes_layers_to_optimize_text_input.bind(on_text_validate = self.on_enter_mod_scopes_layers_to_optimize_text_input)
        self.mod_save_folder_text_input.bind(on_text_validate = self.on_enter_mod_save_folder_text_input)
        self.mod_knowledge_transfer_folder_text_input.bind(on_text_validate = self.on_enter_mod_knowledge_transfer_folder_text_input)

    def on_enter_mod_cnn_model_text_input(self, *args):
        self.accumulation_network_params._cnn_model = int(self.mod_cnn_model_text_input.text)
        self.cnn_model_value.text = self.mod_cnn_model_text_input.text

    def on_enter_mod_learning_rate_text_input(self, *args):
        self.accumulation_network_params.learning_rate = float(self.mod_learning_rate_text_input.text)
        self.learning_rate_value.text = self.mod_learning_rate_text_input.text

    def on_enter_mod_keep_prob_text_input(self, *args):
        self.accumulation_network_params.keep_prob = float(self.mod_keep_prob_text_input.text)
        self.keep_prob_value.text = self.mod_keep_prob_text_input.text

    def on_enter_mod_batch_size_text_input(self, *args):
        self.batch_size = int(self.mod_batch_size_text_input.text)
        self.batch_size_value.text = sself.mod_batch_size_text_input.text

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
        self.accumulation_network_params = scopes_layers_to_optimize
        self.scopes_layers_to_optimize_value.text = self.mod_scopes_layers_to_optimize_text_input.text

    def on_enter_mod_save_folder_text_input(self, *args):
        self.accumulation_network_params._save_folder = self.mod_save_folder_text_input.text
        self.save_folder_value.text = self.mod_save_folder_text_input.text

    def on_enter_mod_knowledge_transfer_folder_text_input(self, *args):
        self.accumulation_network_params._knowledge_transfer_folder = self.mod_knowledge_transfer_folder_text_input.text
        self.knowledge_transfer_folder_value.text = self.mod_knowledge_transfer_folder_text_input.text
        print("------------ ", self.accumulation_network_params.knowledge_transfer_folder)
        if os.path.isdir(self.accumulation_network_params.knowledge_transfer_folder):
            CHOSEN_VIDEO.video._tracking_with_knowledge_transfer = True

    def network_params_to_string(self):
        self.str_model = str(self.accumulation_network_params.cnn_model)
        self.str_lr = str(self.accumulation_network_params.learning_rate)
        self.str_kp = str(self.accumulation_network_params.keep_prob)
        self.batch_size = BATCH_SIZE_IDCNN
        self.str_batch_size = str(self.batch_size)
        self.str_optimiser = "SGD" if not self.accumulation_network_params.use_adam_optimiser else "Adam"
        self.str_layers_to_train = "all" if self.accumulation_network_params.scopes_layers_to_optimize is None else str(self.accumulation_network_params.scopes_layers_to_optimize)
        self.restore_folder = self.accumulation_network_params.restore_folder if self.accumulation_network_params.restore_folder is not None else 'None'
        self.save_folder = self.accumulation_network_params.save_folder if self.accumulation_network_params.save_folder is not None else 'None'
        self.knowledge_transfer_folder = self.accumulation_network_params.knowledge_transfer_folder if self.accumulation_network_params.knowledge_transfer_folder is not None else 'None'

    def create_network_params_labels(self):
        self.cnn_model_label = CustomLabel(font_size = 14, text = "CNN model: ", halign = "left")
        self.learning_rate_label = CustomLabel(font_size = 14, text = "learning_rate: ", halign = "left")
        self.keep_prob_label = CustomLabel(font_size = 14, text = "Dropout ratio: ", halign = "left")
        self.batch_size_label = CustomLabel(font_size = 14, text = "Batch size: ", halign = "left")
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
        self.batch_size_value = CustomLabel(font_size = 14, text = self.str_batch_size, halign = "left")
        self.optimiser_value = CustomLabel(font_size = 14, text = self.str_optimiser, halign = "left")
        self.scopes_layers_to_optimize_value = CustomLabel(font_size = 14, text = self.str_layers_to_train, halign = "left")
        self.restore_folder_value = CustomLabel(font_size = 14, text = self.restore_folder, halign = "left")
        self.save_folder_value = CustomLabel(font_size = 14, text = self.save_folder, halign = "left")
        self.knowledge_transfer_folder_value = CustomLabel(font_size = 14, text = self.knowledge_transfer_folder, halign = "left")
        self.image_size_value = CustomLabel(font_size = 14, text = str(self.accumulation_network_params.image_size), halign = "left")

    def create_display_network_parameters(self):
        self.get_network_parameters()
        self.network_parameters_box = BoxLayout(orientation = "vertical")
        self.network_parameters_box_title = CustomLabel(font_size = 20, text = "idCNN parameters:", size_hint = (1.,.1), halign = "left")
        self.network_parameters_grid = GridLayout(cols = 2)
        network_parameters_labels = [self.cnn_model_label, self.cnn_model_value,
                                    self.learning_rate_label,
                                    self.learning_rate_value,
                                    self.keep_prob_label, self.keep_prob_value,
                                    self.batch_size_label,
                                    self.batch_size_value,
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
        self.fig.canvas.set_window_title('Accumulation ' + str(CHOSEN_VIDEO.video.accumulation_trial))
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
