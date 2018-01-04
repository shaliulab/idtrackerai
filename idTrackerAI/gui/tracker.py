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
sys.path.append('../network')
sys.path.append('../network/crossings_detector_model')
sys.path.append('../network/identification_model')
from segmentation import segmentVideo, segment
from video_utils import blobExtractor
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
from accumulator import accumulate
from network_params import NetworkParams
from trainer import train
from assigner import assigner
from id_CNN import ConvNetwork
from constants import BATCH_SIZE_IDCNN
from constants import THRESHOLD_ACCEPTABLE_ACCUMULATION, VEL_PERCENTILE

class Tracker(BoxLayout):
    def __init__(self, chosen_video = None,
                deactivate_tracking = None,
                **kwargs):
        super(Tracker, self).__init__(**kwargs)
        global CHOSEN_VIDEO, DEACTIVATE_TRACKING
        CHOSEN_VIDEO = chosen_video
        CHOSEN_VIDEO.bind(chosen=self.do)
        DEACTIVATE_TRACKING = deactivate_tracking

    def do(self):
        self.number_of_animals = CHOSEN_VIDEO.video.number_of_animals if not CHOSEN_VIDEO.video.identity_transfer\
                                                                    else CHOSEN_VIDEO.video.knowledge_transfer_info_dict['number_of_animals']
        self.accumulation_network_params = NetworkParams(self.number_of_animals,
                                    learning_rate = 0.005,
                                    keep_prob = 1.0,
                                    scopes_layers_to_optimize = None,
                                    save_folder = CHOSEN_VIDEO.video.accumulation_folder,
                                    image_size = CHOSEN_VIDEO.video.identification_image_size,
                                    video_path = CHOSEN_VIDEO.video.video_path)

        if not CHOSEN_VIDEO.processes_to_restore(['first_accumulation']):
            print("protocol 1")
            self.protocol1()
        elif CHOSEN_VIDEO.processes_to_restore(['first_accumulation']):
            print("restoring protocol 1")
            self.restore_first_accumulation()

    def get_network_parameters(self):
        self.cnn_model_label = CustomLabel(text = "CNN model: " + str(self.accumulation_network_params.cnn_model))
        self.learning_rate_label = CustomLabel(text = "learning_rate: " +  str(self.accumulation_network_params.learning_rate))
        self.keep_prob_label = CustomLabel(text = "Drop out ratio: " + str(self.accumulation_network_params.keep_prob))
        self.batch_size = CustomLabel(text = "Batch size :" + str(BATCH_SIZE_IDCNN))
        optimiser = "SGD" if not self.accumulation_network_params.use_adam_optimiser else "Adam"
        self.use_adam_optimiser_label = CustomLabel(text = "Optimiser: " + optimiser)
        layers_to_train = "all" if self.accumulation_network_params.scopes_layers_to_optimize is None else str(self.accumulation_network_params.scopes_layers_to_optimize)
        self.scopes_layers_to_optimize_label = CustomLabel(text = "Layers to train: " + layers_to_train)
        self.restore_folder_label = CustomLabel(text = "Restore Folder: " + self.accumulation_network_params.restore_folder)
        self.save_folder_label = CustomLabel(text = "Save folder : " + self.accumulation_network_params.save_folder)
        self.knowledge_transfer_folder_label = CustomLabel(text = "Knowledge transfer folder: " + self.accumulation_network_params.knowledge_transfer_folder)
        self.image_size_label = CustomLabel(text = "Image size: " + self.accumulation_network_params.image_size)
        self.number_of_channels_label = CustomLabel(text = "Number of channels: " + self.accumulation_network_params.number_of_channels)

    def create_display_network_parameters(self):
        self.get_network_parameters()
        self.network_parameters_box = BoxLayout(orientation = "vertical")
        self.accumulation_trial = 0
        self.network_parameters_box_title = CustomLabel(font_size = 20, text = "idCNN parameters. Accumulation trial " + self.accumulation_trial)
        network_parameters_labels = [self.cnn_model_label,
                                    self.learning_rate_label,
                                    self.keep_prob_label, self.batch_size,
                                    self.use_adam_optimiser_label,
                                    self.scopes_layers_to_optimize_label,
                                    self.restore_folder_label,
                                    self.save_folder_label,
                                    self.knowledge_transfer_folder_label,
                                    self.image_size_label,
                                    self.number_of_channels_label]
        [self.network_parameters_box.add_widget(label) for label in network_parameters_labels]
        self.add_widget(self.network_parameters_box)

    def protocol1(self):
        CHOSEN_VIDEO.list_of_fragments.reset(roll_back_to = 'fragmentation')
        CHOSEN_VIDEO.list_of_global_fragments.reset(roll_back_to = 'fragmentation')
        if CHOSEN_VIDEO.video.tracking_with_knowledge_transfer:
            if CHOSEN_VIDEO.video.identity_transfer:
                self.accumulation_network_params.restore_folder = CHOSEN_VIDEO.video.knowledge_transfer_model_folder
                self.accumulation_network_params.check_identity_transfer_consistency(CHOSEN_VIDEO.video.knowledge_transfer_info_dict)
            else:
                self.accumulation_network_params.knowledge_transfer_folder = CHOSEN_VIDEO.video.knowledge_transfer_model_folder
            self.accumulation_network_params.scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax']
        self.net = ConvNetwork(self.accumulation_network_params)
        if video.tracking_with_knowledge_transfer:
            self.net.restore()
        CHOSEN_VIDEO.video._first_frame_first_global_fragment.append(CHOSEN_VIDEO.list_of_global_fragments.set_first_global_fragment_for_accumulation(CHOSEN_VIDEO.video, self.net, accumulation_trial = 0))
        if CHOSEN_VIDEO.video.identity_transfer and\
            CHOSEN_VIDEO.video.number_of_animals < CHOSEN_VIDEO.video.knowledge_transfer_info_dict['number_of_animals']:
            tf.reset_default_graph()
            self.accumulation_network_params.number_of_animals = CHOSEN_VIDEO.video.number_of_animals
            self.accumulation_network_params._restore_folder = None
            self.accumulation_network_params.knowledge_transfer_folder = CHOSEN_VIDEO.video.knowledge_transfer_model_folder
            self.net = ConvNetwork(self.accumulation_network_params)
            self.net.restore()
        CHOSEN_VIDEO.list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(CHOSEN_VIDEO.video, accumulation_trial = 0)
        accumulation_manager = AccumulationManager(CHOSEN_VIDEO.video, CHOSEN_VIDEO.list_of_fragments,
                                                    CHOSEN_VIDEO.list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        global_step = 0
        CHOSEN_VIDEO.video._ratio_accumulated_images = accumulate(accumulation_manager,
                                            CHOSEN_VIDEO.video,
                                            global_step,
                                            self.net,
                                            CHOSEN_VIDEO.video.identity_transfer)
        CHOSEN_VIDEO.video._first_accumulation_finished = True
        CHOSEN_VIDEO.video._percentage_of_accumulated_images = [CHOSEN_VIDEO.video.ratio_accumulated_images]
        CHOSEN_VIDEO.video.save()
        CHOSEN_VIDEO.list_of_fragments.save(CHOSEN_VIDEO.video.fragments_path)
        CHOSEN_VIDEO.list_of_global_fragments.save(CHOSEN_VIDEO.video.global_fragments_path, CHOSEN_VIDEO.list_of_fragments.fragments)
        CHOSEN_VIDEO.list_of_fragments.save_light_list(video._accumulation_folder)

    def restore_first_accumulation(self):
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
                    'first_frame_first_global_fragment']
        is_property = [True, True, False, False,
                        False, False, False, False,
                        False, False, False, False,
                        True, False, True, True,
                        True, False, True]
        CHOSEN_VIDEO.video.copy_attributes_between_two_video_objects(CHOSEN_VIDEO.old_video, list_of_attributes, is_property = is_property)
        CHOSEN_VIDEO.video._ratio_accumulated_images = CHOSEN_VIDEO.video.percentage_of_accumulated_images[0]
        self.accumulation_network_params.restore_folder = CHOSEN_VIDEO.video._accumulation_folder
        self.net = ConvNetwork(self.accumulation_network_params)
        self.net.restore()
        logger.info("Saving video")
        CHOSEN_VIDEO.video.save()
        CHOSEN_VIDEO.list_of_fragments.save_light_list(CHOSEN_VIDEO.video._accumulation_folder)
