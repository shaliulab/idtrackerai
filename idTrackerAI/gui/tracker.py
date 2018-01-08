from __future__ import absolute_import, division, print_function
import kivy

from kivy.core.window import Window
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
        self.control_panel = BoxLayout(orientation = "vertical", size_hint = (.26,1.))
        self.add_widget(self.control_panel)
        self.help_button_tracker = HelpButton()
        self.help_button_tracker.size_hint = (1.,1.)
        self.help_button_tracker.create_help_popup("Tracking",\
                                                "A message to help people.")

    def do(self):
        CHOSEN_VIDEO.video.accumulation_trial = 0
        delete = not CHOSEN_VIDEO.processes_to_restore['first_accumulation'] if 'first_accumulation' in CHOSEN_VIDEO.processes_to_restore.keys() else True
        CHOSEN_VIDEO.video.create_accumulation_folder(iteration_number = 0, delete = delete)
        self.number_of_animals = CHOSEN_VIDEO.video.number_of_animals if not CHOSEN_VIDEO.video.identity_transfer\
                                                                    else CHOSEN_VIDEO.video.knowledge_transfer_info_dict['number_of_animals']
        self.accumulation_network_params = NetworkParams(self.number_of_animals,
                                    learning_rate = 0.005,
                                    keep_prob = 1.0,
                                    scopes_layers_to_optimize = None,
                                    save_folder = CHOSEN_VIDEO.video.accumulation_folder,
                                    image_size = CHOSEN_VIDEO.video.identification_image_size,
                                    video_path = CHOSEN_VIDEO.video.video_path)

        self.start_tracking_button = Button(text = "Start protocol cascade")
        self.control_panel.add_widget(self.start_tracking_button)
        self.advanced_controls_button = Button(text = "Advanced idCNN\ncontrols")
        self.control_panel.add_widget(self.advanced_controls_button)
        self.generate_tensorboard_label = CustomLabel(font_size = 16,
                                                        text = "Save tensorboard summaries")
        self.generate_tensorboard_switch = Switch()
        self.control_panel.add_widget(self.generate_tensorboard_label)
        self.control_panel.add_widget(self.generate_tensorboard_switch)
        self.generate_tensorboard_switch.active = False
        if 'first_accumulation' in CHOSEN_VIDEO.processes_to_restore and CHOSEN_VIDEO.processes_to_restore['first_accumulation']:
            print("restoring protocol 1")
            self.restore_first_accumulation()
        elif 'first_accumulation' not in CHOSEN_VIDEO.processes_to_restore:
            print("protocol 1")
            self.protocol_number = 1
            self.create_display_network_parameters()
            self.create_advanced_controls_popup()
            self.advanced_controls_button.bind(on_press = self.show_advanced_controls)
            if CHOSEN_VIDEO.video.number_of_channels > 3:
                self.color_tracking_label = CustomLabel(font_size = 16,
                                                        text = "Enable color-tracking")
                self.color_tracking_switch = Switch()
                self.control_panel.add_widget(self.color_tracking_label)
                self.control_panel.add_widget(self.color_tracking_switch)
                self.color_tracking_switch.active = False
            self.start_tracking_button.bind(on_press = self.protocol1)
        self.control_panel.add_widget(self.help_button_tracker)


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
        self.advanced_controls_popup.bind(on_dismiss = self.update_network_parameters)

    def update_network_parameters(self, *args):
        self.remove_widget(self.network_parameters_box)
        self.create_display_network_parameters()

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

    def on_enter_mod_learning_rate_text_input(self, *args):
        self.accumulation_network_params.learning_rate = float(self.mod_learning_rate_text_input.text)

    def on_enter_mod_keep_prob_text_input(self, *args):
        self.accumulation_network_params.keep_prob = float(self.mod_keep_prob_text_input.text)

    def on_enter_mod_batch_size_text_input(self, *args):
        self.batch_size = int(self.mod_batch_size_text_input.text)

    def on_enter_mod_optimiser_text_input(self, *args):
        if  self.mod_optimiser_text_input.text == "SGD":
            use_adam_optimiser = False
        elif self.mod_optimiser_text_input.text.lower() == "adam":
            use_adam_optimiser = True
        self.accumulation_network_params.use_adam_optimiser = use_adam_optimiser

    def on_enter_mod_scopes_layers_to_optimize_text_input(self, *args):
        if self.mod_scopes_layers_to_optimize_text_input.text == "all":
            scopes_layers_to_optimize = None
        elif self.mod_scopes_layers_to_optimize_text_input.text == "fully":
            scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax']
        self.accumulation_network_params = scopes_layers_to_optimize

    def on_enter_mod_save_folder_text_input(self, *args):
        self.accumulation_network_params._save_folder = self.mod_save_folder_text_input.text

    def on_enter_mod_knowledge_transfer_folder_text_input(self, *args):
        self.accumulation_network_params._knowledge_transfer_folder = self.mod_knowledge_transfer_folder_text_input.text

    def get_network_parameters(self):
        self.str_model = str(self.accumulation_network_params.cnn_model)
        self.cnn_model_label = CustomLabel(font_size = 14, text = "CNN model: " + self.str_model)
        self.str_lr = str(self.accumulation_network_params.learning_rate)
        self.learning_rate_label = CustomLabel(font_size = 14, text = "learning_rate: " +  self.str_lr)
        self.str_kp = str(self.accumulation_network_params.keep_prob)
        self.keep_prob_label = CustomLabel(font_size = 14, text = "Dropout ratio: " + self.str_kp)
        self.str_batch_size = str(BATCH_SIZE_IDCNN)
        self.batch_size_label = CustomLabel(font_size = 14, text = "Batch size :" + self.str_batch_size)
        self.str_optimiser = "SGD" if not self.accumulation_network_params.use_adam_optimiser else "Adam"
        self.optimiser_label = CustomLabel(font_size = 14, text = "Optimiser: " + self.str_optimiser)
        self.str_layers_to_train = "all" if self.accumulation_network_params.scopes_layers_to_optimize is None else str(self.accumulation_network_params.scopes_layers_to_optimize)
        self.scopes_layers_to_optimize_label = CustomLabel(font_size = 14, text = "Layers to train: " + self.str_layers_to_train)
        self.restore_folder = self.accumulation_network_params.restore_folder if self.accumulation_network_params.restore_folder is not None else 'None'
        self.restore_folder_label = CustomLabel(font_size = 14, text = "Restore Folder: " + self.restore_folder)
        self.save_folder = self.accumulation_network_params.save_folder if self.accumulation_network_params.save_folder is not None else 'None'
        self.save_folder_label = CustomLabel(font_size = 14, text = "Save folder : " + self.save_folder)
        self.knowledge_transfer_folder = self.accumulation_network_params.knowledge_transfer_folder if self.accumulation_network_params.knowledge_transfer_folder is not None else 'None'
        self.knowledge_transfer_folder_label = CustomLabel(font_size = 14, text = "Knowledge transfer folder: " + self.knowledge_transfer_folder)
        self.image_size_label = CustomLabel(font_size = 14, text = "Image size: " + str(self.accumulation_network_params.image_size))

    def create_display_network_parameters(self):
        self.get_network_parameters()
        self.network_parameters_box = BoxLayout(orientation = "vertical")
        self.network_parameters_box_title = CustomLabel(font_size = 20, text = "idCNN parameters:")
        network_parameters_labels = [self.network_parameters_box_title,
                                    self.cnn_model_label,
                                    self.learning_rate_label,
                                    self.keep_prob_label, self.batch_size_label,
                                    self.optimiser_label,
                                    self.scopes_layers_to_optimize_label,
                                    self.restore_folder_label,
                                    self.save_folder_label,
                                    self.knowledge_transfer_folder_label,
                                    self.image_size_label]
        [self.network_parameters_box.add_widget(label) for label in network_parameters_labels]
        self.add_widget(self.network_parameters_box)
        self.show_info_tracking()
        self.create_tracking_figures_axes()

    def show_info_tracking(self):
        self.get_info_tracking()
        self.info_tracking_box = BoxLayout(orientation = "vertical")
        self.network_parameters_box.add_widget(self.info_tracking_box)
        self.info_tracking_box_title = CustomLabel(font_size = 20, text = "Deep fingerprint protocols cascade update:")
        self.accumulation_trial_label = CustomLabel(font_size = 14, text = "Accumulation trial " + str(CHOSEN_VIDEO.video.accumulation_trial))
        labels_to_add = [self.info_tracking_box_title,
                        self.accumulation_trial_label,
                        self.iteration_label,
                        self.number_of_images_used_for_training_label,
                        self.number_of_new_images_acquired_label]
        [self.network_parameters_box.add_widget(label) for label in labels_to_add]

    def get_info_tracking(self):
        self.current_protocol = CustomLabel(font_size = 14, text = "Currently running protocol number " + str(self.protocol_number))
        self.iteration_label = CustomLabel(font_size = 14, text = "Currently running training iteration number ")
        self.number_of_images_used_for_training_label = CustomLabel(font_size = 14, text = "Number of images used for training in current iteration: ")
        self.number_of_new_images_acquired_label = CustomLabel(font_size = 14, text = "Number of new images acquired in current iteration: ")

    def protocol1(self, *args):
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
        if CHOSEN_VIDEO.video.tracking_with_knowledge_transfer:
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
        self.global_step = 0
        CHOSEN_VIDEO.video._ratio_accumulated_images = accumulate(accumulation_manager,
                                                                    CHOSEN_VIDEO.video,
                                                                    self.global_step,
                                                                    self.net,
                                                                    CHOSEN_VIDEO.video.identity_transfer,
                                                                    GUI_axes = self.ax_arr,
                                                                    canvas_from_GUI = self.tracking_fig_canvas,
                                                                    plot_flag = True)
        CHOSEN_VIDEO.video._first_accumulation_finished = True
        CHOSEN_VIDEO.video._percentage_of_accumulated_images = [CHOSEN_VIDEO.video.ratio_accumulated_images]
        CHOSEN_VIDEO.video.save()
        CHOSEN_VIDEO.list_of_fragments.save(CHOSEN_VIDEO.video.fragments_path)
        CHOSEN_VIDEO.list_of_global_fragments.save(CHOSEN_VIDEO.video.global_fragments_path, CHOSEN_VIDEO.list_of_fragments.fragments)
        CHOSEN_VIDEO.list_of_fragments.save_light_list(CHOSEN_VIDEO.video._accumulation_folder)

    def create_tracking_figures_axes(self):
        if hasattr(self, 'fig'):
            self.remove_widget(self.tracking_main_figure)
            self.fig.clear()

        self.fig, self.ax_arr = plt.subplots(4)
        self.fig.set_facecolor((.188, .188, .188))
        [(ax.set_facecolor((.188, .188, .188)), ax.tick_params(color='white', labelcolor='white'), ax.xaxis.label.set_color('white'), ax.yaxis.label.set_color('white')) for ax in self.ax_arr]
        [spine.set_edgecolor('white') for ax in self.ax_arr for spine in ax.spines.values()]


        self.fig.canvas.set_window_title('Accumulation ' + str(CHOSEN_VIDEO.video.accumulation_trial)) #+ '-' + str(CHOSEN_VIDEO.video.accumulation_step))
        self.fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        self.tracking_main_figure = FigureCanvasKivyAgg(self.fig)
        self.tracking_fig_canvas = self.fig.canvas
        self.add_widget(self.tracking_main_figure)


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
