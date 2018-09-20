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
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image
from idtrackerai.gui.kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process
import os
import sys
from idtrackerai.video import Video
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import ListOfFragments
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments
from idtrackerai.utils.py_utils import  getExistentFiles

class SelectFile(BoxLayout):

    def __init__(self,
                chosen_video = None,
                deactivate_roi = None,
                deactivate_preprocessing = None,
                deactivate_tracking = None,
                deactivate_validation = None,
                go_to_bind = None,
                **kwargs):
        super(SelectFile,self).__init__(**kwargs)
        global DEACTIVATE_ROI, DEACTIVATE_PREPROCESSING, DEACTIVATE_TRACKING, DEACTIVATE_VALIDATION, CHOSEN_VIDEO
        CHOSEN_VIDEO = chosen_video
        DEACTIVATE_ROI = deactivate_roi
        DEACTIVATE_PREPROCESSING = deactivate_preprocessing
        DEACTIVATE_TRACKING = deactivate_tracking
        DEACTIVATE_VALIDATION = deactivate_validation
        DEACTIVATE_VALIDATION.bind(process = self.activate_process)
        self.go_to_bind = go_to_bind
        self.update ='You did not select a video yet'
        self.main_layout = BoxLayout()
        self.main_layout.orientation = "vertical"
        self.logo = Image(source = os.path.join(os.path.dirname(__file__), 'logo.png'))
        self.main_layout.add_widget(self.logo)
        self.welcome_label = CustomLabel(font_size = 20, text = "Welcome to idtracker.ai")
        self.main_layout.add_widget(self.welcome_label)
        self.general_info_label = CustomLabel(text = "Press on the help icon to get assistance" +
                                            " at any point during the tracking process. " +
                                            "\nFor the documentation visit our website at \nhttp://idtracker.ai " +
                                            "\n Please, report any issues at \nhttps://gitlab.com/polavieja_lab/idtrackerai .")
        self.main_layout.add_widget(self.general_info_label)
        self.add_widget(self.main_layout)
        self.video = None
        self.old_video = None
        self.help_button_welcome = HelpButton()
        self.main_layout.add_widget(self.help_button_welcome)
        self.help_button_welcome.create_help_popup("Getting started",\
                                                "Use the menu on the right to browse and select a video file. "+
                                                "\n\nTo go to the previous parent directory press the symbol '../' " +
                                                ".\n\nClick on the main window to close the popup.")
        self.help_button_welcome.size_hint = (1,.4)
        self.filechooser = FileChooserListView(path = os.getcwd(), size_hint = (1., 1.))
        self.filechooser.bind(selection = self.open)
        self.add_widget(self.filechooser)

    def on_enter_session_folder(self,value):
        new_name_session_folder = self.session_name_input.text
        CHOSEN_VIDEO.video.create_session_folder(name = new_name_session_folder)
        self.welcome_popup.dismiss()
        print("******** previous_session_folder: ", CHOSEN_VIDEO.video.previous_session_folder)
        if CHOSEN_VIDEO.video.previous_session_folder != '':
            CHOSEN_VIDEO.existent_files, CHOSEN_VIDEO.old_video = getExistentFiles(CHOSEN_VIDEO.video, CHOSEN_VIDEO.processes_list)
            if CHOSEN_VIDEO.old_video is not None and hasattr(CHOSEN_VIDEO.old_video, 'first_frame_first_global_fragment'):
                if isinstance(CHOSEN_VIDEO.old_video.first_frame_first_global_fragment, int):
                    first_frame_first_global_fragment = [0 if i < CHOSEN_VIDEO.old_video.accumulation_trial - 1
                                                            else CHOSEN_VIDEO.old_video.first_frame_first_global_fragment
                                                            for i in range(CHOSEN_VIDEO.old_video.accumulation_trial + 1)]
                    CHOSEN_VIDEO.old_video._first_frame_first_global_fragment = first_frame_first_global_fragment

            self.create_restore_popup()
            self.restore_popup.open()
        else:
            self.init_chosen_video_parameters()
            DEACTIVATE_ROI.setter(False)
            DEACTIVATE_PREPROCESSING.setter(False)
            self.go_to_bind()
            self.show_restoring_popup()


    def init_chosen_video_parameters(self):
        CHOSEN_VIDEO.video._apply_ROI = False
        CHOSEN_VIDEO.video._subtract_bkg = False

    def open(self, *args):
        #try:
        CHOSEN_VIDEO.set_chosen_item(self.filechooser.selection[0])
        if CHOSEN_VIDEO.video.video_path is not None:
            self.create_welcome_popup()
            self.create_restoring_popup()
            self.session_name_input.bind(on_text_validate = self.on_enter_session_folder)
            self.welcome_popup.open()
        #except Exception,e:
        #    print(str(e))

    def create_welcome_popup(self):
        self.popup_container = BoxLayout()
        self.session_name_box = BoxLayout(orientation="vertical")
        self.session_name_label = CustomLabel(font_size = 16, text='Give a name to the current tracking session and press ENTER. Use the name of an existent session to load it.')
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
            checkbox_label = Label(id = process,
                                text = process.replace("_", " "),
                                size_hint = (.9, 1))
            process_container.add_widget(checkbox_label)
            self.restore_popup_container.add_widget(process_container)

        self.restore_button = Button(text = "Restore selected processes")
        self.restore_popup_container.add_widget(self.restore_button)
        self.restore_button.bind(on_release = self.get_processes_to_restore)
        self.restore_button.bind(on_press = self.show_restoring_popup)

    def create_restoring_popup(self):
        self.restoring_content = BoxLayout(orientation = "vertical")
        self.restoring_label = CustomLabel(text = "Wait: Loading selected processes ...")
        tabs_titles = ['ROI selection', 'Preprocessing', 'Tracking', 'Global\nvalidation', 'Individual\nvalidation']
        self.restore_btns = [Button(text = process) for process in tabs_titles ]
        self.go_to_buttons_box = BoxLayout(size_hint = (1.,1.))
        self.restoring_content.add_widget(self.restoring_label)
        self.restoring_content.add_widget(self.go_to_buttons_box)
        self.restoring_popup = Popup(title = "Restoring",
                                    content = self.restoring_content,
                                    size_hint = (.66,.3))

    def show_restoring_popup(self, *args):
        self.restoring_popup.open()

    def get_attributes_for_restored_preprocessing(self):
        path_attributes = ['preprocessing_folder', 'blobs_path',
                            'global_fragments_path', 'fragments_path',
                            'apply_ROI','subtract_bkg',
                            'resolution_reduction',
                            'maximum_number_of_blobs',
                            'number_of_channels', 'median_body_length',
                            'model_area', 'identification_image_size',
                            'blobs_path_segmented',
                            'min_threshold','max_threshold',
                            'min_area','max_area',
                            'resize', 'number_of_animals',
                            'original_ROI', 'original_bkg', 'ROI',
                            'width', 'height', 'bkg',
                            'preprocessing_folder',
                            'fragment_identifier_to_index',
                            'number_of_unique_images_in_global_fragments',
                            'maximum_number_of_images_in_global_fragments']
        CHOSEN_VIDEO.video.copy_attributes_between_two_video_objects(CHOSEN_VIDEO.old_video, path_attributes)
        CHOSEN_VIDEO.video._has_preprocessing_parameters = True
        CHOSEN_VIDEO.video._has_been_segmented = True
        CHOSEN_VIDEO.video._has_been_preprocessed = True
        CHOSEN_VIDEO.video.save()
        CHOSEN_VIDEO.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video, CHOSEN_VIDEO.video.blobs_path)
        CHOSEN_VIDEO.list_of_fragments = ListOfFragments.load(CHOSEN_VIDEO.video.fragments_path)
        CHOSEN_VIDEO.list_of_global_fragments = ListOfGlobalFragments.load(CHOSEN_VIDEO.video.global_fragments_path, CHOSEN_VIDEO.list_of_fragments.fragments)

    def process_has_to_be_restored(self, process):
        return process in CHOSEN_VIDEO.processes_to_restore.keys() and CHOSEN_VIDEO.processes_to_restore[process]

    def get_processes_to_restore(self, *args):
        CHOSEN_VIDEO.processes_to_restore = {checkbox.group: checkbox.active for checkbox
                                        in self.processes_checkboxes}

        if CHOSEN_VIDEO.processes_to_restore is None or CHOSEN_VIDEO.processes_to_restore == {}:
            self.init_chosen_video_parameters()
            DEACTIVATE_ROI.setter(False)
            DEACTIVATE_PREPROCESSING.setter(False)
        elif not CHOSEN_VIDEO.processes_to_restore['preprocessing']:
            self.init_chosen_video_parameters()
            DEACTIVATE_ROI.setter(False)
            DEACTIVATE_PREPROCESSING.setter(False)
        elif self.process_has_to_be_restored('post_processing'):
            DEACTIVATE_ROI.restored = '(restored)'
            DEACTIVATE_PREPROCESSING.restored = '(restored)'
            DEACTIVATE_TRACKING.restored = '(restored)'
            DEACTIVATE_VALIDATION.setter(False)
            CHOSEN_VIDEO.video.__dict__.update(CHOSEN_VIDEO.old_video.__dict__)
        elif CHOSEN_VIDEO.processes_to_restore['preprocessing']:
            path_attributes = ['preprocessing_folder', 'blobs_path',
                                'global_fragments_path', 'fragments_path',
                                'apply_ROI','subtract_bkg',
                                'resolution_reduction',
                                'maximum_number_of_blobs',
                                'number_of_channels', 'median_body_length',
                                'model_area', 'identification_image_size',
                                'blobs_path_segmented',
                                'min_threshold','max_threshold',
                                'min_area','max_area',
                                'resize', 'number_of_animals',
                                'original_ROI', 'original_bkg', 'ROI',
                                'width', 'height', 'bkg',
                                'preprocessing_folder',
                                'fragment_identifier_to_index',
                                'number_of_unique_images_in_global_fragments',
                                'maximum_number_of_images_in_global_fragments',
                                'segmentation_time', 'crossing_detector_time',
                                'fragmentation_time','there_are_crossings',
                                'track_wo_identities']
            CHOSEN_VIDEO.video.copy_attributes_between_two_video_objects(CHOSEN_VIDEO.old_video, path_attributes)
            CHOSEN_VIDEO.video.update_paths(CHOSEN_VIDEO.video.path_to_video_object)
            CHOSEN_VIDEO.video._has_preprocessing_parameters = True
            CHOSEN_VIDEO.video._has_been_segmented = True
            CHOSEN_VIDEO.video._has_been_preprocessed = True
            CHOSEN_VIDEO.video.save()
            CHOSEN_VIDEO.list_of_blobs = ListOfBlobs.load(CHOSEN_VIDEO.video, CHOSEN_VIDEO.video.blobs_path)
            if CHOSEN_VIDEO.video.number_of_animals != 1:
                CHOSEN_VIDEO.list_of_fragments = ListOfFragments.load(CHOSEN_VIDEO.video.fragments_path)
                CHOSEN_VIDEO.list_of_global_fragments = ListOfGlobalFragments.load(CHOSEN_VIDEO.video.global_fragments_path, CHOSEN_VIDEO.list_of_fragments.fragments)
            ### Patch to recompute the number of images in global fragments
            CHOSEN_VIDEO.video._number_of_unique_images_in_global_fragments = CHOSEN_VIDEO.list_of_fragments.compute_total_number_of_images_in_global_fragments()
            DEACTIVATE_ROI.restored = '(restored)'
            DEACTIVATE_PREPROCESSING.restored = '(restored)'
            DEACTIVATE_TRACKING.setter(False)
        self.go_to_bind()
        self.restore_popup.dismiss()


    def activate_process(self, *args):
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
