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
from kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process

import os
import sys
sys.path.append('../')

from video import Video
from py_utils import getExistentFiles

class SelectFile(BoxLayout):

    def __init__(self,
                chosen_video = None,
                deactivate_roi = None,
                deactivate_validation = None,
                setup_logging = None,
                **kwargs):
        super(SelectFile,self).__init__(**kwargs)
        global DEACTIVATE_ROI, DEACTIVATE_VALIDATION, CHOSEN_VIDEO
        CHOSEN_VIDEO = chosen_video
        DEACTIVATE_ROI = deactivate_roi
        DEACTIVATE_VALIDATION = deactivate_validation
        DEACTIVATE_VALIDATION.bind(process = self.activate_process)
        self.setup_logging = setup_logging
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

    def on_enter_session_folder(self,value):
        new_name_session_folder = self.session_name_input.text
        CHOSEN_VIDEO.video.create_session_folder(name = new_name_session_folder)
        CHOSEN_VIDEO.logger = self.setup_logging(path_to_save_logs = CHOSEN_VIDEO.video.session_folder, video_object = CHOSEN_VIDEO.video)
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

        if CHOSEN_VIDEO.processes_to_restore is None or CHOSEN_VIDEO.processes_to_restore == {}:
            DEACTIVATE_ROI.setter(False)
        elif not CHOSEN_VIDEO.processes_to_restore['preprocessing']:
            DEACTIVATE_ROI.setter(False)
        elif CHOSEN_VIDEO.processes_to_restore['assignment'] or CHOSEN_VIDEO.processes_to_restore['correct_duplications']:
            DEACTIVATE_VALIDATION.setter(False)
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
