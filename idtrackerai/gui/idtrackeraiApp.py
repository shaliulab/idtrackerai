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
import os
import sys
import numpy as np
import logging.config
import cv2
from functools import partial
import kivy
import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
from kivy.app import App
from kivy.core.window import Window
from kivy.logger import Logger
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.tabbedpanel import TabbedPanelItem
from kivy.uix.tabbedpanel import TabbedPanelHeader
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.switch import Switch
from kivy.graphics import *
from kivy.config import Config
from idtrackerai.gui.kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process
from idtrackerai.gui.select_file import SelectFile
from idtrackerai.gui.preprocessing_preview import PreprocessingPreview
from idtrackerai.gui.roi_selector import ROISelector
from idtrackerai.gui.tracker import Tracker
from idtrackerai.gui.validator import Validator
from idtrackerai.gui.individual_validator import IndividualValidator
from idtrackerai.gui.visualise_video import VisualiseVideo
from idtrackerai.video import Video
from idtrackerai.utils.py_utils import getExistentFiles, get_spaced_colors_util
from idtrackerai.utils.video_utils import cumpute_background, blob_extractor
from idtrackerai.preprocessing.segmentation import segment_frame
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import ListOfFragments
from idtrackerai.groundtruth_utils.generate_groundtruth import generate_groundtruth
from idtrackerai.groundtruth_utils.compute_groundtruth_statistics import get_accuracy_wrt_groundtruth
from idtrackerai.constants import THRESHOLD_ACCEPTABLE_ACCUMULATION, RESTORE_CRITERION, VEL_PERCENTILE, PROCESSES

print('\n************************************************************************************')
print('LICENSE')
print('idtracker.ai  Copyright (C) 2017-  Francisco Romero Ferrero, Mattia G. Bergomi,')
print('Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the')
print('Champalimaud Foundation')
print('This program comes with ABSOLUTELY NO WARRANTY.')
print('This is free software, and you are welcome to redistribute it')
print('under certain conditions. Check http://idtracker.ai/ for more information.')
print('************************************************************************************\n')

"""
Init variables
    PROCESSES: list of strings.
        list of all the processes that can be saved and loaded while
        tracking a video
"""
# PROCESSES = ['preprocessing','protocols1_and_2', 'protocol3_pretraining',
#             'protocol3_accumulation', 'residual_identification',
#             'post_processing']

class Root(TabbedPanel):
    global DEACTIVATE_ROI, DEACTIVATE_PREPROCESSING, DEACTIVATE_TRACKING, DEACTIVATE_VALIDATION, CHOSEN_VIDEO
    DEACTIVATE_ROI = Deactivate_Process()
    DEACTIVATE_PREPROCESSING = Deactivate_Process()
    DEACTIVATE_TRACKING = Deactivate_Process()
    DEACTIVATE_VALIDATION = Deactivate_Process()
    CHOSEN_VIDEO = Chosen_Video(processes_list = PROCESSES)

    def __init__(self, **kwargs):
        super(Root, self).__init__(**kwargs)
        self.bind(current_tab = self.content_changed_cb)
        self.add_welcome_tab()
        self.add_ROI_selection_tab()
        self.add_preprocessing_tab()
        self.add_tracking_tab()
        self.add_validation_tab()
        self.add_individual_validator_tab()
        DEACTIVATE_ROI.bind(process = self.manage_ROI_selection)
        DEACTIVATE_PREPROCESSING.bind(process = self.manage_preprocessing)
        DEACTIVATE_TRACKING.bind(process = self.manage_tracking)
        DEACTIVATE_VALIDATION.bind(process = self.manage_validation)
        DEACTIVATE_VALIDATION.bind(process = self.manage_individual_validation)

    def add_welcome_tab(self):
        self.welcome_tab = TabbedPanelItem(text = "Welcome")
        self.select_file = SelectFile(chosen_video = CHOSEN_VIDEO,
                                    deactivate_roi = DEACTIVATE_ROI,
                                    deactivate_preprocessing = DEACTIVATE_PREPROCESSING,
                                    deactivate_tracking = DEACTIVATE_TRACKING,
                                    deactivate_validation = DEACTIVATE_VALIDATION,
                                    go_to_bind = self.welcome_go_to_bind)
        self.welcome_tab.add_widget(self.select_file)
        self.add_widget(self.welcome_tab)

    def welcome_go_to_bind(self):
        self.select_file.restoring_label.text = "Click on the active button to proceed"
        activators = [DEACTIVATE_ROI, DEACTIVATE_PREPROCESSING,
                    DEACTIVATE_TRACKING, DEACTIVATE_VALIDATION,
                    DEACTIVATE_VALIDATION]
        [(self.select_file.go_to_buttons_box.add_widget(btn),
            setattr(btn, 'disabled', activators[i].process),
            setattr(btn, 'text', btn.text + '\n' + str(activators[i].restored)),
            btn.bind(on_release = partial(self.switch, self.tab_list[-(i + 2) ])))
            for i, btn in enumerate(self.select_file.restore_btns)]

    def add_ROI_selection_tab(self):
        self.ROI_selection_tab = TabbedPanelItem(text = 'ROI selection')
        self.ROI_selection_tab.id = "ROI selection"
        self.ROI_selection_tab.disabled = True
        self.add_widget(self.ROI_selection_tab)

    def manage_ROI_selection(self, *args):
        print("from root ROI: ", DEACTIVATE_ROI.process)
        self.ROI_selection_tab.disabled = DEACTIVATE_ROI.process
        if not DEACTIVATE_ROI.process:
            self.roi_selector = ROISelector(chosen_video = CHOSEN_VIDEO,
                                        deactivate_roi = DEACTIVATE_ROI)
            self.roi_selector.id = "roi_selector"
            self.ROI_selection_tab.add_widget(self.roi_selector)
        else:
            if hasattr(self, 'roi_selector'):
                self.ROI_selection_tab.clean(self.roi_selector)

    def add_preprocessing_tab(self):
        self.preprocessing_tab = TabbedPanelItem(text = 'Preprocessing')
        self.preprocessing_tab.id = "Preprocessing"
        self.preprocessing_tab.disabled = True
        self.add_widget(self.preprocessing_tab)

    def manage_preprocessing(self, *args):
        print("from root preprocessing: ", DEACTIVATE_PREPROCESSING.process)
        self.preprocessing_tab.disabled = DEACTIVATE_PREPROCESSING.process
        if not DEACTIVATE_PREPROCESSING.process:
            self.preprocessor = PreprocessingPreview(chosen_video = CHOSEN_VIDEO,
                                        deactivate_preprocessing = DEACTIVATE_PREPROCESSING,
                                        deactivate_tracking = DEACTIVATE_TRACKING)
            self.preprocessor.id = "preprocessor"
            self.preprocessing_tab.add_widget(self.preprocessor)
            if hasattr(self, 'roi_selector'):
                self.roi_selector.btn_save_roi.bind(on_press = self.preprocessor.activate_ROI_switch)
                self.roi_selector.btn_no_roi.bind(on_press = self.preprocessor.deactivate_ROI_switch)
        else:
            if hasattr(self, 'preprocessor'):
                self.preprocessing_tab.clean(self.preprocessor)

    def add_tracking_tab(self):
        self.tracking_tab = TabbedPanelItem(text = 'Tracking')
        self.tracking_tab.id = "Tracking"
        self.tracking_tab.disabled = True
        self.add_widget(self.tracking_tab)

    def manage_tracking(self, *args):
        print("from root tracker: ", DEACTIVATE_TRACKING.process)
        self.tracking_tab.disabled = DEACTIVATE_TRACKING.process
        if not DEACTIVATE_TRACKING.process:
            self.tracker = Tracker(chosen_video = CHOSEN_VIDEO,
                                deactivate_tracking = DEACTIVATE_TRACKING,
                                deactivate_validation = DEACTIVATE_VALIDATION)
            self.tracker.id = "tracker"
            self.tracking_tab.add_widget(self.tracker)
            if hasattr(self, 'preprocessor'):
                self.preprocessor.go_to_tracking_button.bind(on_press = partial(self.switch, self.tab_list[2]))
                self.preprocessor.go_to_tracking_button.disabled = False
        else:
            if hasattr(self, 'tracker'):
                self.tracking_tab.clean(self.tracker)

    def add_validation_tab(self):
        self.validation_tab = TabbedPanelItem(text='Global Validation')
        self.validation_tab.id = "Global validation"
        self.validation_tab.disabled = True
        self.add_widget(self.validation_tab)

    def manage_validation(self, *args):
        print("from root global validation: ", DEACTIVATE_VALIDATION.process)
        self.validation_tab.disabled = DEACTIVATE_VALIDATION.process
        if not DEACTIVATE_VALIDATION.process:
            self.validator = Validator(chosen_video = CHOSEN_VIDEO,
                                        deactivate_validation = DEACTIVATE_VALIDATION)
            if hasattr(self, 'tracker'):
                self.tracker.go_to_validation_button.bind(on_release = partial(self.switch, self.tab_list[1]))
            self.validator.id = "validator"
            self.validation_tab.add_widget(self.validator)
        else:
            if hasattr(self, 'validator'):
                self.validation_tab.clean(self.validator)

    def add_individual_validator_tab(self):
        self.individual_validation_tab = TabbedPanelItem(text='Individual Validation')
        self.individual_validation_tab.id = "Individual validation"
        self.individual_validation_tab.disabled = True
        self.add_widget(self.individual_validation_tab)

    def manage_individual_validation(self, *args):
        print("from root individual validation: ", DEACTIVATE_VALIDATION.process)
        self.individual_validation_tab.disabled = DEACTIVATE_VALIDATION.process
        if not DEACTIVATE_VALIDATION.process:
            self.individual_validator = IndividualValidator(chosen_video = CHOSEN_VIDEO,
                                        deactivate_validation = DEACTIVATE_VALIDATION)
            self.individual_validator.id = "individual_validator"
            self.individual_validation_tab.add_widget(self.individual_validator)
        else:
            if hasattr(self, 'individual_validator'):
                self.individual_validation_tab.clean(self.individual_validator)

    def content_changed_cb(self, obj, value):
        print('VALUE', value.__dict__)
        print('CONTENT', value.content)
        print("OBJECT", obj)
        ### FIXME: disabled feature not working well and had to add a condition
        # Added this condition after switching to kivy 1.10.1 because the disabled"
        # feature is working differently than for kivy 1.9.
        print("ID", value.content.id)
        if value.content is not None:
            if value.content.id == 'roi_selector':
                if not self.roi_selector.has_been_executed:
                    self.roi_selector.do()
            if value.content.id == 'preprocessor':
                if not self.preprocessor.has_been_executed:
                    self.preprocessor.do()
            if value.content.id == "tracker":
                self.tracker.do()
            if value.content.id == "validator":
                self.validator.do()
            if value.content.id == "individual_validator":
                self.individual_validator.do()

    def switch(self, tab, *args):
        print("0000000000000000 ", hasattr(self, 'tracker') and hasattr(self.tracker, 'this_is_the_end_popup'))
        if hasattr(self, 'tracker') and hasattr(self.tracker, 'this_is_the_end_popup'):
            self.tracker.this_is_the_end_popup.dismiss()
        elif hasattr(self, 'select_file') and hasattr(self.select_file, 'restoring_popup'):
            self.select_file.restoring_popup.dismiss()
        self.switch_to(tab)


class MainWindow(BoxLayout):
    pass

class idtrackeraiApp(App):
    Config.set('kivy', 'keyboard_mode', '')
    Config.set('graphics', 'fullscreen', '0')
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    Config.set('kivy', 'log_level', 'debug')
    Config.write()
    global CHOSEN_VIDEO

    def build(self):
        return MainWindow()

    def create_video_logs_folder(self):
        if hasattr(CHOSEN_VIDEO.video, 'session_folder')\
            and os.path.exists(CHOSEN_VIDEO.video.session_folder)\
            and CHOSEN_VIDEO.video is not None:

            CHOSEN_VIDEO.video.logs_folder = os.path.join(CHOSEN_VIDEO.video.session_folder, 'logs')
            if not os.path.isdir(CHOSEN_VIDEO.video.logs_folder):
                os.makedirs(CHOSEN_VIDEO.video.logs_folder)
            return True
        return False

    def on_stop(self):
        log_dir = Config.get('kivy', 'log_dir')
        log_name = Config.get('kivy', 'log_name')
        _dir = kivy.kivy_home_dir
        if log_dir and os.path.isabs(log_dir):
            _dir = log_dir
        else:
            _dir = os.path.join(_dir, log_dir)
        log_files = os.listdir(_dir)
        log_files.sort(key=lambda f: os.path.getmtime(os.path.join(_dir, f)))
        log_file = os.path.join(_dir, log_files[-1])
        move_logs = self.create_video_logs_folder()
        if move_logs:
            os.rename(os.path.join(_dir, log_file), os.path.join(CHOSEN_VIDEO.video.logs_folder, os.path.split(log_file)[-1]))

def get_error_and_save_logs():
    import traceback
    ex_type, ex, tb = sys.exc_info()
    traceback.print_exception(ex_type, ex, tb)
    idtrackeraiApp().on_stop()
    print("\n")
    print("idtracker.ai quit unexpectedly.")
    print("Check the log files in")
    try:
        print(CHOSEN_VIDEO.video.logs_folder)
        print("and attach them to the issue.")
    except:
        print(os.path.join(kivy.kivy_home_dir, Config.get('kivy', 'log_dir')))
        print("and attach them to the issue.")
    print("If this error persists please open an issue at")
    print("https://gitlab.com/polavieja_lab/idtrackerai")

def run_app():
    try:
        idtrackeraiApp().run()
    except:
        get_error_and_save_logs()

if __name__ == '__main__':
    run_app()
