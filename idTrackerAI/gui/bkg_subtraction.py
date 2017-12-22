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
from visualise_video import VisualiseVideo
from kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process
import sys
sys.path.append('../')
sys.path.append('../utils')

from video_utils import computeBkg

class BkgSubtraction(BoxLayout):
    def __init__(self, chosen_video = None, **kwargs):
        super(BkgSubtraction, self).__init__(**kwargs)
        global CHOSEN_VIDEO
        CHOSEN_VIDEO = chosen_video
        self.bkg = None
        #set useful popups
        #saving:
        self.saving_popup = Popup(title='Saving',
            content=CustomLabel(text='wait ...'),
            size_hint=(.3,.3))
        self.saving_popup.bind(on_open=self.save_bkg)
        #computing:
        self.computing_popup = Popup(title='Computing',
            content=CustomLabel(text='wait ...'),
            size_hint=(.3,.3))
        self.computing_popup.bind(on_open=self.compute_bkg)
        global CHOSEN_VIDEO

    def subtract_bkg(self, *args):
        if CHOSEN_VIDEO.old_video.bkg is not None:
            self.bkg = CHOSEN_VIDEO.old_video.bkg
        elif CHOSEN_VIDEO.video.bkg is not None:
            self.bkg = CHOSEN_VIDEO.video.bkg
        else:
            self.computing_popup.open()

    def save_bkg(self, *args):
        self.computing_popup.dismiss()
        CHOSEN_VIDEO.video.save()
        self.saving_popup.dismiss()


    def compute_bkg(self, *args):
        self.bkg = computeBkg(CHOSEN_VIDEO.video)
        CHOSEN_VIDEO.video._bkg = self.bkg
