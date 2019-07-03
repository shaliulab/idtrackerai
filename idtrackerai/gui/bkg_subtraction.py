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
from idtrackerai.gui.kivy_utils import HelpButton, CustomLabel, Chosen_Video, Deactivate_Process
from idtrackerai.utils.segmentation_utils import cumpute_background
from idtrackerai.gui.visualise_video import VisualiseVideo
global CHOSEN_VIDEO
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

    def subtract_bkg(self, *args):
        if CHOSEN_VIDEO.old_video.original_bkg is not None:
            self.original_bkg = CHOSEN_VIDEO.old_video.original_bkg
        elif CHOSEN_VIDEO.video.original_bkg is not None:
            self.original_bkg = CHOSEN_VIDEO.video.original_bkg
        else:
            self.computing_popup.open()

    def save_bkg(self, *args):
        CHOSEN_VIDEO.video.save()
        self.saving_popup.dismiss()
        self.visualiser.visualise(self.visualiser.video_slider.value, func = self.shower)

    def compute_bkg(self, *args):
        self.bkg = cumpute_background(CHOSEN_VIDEO.video)
        CHOSEN_VIDEO.video._original_bkg = self.bkg
        self.update_bkg_and_ROI_in_CHOSEN_VIDEO()
        self.computing_popup.dismiss()
        self.saving_popup.open()

    def update_bkg_and_ROI_in_CHOSEN_VIDEO(self):
        CHOSEN_VIDEO.video.resolution_reduction = CHOSEN_VIDEO.video.resolution_reduction
