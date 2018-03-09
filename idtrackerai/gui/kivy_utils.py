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
# (2018). idtracker.ai: Tracking all individuals with correct identities in large
# animal collectives (submitted)

from __future__ import absolute_import, division, print_function
import os
import kivy
from kivy.properties import StringProperty
from kivy.properties import BooleanProperty, NumericProperty
from kivy.event import EventDispatcher
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import *
import sys
from idtrackerai.video import Video

HELP_BUTTON_OFF_PATH = os.path.join(os.path.dirname(__file__), 'help_button.png')
HELP_BUTTON_ON_PATH = os.path.join(os.path.dirname(__file__), 'help_button_on.png')

class HelpButton(ButtonBehavior, Image):
    def __init__(self, **kwargs):
        super(HelpButton, self).__init__(**kwargs)
        self.source = HELP_BUTTON_OFF_PATH
        self.size_hint = (.15,.15)

    def on_press(self):
        self.source = HELP_BUTTON_ON_PATH

    def on_release(self):
        self.source = HELP_BUTTON_OFF_PATH

    def create_help_popup(self, title, text):
        self.help_popup_container = BoxLayout()
        self.help_label = Label(text=text)
        self.help_popup_container.add_widget(self.help_label)
        self.help_label.bind(width=lambda s, w:
                   s.setter('text_size')(s, (w, None)))
        self.help_label.size_hint = (1,1)
        self.help_popup = Popup(title = title,
                            content = self.help_popup_container,
                            size_hint = (.5, .5))
        self.bind(on_press = self.open_popup)

    def open_popup(self, *args):
        self.help_popup.open()

class Chosen_Video(EventDispatcher):
    chosen = StringProperty('')

    def __init__(self, processes_list = None, **kwargs):
        super(Chosen_Video,self).__init__(**kwargs)
        self.chosen = 'Default String'
        self.video = Video()
        self.processes_list = processes_list
        self.bind(chosen=self.on_modified)
        self.processes_to_restore = {}
        self.old_video = None

    def set_chosen_item(self, chosen_string):
        self.chosen = chosen_string

    def on_modified(self, instance, value):
        try:
            print('000', value)
            self.video.video_path = value
            print('111')
        except Exception,e:
            print(str(e))
            print("Choose a video to proceed")

class Deactivate_Process(EventDispatcher):
    process = BooleanProperty(True)

    def __init__(self, **kwargs):
        super(Deactivate_Process,self).__init__(**kwargs)
        self.process = True
        self.restored = ''
        self.bind(process = self.on_modified)

    def setter(self, new_value):
        self.process = new_value

    def on_modified(self, instance, value):
        print("modifying to ", value)
        return value

class CustomLabel(Label):
    def __init__(self, font_size = 16, text = '', halign = None, **kwargs):
        super(CustomLabel,self).__init__(**kwargs)
        self.text = text
        self.bind(size=lambda s, w: s.setter('text_size')(s, w))
        self.text_size = self.size
        self.size = self.texture_size
        self.font_size = font_size
        self.halign = "center" if halign is None else halign
        self.valign = "middle"
