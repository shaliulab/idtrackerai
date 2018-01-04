from __future__ import absolute_import, division, print_function
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
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing')
sys.path.append('../groundtruth_utils')

from video import Video


class HelpButton(ButtonBehavior, Image):
    def __init__(self, **kwargs):
        super(HelpButton, self).__init__(**kwargs)
        self.source = './help_button.png'
        self.size_hint = (.15,.15)

    def on_press(self):
        self.source = './help_button_on.png'

    def on_release(self):
        self.source = './help_button.png'

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
        self.processes_to_restore = None

    def set_chosen_item(self, chosen_string):
        self.chosen = chosen_string

    def on_modified(self, instance, value):
        try:
            self.video.video_path = value
        except Exception,e:
            print(str(e))
            print("Choose a video to proceed")


class Deactivate_Process(EventDispatcher):
    process = BooleanProperty(True)

    def __init__(self, **kwargs):
        super(Deactivate_Process,self).__init__(**kwargs)
        self.process = True
        self.bind(process = self.on_modified)

    def setter(self, new_value):
        self.process = new_value

    def on_modified(self, instance, value):
        print("modifying validation to ", value)
        return value


class CustomLabel(Label):
    def __init__(self, font_size = 16, text = '', **kwargs):
        super(CustomLabel,self).__init__(**kwargs)
        self.text = text
        self.bind(size=lambda s, w: s.setter('text_size')(s, w))
        self.text_size = self.size
        self.size = self.texture_size
        self.font_size = font_size
        self.halign = "center"
        self.valign = "middle"
