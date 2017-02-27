import kivy

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.label import Label
from kivy.properties import StringProperty
from kivy.event import EventDispatcher
import os
import cPickle as pickle

class SelectFile(BoxLayout):

    def __init__(self,**kwargs):
        super(SelectFile,self).__init__(**kwargs)
        self.update = ''

    def open(self, path, filename):
        if filename:
            if '.avi' not in filename[0]:
                return True
            else:
                print 'you selected something'
                # print self.ids.inside.options.choices
                # global pathToVideo
                self.update = filename[0]


                return False

        else:
            print('Select a file before opening it!')
            return True


class tabs(TabbedPanel):

    pathToVideo = StringProperty("You did not select a video yet")

    def switch(self,id):
        self.switch_to(id)
    def assignPath(self, string):
        self.pathToVideo = string

    pass


class idApp(App):
    def build(self):
        return tabs()

if __name__ == '__main__':
    idApp().run()
