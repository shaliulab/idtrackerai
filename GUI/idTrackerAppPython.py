from __future__ import division
# third party
import pyautogui

# kivy imports
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.tabbedpanel import TabbedPanelHeader
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout


class idTrackerDeepApp(App):
    def build(self):
        #set window dimension
        self.wScreen, self.hScreen = self.getScreenSize()
        Window.size = (self.wScreen, self.hScreen)

        # Create tabbed panel
        tb_panel= TabbedPanel()
        tb_panel.do_default_tab = False
        tb_panel.tab_width = int(self.wScreen / 4)

        #Create text tab
        th_mask_head = TabbedPanelHeader(text = 'Mask and bkg subtraction')

        mask_layout = GridLayout(cols = 1)
        divLabelSwitch = FloatLayout(size_hint=(None, None))
        divSwitch = FloatLayout(size_hint=(None, None))
        mask_layout.add_widget(divLabelSwitch)
        mask_layout.add_widget(divSwitch)
        # mask_layout = BoxLayout()

        # add bkg switch
        bkg_switch = Switch(active = True)
        bkg_switch.bind(active = self.onSwitchValueChange)
        divLabelSwitch.add_widget(Label(text='Background subtraction'))
        divSwitch.add_widget(bkg_switch)

        divVideo = FloatLayout(size_hint=(None, None))
        mask_layout.add_widget(divVideo)
        divVideo.add_widget(Button(text='Hello', size_hint=(1,1)))

        # sliderFrames = self.generateTrackBar(0,100,0)
        # sliderFrames.bind(value=self.OnSliderValueChange)
        # mask_layout.add_widget(sliderFrames)
        #
        # mask_layout.add_widget(Label(text = 'Background subtraction'))
        th_mask_head.content = mask_layout


        # #Create image tab
        # th_img_head= TabbedPanelHeader(text='Preprocessing parameters')
        # th_img_head.content= Image(source='sample.jpg',pos=(400, 100), size=(400, 400))
        #
        # #Create button tab
        # th_btn_head = TabbedPanelHeader(text='Tracking')
        # th_btn_head.content= Button(text='This is my button',font_size=20)
        #
        #
        # #Create player tab
        # th_play_head = TabbedPanelHeader(text='Player')
        # th_play_head.content= Button(text='This is my button',font_size=20)
        #
        tb_panel.add_widget(th_mask_head)
        # tb_panel.add_widget(th_img_head)
        # tb_panel.add_widget(th_btn_head)
        # tb_panel.add_widget(th_play_head)

        return tb_panel

    def getScreenSize(self):
        wScreen, hScreen = pyautogui.size()
        return wScreen, hScreen

    def generateTrackBar(self, minV, maxV, value):
        self.s = Slider(min=minV, max=maxV, value=value, step=1, pos=(0,0), size=(100,200))

        return self.s

    def OnSliderValueChange(self,instance,value):
        print int(value)

    def onSwitchValueChange(self,instance, value):
        print('the switch', instance, 'is', value)






if __name__ == '__main__':
    idTrackerDeepApp().run()
