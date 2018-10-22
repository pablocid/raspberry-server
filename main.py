from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from screen1 import Screen1
from screen2 import Screen2
#from screen3 import Screen3
#from tools import config_set, options_load, options_save
from kivy.config import Config
import os
import sys
from operator import itemgetter

Config.set('kivy', 'keyboard_mode', 'dock')
Config.write()

class CamApp(App):
    def build(self):
        self.Sm=ScreenManager()
        self.Sc1=Screen1(name='Menu')
        self.Sm.add_widget(self.Sc1)
        self.Sc2=Screen2(name='Camera')
        self.Sm.add_widget(self.Sc2)
#        self.Sc3=Screen3(name='Sum')
#        self.Sm.add_widget(self.Sc3)
        self.Sm.current='Menu'
        return self.Sm

if __name__ == '__main__':
    CamApp().run()