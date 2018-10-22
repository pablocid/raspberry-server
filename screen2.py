import datetime
import os
import sys
import time
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from tools import CvCamera, config_get, config_set, Labeler, list_editor
import threading
import socket

class Screen2(Screen):
    def __init__(self, **kwargs):
        super(Screen2, self).__init__(**kwargs)
        self.base = sys.argv[0][:sys.argv[0].rfind('/')]
        self.home = os.path.expanduser("~")
        print(self.home)
        if '/root' in self.home:
            self.home=self.home.replace('/root', '/home/pi')
        print(self.home)
        self.grid=GridLayout(cols=3, rows=1, pos_hint={'top':0.1, 'center_x':0.5}, size_hint=(0.6, 0.1))
        self.grid.add_widget(Button(text='Back', on_press=self.change_back))
        self.grid.add_widget(Button(text='Capture', on_press=self.save_all, on_release=self.cicle))
        self.grid.add_widget(Button(text='Check and Save', on_press=self.change_next))
        self.add_widget(self.grid)
        self.cameraman = CvCamera(pos_hint={'top': 1, 'center_x': 0.5}, allow_stretch=True, size_hint_y=0.9)
        self.add_widget(self.cameraman)
        self.slide_thresh=Slider(size_hint_x=0.06, pos_hint={'center_x':0.9}, min=0, max=0.01, value=0.0011, orientation='vertical', step=0.0001, on_touch_move=self.set_thresh)
        self.add_widget(self.slide_thresh)
        self.on_enter=self.in_screen

    def set_thresh(self, a, b):
        self.cameraman.set_umbral(valor=self.slide_thresh.value)

    def in_screen(self):



        #self.config=config_get()
        self.cameraman.cam_init()
        self.camerastate='running'

        self.breaker = False

        t1 = threading.Thread(target=self.server_call,)
        t1.start()

        #t1.join()


    def save_all(self, dt):
        self.cameraman.set_name_save(name_save=self.home+'/fruit_data/'+time.strftime('%Y%m%d_%H%M%S'))

    def cicle(self, dt):
        self.cameraman.set_live(on_live=False)

    def server_call(self):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('localhost', 8008))
        s.listen(1)

        while True:
            if self.breaker:
                s.close()
                break
            c, addr = s.accept()
            a = c.recv(1024).decode('utf-8')
            if a == 'capture':
                if self.cameraman.get_live:
                    self.cameraman.set_name_save(name_save=self.home + '/fruit_data/' + time.strftime('%Y%m%d_%H%M%S'))
                    self.cameraman.set_live(on_live=False)
                    c.send('capture done'.encode('utf-8'))
                else:
                    c.send('busy'.encode('utf-8'))
            else:
                c.send('duno bro'.encode('utf-8'))

    def label_copy(self):
        self.grid.size_hint=(0,0)
        self.Labeler.size_hint=(0,0)
        self.confirmation=GridLayout(cols=1, rows=3, size_hint=(1, 1))
        self.confirmation.add_widget(Label(text='This label is already used. Overwrite the photo?'))
        self.confirmation.add_widget(Button(text='Yes', on_press=self.label_copy_confirm))
        self.confirmation.add_widget(Button(text='No', on_press=self.label_copy_confirm))
        self.add_widget(self.confirmation)

    def label_copy_confirm(self, dt):
        if dt.text=='Yes':
            self.cameraman.set_live(on_live=False)
        self.grid.size_hint = (0.6, 0.1)
        self.Labeler.size_hint = (1, 0.06)
        self.clear_widgets([self.confirmation])

    def change_back(self,dt):

        self.manager.current='Menu'
        self.cameraman.cam_cancel()
        self.breaker=True


    def change_next(self,dt):

        config_set(3, '_'.join(list(map(str, self.cameraman.camera.resolution))))
        self.manager.current='Sum'
        self.cameraman.cam_cancel()