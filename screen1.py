from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

import os
import time
import sys
import socket
import threading

class wifi_client:
    def __init__(self, **kwargs):
        #my_thread = threading.Thread(target=thread_test)
        #my_thread.start()
        self.wifi_scan_list=None
        self.prior=''
        self.indexer=[]
        self.prop_indexer=['ssid', 'psk', 'key_mgmt', 'scan_ssid']
        self.conf_file=None

        self.s = socket.socket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind(('localhost', 8008))
        self.s.listen(1)
        self.s.settimeout(5.0)

    def conf_parser(self, action=None):
        if action==None:
            self.conf_file=open('/etc/wpa_supplicant/wpa_supplicant.conf', 'r').read()
            if 'network={' not in self.conf_file:
                self.conf_file=[]
            else:
                self.conf_file=self.conf_file[self.conf_file.find('network={'):].split('network={')[1:]
                temp=[]
                for n in self.conf_file:
                    temp.append(['','','',''])
                    for i in n.split('\n'):
                        if 'ssid=' in i or 'psk=' in i or 'key_mgmt=' in i or 'scan_ssid' in i:
                            temp[len(temp)-1][self.prop_indexer.index(i[i.find('\t')+1:i.find('=')])]=i[i.find('=')+1:]
                self.conf_file=temp

    def scan_list(self):
        self.wifi_scan_list=os.popen('sudo iwlist wlan0 scan').read()
        if len(self.wifi_scan_list)==0:
            temp=os.system('sudo ifconfig wlan0 up')
            if temp==65280:
                temp=os.system('sudo rfkill unblock 0')
            os.system('sudo ifconfig wlan0 up')
            self.wifi_scan_list = os.popen('sudo iwlist wlan0 scan').read()
        if 'Cell' in self.wifi_scan_list:
            self.wifi_scan_list=self.wifi_scan_list.split('          Cell')[1:]
            temp=[]
            for n in self.wifi_scan_list:
                temp.append([])
                temp_b=n[n.find('ESSID:'):]
                temp[len(temp)-1].append(temp_b[6:temp_b.find('\n')])
                temp_b = n[n.find('Encryption key'):]
                temp[len(temp) - 1].append(temp_b[15:temp_b.find('\n')])
            self.wifi_scan_list=temp
        else:
            self.wifi_scan_list=None

    def conf_flush(self):
        template='ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\nupdate_config=1\ncountry=CL\n\n'
        for n in self.conf_file:
            temp='network={\n'
            for i in range(len(n)):
                temp=temp+'\t'+self.prop_indexer[i]+'='+n[i]+'\n'
            template=template+temp+'}\n\n'
        temp=open('/etc/wpa_supplicant/wpa_supplicant.conf', 'w')
        temp.write(template)
        temp.close()

    def reconect(self):
        os.system('wpa_cli -i wlan0 reconfigure')
        os.system('wpa_cli -i wlan0 select_network 0')

    def check_selection(self, ssid):
        if ssid not in [i[0] for i in self.conf_file]:
            if [i[1] for i in self.wifi_scan_list][[i[0] for i in self.wifi_scan_list].index(ssid)]=='on':
                return False
            else:
                self.select_network(ssid, new=True)
                return True
        else:
            self.select_network(ssid)
            return True

    def select_network(self, ssid, key=None, new=False, hidden=False):
        self.keyword = None
        self.security = None
        self.ssid = None
        self.scan_ssid = '0'
        if new:
            if hidden:
                self.scan_ssid='1'
            if key==None:
                self.ssid = ssid
            else:
                self.ssid = ssid
                self.keyword = '"'+key+'"'
                self.security = 'WPA-PSK'
            self.conf_file.append([self.ssid, self.keyword, self.security, self.scan_ssid])
            self.conf_new(prior=ssid)
            self.reconect()
        else:
            self.conf_new(prior=ssid)
            self.reconect()

    def conf_new(self, prior):
        temp=self.conf_file[[i[0] for i in self.conf_file].index(prior)]
        self.conf_file.pop([i[0] for i in self.conf_file].index(prior))
        self.conf_file=[temp]+self.conf_file
        self.conf_flush()

    def get_wifi_list(self):
        self.conf_parser()
        self.scan_list()
        return [i[0] for i in self.wifi_scan_list]

    def test_connection(self):
        test=os.popen('ip route ls').read().split('\n')
        if len(test)>1:
            for n in test:
                if 'default via ' in n:
                    ip_test=n[n.find('default via ')+12:n.find(' dev ')]
        else:
            return False
        test_ping=os.popen('ping -c 3 '+ ip_test)
        if 'unreachable' in test_ping:
            return False
        for n in test[1:]:
            if '.'.join(ip_test.split('.')[:3]) in n:
                self.ip = n[n.find('src ')+4:n.find(' metric')]
                return True
    def get_ip(self):
        return self.ip

    def wait_helper(self, lista):
        try:
            c, addr = self.s.accept()
            c.send('test'.encode('utf-8'))
            lista.append(True)
            return
        except:
            lista.append(False)
            return
    def network_cleanup(self):
        self.s.close()


class Screen1(Screen):
    def __init__(self, **kwargs):
        super(Screen1, self).__init__(**kwargs)

        self.layout = FloatLayout()
        self.add_widget(self.layout)
        self.text=Label(pos_hint={'center_x':0.5,'center_y':0.5})
        self.grid = GridLayout(cols=2, rows=1, size_hint=(0.3, 0.1), pos_hint={'center_x': 0.5, 'center_y': 0.7})
        self.layout.add_widget(self.grid)
        self.layout.add_widget(self.text)

        self.pannic_button=Button(size_hint=(0.05, 0.05), pos_hint={'x': 0, 'top': 1}, on_release=self.pannic_trigger, text='x')
        self.add_widget(self.pannic_button)
        self.pannic_text=TextInput(multiline=False, password=True, size_hint=(0.8, 0.06), pos_hint={'x': 0, 'center_y': 0.5}, on_text_validate=self.pannic_attack)

        self.ssid=''

        self.attemps=[]
        self.network = wifi_client()

        self.web_test()

    def web_test(self, dt=None):
        print('Testing conection...')
        self.text.text='Testing conection...'

        self.grid.clear_widgets()
        check=self.network.test_connection()
        if check:
            self.text.text='Pass\n'+self.network.get_ip()
            self.grid.add_widget(Button(text='Choose different Wi-Fi network', on_release=self.connect_wifi))
            self.grid.add_widget(Button(text='Listen server', on_release=self.listener))
        else:
            self.no_connection()

    def no_connection(self):
        self.text.text='No internet connection'
        self.grid.clear_widgets()
        self.grid.add_widget(Button(text='Try again', on_release=self.ask_no_connection))
        self.grid.add_widget(Button(text='Connect to Wi-Fi network', on_release=self.ask_no_connection))

    def ask_no_connection(self, dt):
        if dt.text=='Try again':

            self.web_test()
        else:

            self.text.text = ' '
            self.connect_wifi()
    def connect_wifi(self, dt=None):
        self.wifi_list=self.network.get_wifi_list()
        if len(self.wifi_list)>0:
            self.show_wifi()
        else:
            self.no_connection()

    def show_wifi(self):
        self.page_scroll = ScrollView(do_scroll_x=False, size_hint=(1, None), size=(Window.width, Window.height))
        self.wifi_grid = GridLayout(cols=1, spacing=5, size_hint_y=None)
        self.wifi_grid.bind(minimum_height=self.wifi_grid.setter('height'))

        self.grid.clear_widgets()
        self.text.text=''
        self.page_scroll.clear_widgets()
        self.wifi_grid.clear_widgets()

        self.page_scroll.add_widget(self.wifi_grid)
        self.add_widget(self.page_scroll)

        for n in range(len(self.wifi_list)):
            n = self.wifi_list[n]
            self.wifi_grid.add_widget(Button(text=n, size_hint_y=None, height=250, on_release=self.wifi_choose))


    def wifi_choose(self, dt):
        self.ssid=dt.text
        check=self.network.check_selection(ssid=dt.text)
        if check:
            self.web_test()
        else:
            self.password=TextInput(multiline=False, size_hint=(0.8, 0.06), pos_hint={'center_x': 0.5, 'center_y': 0.5}, on_text_validate=self.wifi_psk)
            self.layout.add_widget(self.password)
        self.clear_widgets([self.page_scroll])
    def wifi_psk(self, dt):
        self.layout.clear_widgets([self.password])
        self.network.select_network(ssid=self.ssid, key=dt.text, new=True)
        self.web_test()
    def pannic_trigger(self, a):
        self.add_widget(self.pannic_text)
    def pannic_attack(self, dt):
        if dt.text=='agro1234':
            chao
        else:
            self.pannic_text.text=''
            self.clear_widgets([self.pannic_text])
    def listener(self, dt):
        self.text.text = 'Listening...'
        temp=[]
        t1 = threading.Thread(target=self.network.wait_helper, args=(temp,))
        t1.start()
        t1.join()
        if temp[0]:
            self.text.text='Connection reached'
            self.grid.clear_widgets()

            self.grid.add_widget(Button(text='Change Wi-Fi network', on_release=self.ask_no_connection))
            self.grid.add_widget(Button(text='Continue', on_release=self.change_page))
        else:
            self.text.text = 'Connection not reached'
            self.grid.clear_widgets()
            self.grid.add_widget(Button(text='Change Wi-Fi network', on_release=self.ask_no_connection))
            self.grid.add_widget(Button(text='Try again', on_release=self.listener))


    def change_page(self,dt):
        self.network.network_cleanup()
        self.manager.current = 'Camera'