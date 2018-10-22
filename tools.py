import os
import datetime
import numpy as np
import cv2
import math
import sys
import time
#from correction_4 import perspective_check, virtual_area
from ar_markers import detect_markers
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from picamera.array import PiRGBArray
from picamera import PiCamera

def options_load(dir):
    if dir[::-1][0]!='/':
        dir=dir+'/'
    temp = []
    for file in os.listdir(dir):
        temp.append([file[:file.rfind('.')]])
        file = open(dir + file, 'r')
        new = file.read()
        file.close()
        new = new.split('>')
        for s in new:
            if len(s)==0:
                continue
            temp[len(temp) - 1].append([])
            s=[i.split('\t') for i in s.split('\n') if len(i) != 1]
            temp[len(temp)-1][len(temp[len(temp)-1])-1].append(s[0][0])
            for i in s[1:]:
                i=[l for l in i if len(l)!=0]
                if len(i)!=0:
                    temp[len(temp) - 1][len(temp[len(temp)-1])-1].append(i)
    return temp

def options_save(dir, options_loaded):
    if dir[::-1][0]!='/':
        dir=dir+'/'
    for n in options_loaded:
        temp=[n[0]]
        file=open(dir+n[0]+'.tsv', 'w')
        file.close()
        file = open(dir+n[0] + '.tsv', 'a')
        for i in n[1:]:
            file.write('>'+i[0]+'\n'+'\n'.join(['\t'.join(s) for s in i[1:]])+'\n')
        file.close()

def contour_save(contours, new_label=None):
    new=[]
    for n in contours:
        if new_label!=None:
            new.append('\t'.join([new_label] + ['_'.join(i) for i in n[:, 0, :].astype('unicode_').tolist()]))
        else:
            new.append('\t'.join([config_get(1)]+['_'.join(i) for i in n[:,0,:].astype('unicode_').tolist()]))
    return '\n'.join(new)

def contour_load(path, fruit_dict=False):
    file=open(path, 'r')
    temp=file.read()
    file.close()
    temp=[i.split('\t') for i in temp.split('\n') if len(i)!=1]
    if fruit_dict:
        temp_d=[]
    if len(temp[0])==1:
        if fruit_dict:
            return [None, None]
        else:
            return []
    else:
        for n in range(len(temp)):
            if fruit_dict:
                temp_d.append(temp[n][0])
            temp[n]=temp[n][1:]
            temp[n]=np.array([i.split('_') for i in temp[n]]).astype('int64').reshape(len(temp[n]), 1, 2)
        if fruit_dict:
            return [temp, temp_d]
        else:
            return temp

def config_set(cat=None, to_set=None):
    base = sys.argv[0][:sys.argv[0].rfind('/')]
    if cat==None:
        if to_set=='create':
            filess=open(base+'/fruits.config', 'w')
            filess.write('\n\n\n\n')
            filess.close()
    else:
        if to_set!=None:
            filess=open(base + '/fruits.config', 'r')
            conf=filess.read().split('\n')
            filess.close()
            conf[cat]=to_set
            filess = open(base+'/fruits.config', 'w')
            filess.write('\n'.join(conf))
            filess.close()

def config_get(cat=None):
    base = sys.argv[0][:sys.argv[0].rfind('/')]
    filess = open(base+'/fruits.config', 'r')
    conf = filess.read().split('\n')
    filess.close()
    if cat!=None:
        if cat==3:
            return list(map(int,conf[3].split('_')))
        else:
            return conf[cat]
    else:
        return conf

def list_editor(lista_full_dir, load=False, add=None, delete=None):
    if load:
        file = open(lista_full_dir, 'r')
        list_whole = [n for n in file.read().split('\n')[::-1] if len(n) != 0]
        file.close()
        return list_whole
    if add!=None:
        file = open(lista_full_dir, 'a')
        file.write('\n'+add)
        file.close()
    if delete!=None:
        file = open(lista_full_dir, 'r')
        list_whole = [n for n in file.read().split('\n')[::-1] if len(n) != 0]
        file.close()
        list_whole.remove(delete)
        file = open(lista_full_dir, 'w')
        file.write('\n'.join(list_whole))
        file.close()

def file_append(list_file, list_labels):
    if len(list_labels)!=0:
        home = os.path.expanduser("~")
        file=open(home+'/fruit_data/'+list_file+'.list', 'a')
        file.write('\n'.join(list_labels)+'\n')
        file.close()

def in_colors(img, contour, tmpl_parsed, pixel_factor=None, color_reduction=None, all_template=None):
    if color_reduction!=None:
        k = 32
        lut = np.zeros((256), np.uint8)
        espacio = np.linspace(0, 255, k).astype('uint8')
        espacio_dos = np.linspace(0, 255, k + 1).astype('uint8')
        for n in range(len(espacio_dos)):
            if n == 0:
                continue
            lut[espacio_dos[n - 1]:espacio_dos[n] + 1] = espacio[n - 1]
        img[:, :, 0] = lut[img[:, :, 0]]
        img[:, :, 1] = lut[img[:, :, 1]]
        img[:, :, 2] = lut[img[:, :, 2]]
    d=contour.copy()
    d[:,:,0]=d[:,:,0]-min(d[:,:,0])
    d[:, :, 1] = d[:, :, 1] - min(d[:, :, 1])
    masked=np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.drawContours(masked, [d], -1, 255, -1)
    masked=cv2.bitwise_and(img, img, mask=masked)
    temp=np.zeros(masked.shape[:2], np.uint8)
    for n in tmpl_parsed:
        boundaries_c = [(
            [
                0,
                range(0, 257, 8)[n[1]],
                range(0, 257, 8)[::-1][n[0] + 1]
            ],  # lower
            [
                250 * n[2],
                range(0, 257, 8)[n[1] + 1] - 1,
                range(0, 257, 8)[::-1][n[0]] - 1
            ]  # upper
        )]
        for (lower, upper) in boundaries_c:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
        temp = cv2.add(temp, cv2.inRange(masked, lower, upper))

    if pixel_factor != None:
        valores=np.count_nonzero(temp) / pixel_factor
    else:
        valores=np.count_nonzero(temp)

    if all_template!=None:
        for n in range(len(valores)):
            total=valores[n][all_template]
            counter=0
            for v in range(len(valores[n])):
                v=v+counter
                valores[n].insert(v+1, (valores[n][v]/total)*100)
                counter=counter+1
    return valores

def template_reader(path):
    filess = open(path, 'r')
    filess_tab = filess.read().replace('\n', '\t').split('\t')
    if filess_tab[len(filess_tab) - 1] == '':
        filess_tab = filess_tab[:len(filess_tab) - 1]
    filess.close()
    filess_tab = [float(n) for n in filess_tab]
    filess_tab = np.asarray(filess_tab)
    chromatic_arr = filess_tab.reshape(32, 32)
    filess_tab = np.argwhere(chromatic_arr > 0 )
    n_arr = []
    for (x, y) in filess_tab:
        n_arr.append([x, y, chromatic_arr[x][y]])
    return n_arr

def live_feed_roi(img, thresh, stbl_f=0, live=True):
    all = np.array([[[0,0]], [[0, img.shape[0]]], [[img.shape[1], img.shape[0]]], [[img.shape[1], 0]]])
    x_a, y_a, w_a, h_a = cv2.boundingRect(all)
    all_area=w_a*h_a
    img_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    check=False
    if live:
        img_roi = cv2.GaussianBlur(img_roi, (3, 3), 0)
        img_roi = cv2.Canny(img_roi, 20, 260)
        M = np.ones((2,2), np.uint8)
        img_roi = cv2.dilate(img_roi, M, iterations=2)
        img_roi = cv2.erode(img_roi, M, iterations=1)
        markers=detect_markers(img)
        if len(markers)>0:
            marker=markers[0].get_contours()
        else:
            marker=None
        _, cnts, _ = cv2.findContours(img_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #else:
    #    img_roi = cv2.GaussianBlur(img_roi, (3, 3), 0)
    #    img_roi = cv2.Canny(img_roi, 20, 260)
    #    M = np.ones((2,2), np.uint8)
    #    img_roi = cv2.dilate(img_roi, M, iterations=2)
    #    img_roi = cv2.erode(img_roi, M, iterations=1)

    if live:
        counter=0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if (w * h) > (all_area * thresh):
                if (x - 1) <= 0 or (y - 1) <= 0 or (x + w + 1) >= w_a or (y + h + 1) >= h_a:
                    cv2.drawContours(img, [c], -1, (0,0,255), -1)
                    check=False
                else:
                    cv2.drawContours(img, [c], -1, (0, 255, 0), -1)
                    counter=counter+1
        if counter==stbl_f and marker!=None:
            #markers[0].highlite_marker(img)
            check=True

            if not perspective_check(img, marker[:,0,:]):
                check=False
        else:
            check=False
            #cv2.rectangle(img, (x_a, y_a), (x_a + w_a, y_a + h_a), (0, 0, 255), 10)
        if check:
            cv2.rectangle(img, (x_a, y_a), (x_a + w_a, y_a + h_a), (0, 255, 0), 10)
        return [img, counter, check]
    else:
        #img=cv2.bitwise_and(img, img, mask=masked)
        #cv2.rectangle(img, (x_a, y_a), (x_a + w_a, y_a + h_a), (0, 255, 0), 10)
        return img

class CvCamera(Image):
    def __init__(self, **kwargs):
        super(CvCamera, self).__init__(**kwargs)
        self.base = sys.argv[0][:sys.argv[0].rfind('/')]
        self.home = os.path.expanduser("~")
        self.camera = PiCamera()
        self.camera.resolution=(1920,1080)
        self.camera.framerate=32
        self.camera.awb_mode='off'
        self.camera.awb_gains=(1.7,1.4)
        self.camera.exposure_mode='sports'
        self.camera.shutter_speed=9000
        self.camera.iso=200
        self.rawCapture = np.empty((1088,1920, 3), dtype=np.uint8)
        self.stable_check_f = 0
        self.umbral=0.0011
        self.live=True
        self.contours=[]
        self.name_save=''
        self.permission=False
    def set_live(self, on_live):
        self.live=on_live
    def get_check(self):
        return self.permission
    def set_name_save(self, name_save):
        self.name_save=name_save
    def set_umbral(self, valor):
        self.umbral=valor
    def get_live(self):
        return self.live
    def camera_play(self, dt):
        self.camera.capture(self.rawCapture, format="bgr", use_video_port=True)
        buf=self.rawCapture[184:904,320:1600]
        if self.live:
            #buf, self.stable_check_f, self.permission = live_feed_roi(buf, self.umbral, self.stable_check_f, live=self.live)

            buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGBA)
            buf = cv2.flip(buf, 0)
            buf = buf.tostring()
            image_texture = Texture.create(size=(1280,720), colorfmt='rgba')
            image_texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
            self.texture = image_texture
        #if self.live == False and self.permission:
        if self.live == False:
            #print('false', self.name_save) ##
            buf = self.rawCapture[184:904, 320:1600]
            cv2.imwrite(self.name_save+'.png', buf)



            #buf = live_feed_roi(buf, self.umbral, self.stable_check_f, live=self.live)

            #buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGBA)



            #buf = cv2.flip(buf, 0)
            #buf = buf.tostring()
            #image_texture = Texture.create(size=(1280,720), colorfmt='rgba')
            #image_texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
            #self.texture = image_texture



            #self.texture.save(self.name_save+'.png')

            #print(self.name_save+'.png') ##
            time.sleep(0.5)
            self.live=True
        else:
            self.live=True
    def cam_init(self):
        self.update = Clock.schedule_interval(self.camera_play, 1 / 25)

    def cam_cancel(self):
        self.update.cancel()

class Microwave(GridLayout):
    def __init__(self, **kwargs):
        super(Microwave, self).__init__(**kwargs)
        self.cols=3
        button_uva=Button(text='Uva')
        button_uva.bind(on_press=self.set_analize)
        self.add_widget(button_uva)
#        button_racimo = Button(text='Racimo')
#        button_racimo.bind(on_press=self.set_analize)
#        self.add_widget(button_racimo)
        button_naranja = Button(text='Naranja')
        button_naranja.bind(on_press=self.set_analize)
        self.add_widget(button_naranja)
        button_manzana = Button(text='Manzana')
        button_manzana.bind(on_press=self.set_analize)
        self.add_widget(button_manzana)
    def set_analize(self, instance):
        config_set(1, instance.text)

class Labeler(GridLayout):
    def __init__(self, **kwargs):
        super(Labeler, self).__init__(**kwargs)
        base = sys.argv[0][:sys.argv[0].rfind('/')]
        self.config=config_get()
        self.profiles = options_load(base+'/lists_tsv')
        self.rows=1
        self.profiles=self.profiles[int(self.config[1])][int(self.config[2])]
        #self.profiles = self.profiles[0][1]

        self.index=[]
        self.index_num=[]

        self.current=['','']

        self.buttons=[]

        self.options_list=[]

        counter=0
        self.buttons_index=[]

        for n in range(len(self.profiles)):
            if n==0:
                self.options_list.append([])
                self.index.append('None')
                self.index_num.append(n)
                continue
            if '(num)' in self.profiles[n][0]:
                self.index_num.append(n)
                self.options_list.append([])
                self.index.append('None')
                self.buttons.append(Label(text=self.profiles[n][0][:self.profiles[n][0].find('(num)')], id='label'+str(counter)))
                self.buttons_index.append('label'+str(counter))
                self.buttons.append(TextInput(multiline=False, text='0', input_filter='int', id='input'+str(counter)))
                self.buttons_index.append('input' + str(counter))
            else:
                self.options_list.append(self.profiles[n])
                for i in self.profiles[n]:
                    self.index.append(i)
                    self.index_num.append(n)
                self.buttons.append(Button(text=self.profiles[n][0], id='button'+str(counter), on_press=self.set_current))
                self.buttons_index.append('button' + str(counter))
            counter=counter+1
        for n in self.buttons:
            self.add_widget(n)
        #self.add_widget(Button(text='total', on_press=self.total_text))
    def set_current(self, dt):
        self.current[0] = dt.text
        self.current[1] = dt.id
    def list_display(self):
        return self.options_list[self.index_num[self.index.index(self.current[0])]]
    def total_text(self):
        text=''
        for n in self.buttons:
            if 'label' in n.id:
                text = text + n.text + '#'
            else:
                text=text+n.text+'_'
        return text[:len(text)-1]

    def obey(self, label):
        temp=[]
        label=label.split('_')
        for n in label:
            if '#' in n:
                n=n.split('#')
                temp.append(n[0])
                temp.append(n[1])
            else:
                temp.append(n)
        for n in range(len(self.buttons)):
            self.buttons[n].text=temp[n]

class Contour_fix(FloatLayout, Widget):
    def __init__(self, img_dir, **kwargs):
        self.img_dir=img_dir
        super(Contour_fix, self).__init__(**kwargs)
        self.Imagen = Image(source=self.img_dir + '.png', keep_ratio=True, allow_stretch=True)

        self.Labeler = Labeler(pos_hint={'top': 0.95, 'center_x': 0.5}, size_hint=(1, 0.06))
        for n in self.Labeler.buttons:
            if 'button' in n.id:
                n.on_release = self.scroll_labeler



        #lut = np.zeros(max(self.temp.flatten())+1, np.uint8)
        #lut[1:] = 255

        self.add_widget(self.Imagen)
        self.add_widget(self.Labeler)


    def get_current(self):
        return self.img_dir + '.png'

    def delete_file(self):
        os.remove(self.img_dir+'.png')
        #os.remove(self.img_dir + '.fruit')


    def scroll_labeler(self):
        grid = GridLayout(cols=1, spacing=5, size_hint_y=None)
        grid.bind(minimum_height=grid.setter('height'))
        self.page_scroll = ScrollView(do_scroll_x=False, size_hint=(1, None), size=(Window.width, Window.height))
        for n in self.Labeler.list_display():
            btn = Button(text=n, size_hint_y=None, height=200)
            btn.bind(on_press=self.flush_labeler)
            grid.add_widget(btn)
        self.page_scroll.add_widget(grid)
        self.add_widget(self.page_scroll)

    def flush_labeler(self, dt):
        self.Labeler.buttons[self.Labeler.buttons_index.index(self.Labeler.current[1])].text=dt.text
        self.clear_widgets([self.page_scroll])