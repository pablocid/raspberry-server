import os
import datetime
import numpy as np
import cv2
import math
import sys
import time
from math import atan2, sin, cos, sqrt

from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from detect import detect_markers
#from picamera import PiCamera

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
        return
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def line_length(linea):
    return sqrt(((linea[1][0]-linea[0][0])**2)+((linea[1][1]-linea[0][1])**2))

def point_proyection(point, rad):
    x = int(point[0] + (3500 * cos(rad)))
    y = int(point[1] + (3500 * sin(rad)))
    return (int(x), int(y))

def line_newPoint(point, length,rad):
    x = int(point[0] + (length * cos(rad)))
    y = int(point[1] + (length * sin(rad)))
    return (int(x), int(y))

def line_angle(line):
    return atan2(line[1][1] - line[0][1], line[1][0] - line[0][0])

def perspective_check(img, ret_contours=False):
    id_list=[]
    x_center = []
    y_center = []
    markers = detect_markers(img)

    counter=0

    if len(markers)>0:
        for n in range(len(markers)):
            n -= counter
            if markers[n].id not in [1742, 2713, 3116]:
                return [False, []]
            if markers[n].id not in id_list:
                id_list.append(markers[n].id)
                x_center.append(markers[n].center[0])
                y_center.append(markers[n].center[1])
            else:
                del(markers[n])
                counter+=1
        if len(id_list)!=3:
            if ret_contours:
                return [False, [], []]
            else:
                return [False, []]
    else:
        if ret_contours:
            return [False, [], []]
        else:
            return [False, []]

    x_center = int(round(sum(x_center) / len(x_center),0))
    y_center = int(round(sum(y_center) / len(y_center),0))

    uno={}
    tres={}

    marker=markers[id_list.index(2713)]

    counter = 0
    temp = 100000

    for c in marker.contours[:, 0, :]:
        temp_b = line_length([[x_center, y_center], c])
        if temp_b < temp:
            point = counter
            temp = temp_b
        counter += 1

    uno['a'] = marker.contours[{0: 2, 1: 3, 2: 0, 3: 1}[point], 0, :]
    tres['a'] = marker.contours[{0: 2, 1: 3, 2: 0, 3: 1}[point], 0, :]
    temp = 100000
    counter = 0
    for c in marker.contours[:, 0, :]:
        if counter != point and counter != {0: 2, 1: 3, 2: 0, 3: 1}[point]:
            temp_b = line_length([c, markers[id_list.index(1742)].center])
            if temp_b < temp:
                point_b = counter
                temp = temp_b
        counter += 1
    uno['b'] = marker.contours[point_b, 0, :]
    tres['b'] = marker.contours[{0: 2, 1: 3, 2: 0, 3: 1}[point_b], 0, :]
    marker = markers[id_list.index(1742)]
    counter = 0
    counter_fuga=0
    temp = 100000
    temp_fuga=100000
    for c in marker.contours[:, 0, :]:
        temp_b = line_length([[x_center, y_center], c])
        temp_b_fuga = line_length([uno['b'], c])
        if temp_b < temp:
            point = counter
            temp = temp_b
        if temp_b_fuga < temp_fuga:
            point_fuga=counter_fuga
            temp_fuga = temp_b_fuga
        counter_fuga+=1
        counter += 1
    uno['c'] = marker.contours[{0: 2, 1: 3, 2: 0, 3: 1}[point], 0, :]
    uno['fuga']=line_intersection([uno['a'], uno['c']], [tres['b'], marker.contours[{0: 2, 1: 3, 2: 0, 3: 1}[point_fuga], 0, :]])

    marker = markers[id_list.index(3116)]
    counter = 0
    counter_fuga = 0
    temp = 100000
    temp_fuga = 100000
    for c in marker.contours[:, 0, :]:
        temp_b = line_length([[x_center, y_center], c])
        temp_b_fuga = line_length([tres['b'], c])
        if temp_b < temp:
            point = counter
            temp = temp_b
        if temp_b_fuga < temp_fuga:
            point_fuga = counter_fuga
            temp_fuga = temp_b_fuga
        counter_fuga += 1
        counter += 1
    tres['c'] = marker.contours[{0: 2, 1: 3, 2: 0, 3: 1}[point], 0, :]
    tres['fuga'] = line_intersection([tres['a'], tres['c']],
                                    [uno['b'], marker.contours[{0: 2, 1: 3, 2: 0, 3: 1}[point_fuga], 0, :]])
    if ret_contours:
        return [True, [uno, tres], [n.contours for n in markers]]
    else:
        return [True, [uno, tres]]

def virtual_area(largo, alto, diccionario):
    """
    :param largo: en cm
    :param alto: en cm
    :param diccionario: coordenadas abc para largo y alto
    :return: cuatro puntos
    """
    #alto = round(((6.75*alto)-21.2625)/(3.6*alto), 2)
    #largo = round(((6.75*largo)-21.2625)/(3.6*largo), 2)

    #alto = round(((9.5 * alto) - 34.2) / (0+(5.9 * alto)), 2)
    #largo = round(((9.5 * largo) - 34.2) / (0 + (5.9 * largo)), 2)

    alto = round((56.05+(9.5*alto)) / (56.05+(5.9*alto)), 4)
    largo = round((56.05 + (9.5 * largo)) / (56.05 + (5.9 * largo)), 4)

    largo = ((line_length([diccionario[0]['a'], diccionario[0]['c']])*line_length([diccionario[0]['b'], diccionario[0]['c']]))*(1-largo))/((largo*line_length([diccionario[0]['b'], diccionario[0]['c']]))-line_length([diccionario[0]['a'], diccionario[0]['c']]))
    alto = ((line_length([diccionario[1]['a'], diccionario[1]['c']]) * line_length(
        [diccionario[1]['b'], diccionario[1]['c']])) * (1 - alto)) / (
                        (alto * line_length([diccionario[1]['b'], diccionario[1]['c']])) - line_length(
                    [diccionario[1]['a'], diccionario[1]['c']]))
    alto = line_newPoint(diccionario[1]['c'], alto, line_angle([diccionario[1]['a'], diccionario[1]['c']]))
    largo = line_newPoint(diccionario[0]['c'], largo, line_angle([diccionario[0]['a'], diccionario[0]['c']]))

    if diccionario[1]['fuga']==None:
        final_one=[largo, point_proyection(largo, line_angle([diccionario[1]['a'], diccionario[1]['c']]))]
    else:
        final_one=[largo, diccionario[1]['fuga']]
    if diccionario[0]['fuga']==None:
        final_two=[alto, point_proyection(alto, line_angle([diccionario[0]['a'], diccionario[0]['c']]))]
    else:
        final_two=[alto, diccionario[0]['fuga']]
    final = line_intersection(final_one, final_two)
    return [tuple(diccionario[0]['a']), largo, final, alto]

def ray_castQuery(contours, points, res=(1920,1080)):
    if type(contours)==list:
        black=np.zeros(res[::-1], np.uint8)
        cv2.polylines(black, contours, True, 255, thickness=1)
    else:
        black = contours.copy()
    points=np.array(points)
    test_instructions = ['black[:n[1], n[0]]', 'black[n[1], n[0]:]', 'black[n[1]+1:, n[0]]', 'black[n[1], :n[0]]']
    final=[]
    for n in points:
        counter=0
        location=False
        while counter<4:
            test=eval(test_instructions[counter])
            reject=False
            prev=test[0]
            for t in test[1:]:
                if t==0:
                    prev=0
                    continue
                if t == 255 == prev:
                    reject=True
                    break
                else:
                    prev=255
            if reject:
                counter+=1
                continue
            if np.count_nonzero(test)%2 != 0:
                location=True
                break
            counter+=1
        final.append(location)
    return final


def live_feed_roi2(img, thresh, ignore=None, area=None, fancy=False):
    try:
        area[0]
        img = four_point_transform(img, area)
    except:
        pass
    img=cv2.blur(img, (2,2))
    new_x = 640 / img.shape[1]
    frame = img.copy()
    img = cv2.resize(img, None, None, fx=new_x, fy=new_x, interpolation=cv2.INTER_LINEAR)
    all = np.array([[[0, 0]], [[0, frame.shape[0]]], [[frame.shape[1], frame.shape[0]]], [[frame.shape[1], 0]]])
    x_a, y_a, w_a, h_a = cv2.boundingRect(all)
    all_area = w_a * h_a

    #print(all_area)
    #print(line_length([all[0,0,:], all[1,0,:]])*line_length([all[0,0,:], all[2,0,:]]))

    img_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    check = False
    img_roi = cv2.Canny(img_roi, 10, 100)
    M = np.ones((2, 2), np.uint8)
    img_roi = cv2.dilate(img_roi, M, iterations=1)
    _, cnts, _ = cv2.findContours(img_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counter = 0
    if fancy:
        if ignore!=None:
            factor=cv2.minAreaRect(ignore[0])[1][0]
            factor=3.6/factor
            factor_area=cv2.minAreaRect(ignore[0])[1]
            factor_area=12.96/(factor_area[0]*factor_area[1])
        else:
            factor=1
        final=[]

    for c in cnts:
        c[:, 0, 0] = c[:, 0, 0] // new_x
        c[:, 0, 1] = c[:, 0, 1] // new_x
        x, y, w, h = cv2.boundingRect(c)

        if all_area*0.1 > (w * h) > (all_area * thresh):
            if x > 0 < y and (x+w) < (x_a+w_a) and (y+h) < (y_a + h_a):
                if ignore != None:
                    if not ray_castQuery(ignore, [(int(x + (w / 2)), int(y + (h / 2)))],
                                         res=(frame.shape[:2][::-1]))[0]:
                        if fancy:
                            ellipse = cv2.fitEllipse(c)
                            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

                            cv2.line(frame, (int(ellipse[0][0]), int(ellipse[0][1])), line_newPoint(ellipse[0], ellipse[1][1]/2, (ellipse[2]+90)*(math.pi/180)), (255,0,0), 2)
                            cv2.line(frame, (int(ellipse[0][0]), int(ellipse[0][1])),
                                     line_newPoint(ellipse[0], ellipse[1][0] / 2, ellipse[2] * (math.pi / 180)),
                                     (255, 0, 255), 2)

                            #cv2.line(frame, ellipse[0], ellipse[1], (255, 0, 0), 5)
                            #rect = cv2.minAreaRect(c)
                            ##print(rect)
                            #box = cv2.boxPoints(rect)
                            #box = np.int0(box)
                            #cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                            area_roi = np.zeros((h, w), np.uint8)
                            cv2.ellipse(area_roi, ((int(w / 2), int(h / 2)), ellipse[1], ellipse[2]), 255, -1)

                            final.append([round(ellipse[1][0]*factor, 2), round(ellipse[1][1]*factor, 2), round(np.count_nonzero(area_roi)*factor_area, 2)])


                            text='w='+str(final[-1][0])+'; h='+str(final[-1][1])
                            cv2.putText(frame, text, (int(ellipse[0][0]), int(ellipse[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                        cv2.LINE_AA)


                        else:
                            cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                        check = True
                        counter = counter + 1
                else:
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                    check = True
                    counter = counter + 1
    if check:
        pass
        #cv2.polylines(frame, [area], True, (255, 255, 255), thickness=1)
    # cv2.drawContours(frame, ignore, -1, 255, 3)
    if fancy:
        return [frame, check, final]
    else:
        return [frame, check]

def live_feed_roi(img, thresh, ignore=None, area=None):
    try:
        area[0]
        img = four_point_transform(img, area)
    except:
        pass
    new_x = 640 / img.shape[1]
    frame = img.copy()
    img = cv2.resize(img, None, None, fx=new_x, fy=new_x, interpolation=cv2.INTER_LINEAR)
    all = np.array([[[0, 0]], [[0, frame.shape[0]]], [[frame.shape[1], frame.shape[0]]], [[frame.shape[1], 0]]])
    x_a, y_a, w_a, h_a = cv2.boundingRect(all)
    all_area = w_a * h_a

    #print(all_area)
    #print(line_length([all[0,0,:], all[1,0,:]])*line_length([all[0,0,:], all[2,0,:]]))

    img_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    check = False
    img_roi = cv2.Canny(img_roi, 10, 100)
    M = np.ones((2, 2), np.uint8)
    img_roi = cv2.dilate(img_roi, M, iterations=1)
    _, cnts, _ = cv2.findContours(img_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counter = 0
    for c in cnts:
        c[:, 0, 0] = c[:, 0, 0] // new_x
        c[:, 0, 1] = c[:, 0, 1] // new_x
        x, y, w, h = cv2.boundingRect(c)
        if all_area > (w * h) > (all_area * thresh):
            if x > 0 < y and (x+w) < (x_a+w_a) and (y+h) < (y_a + h_a):
                if ignore != None:
                    if not ray_castQuery(ignore, [(int(x + (w / 2)), int(y + (h / 2)))],
                                         res=(frame.shape[:2][::-1]))[0]:
                        cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                        check = True
                        counter = counter + 1
                else:
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                    check = True
                    counter = counter + 1
    if check:
        pass
        #cv2.polylines(frame, [area], True, (255, 255, 255), thickness=1)
    # cv2.drawContours(frame, ignore, -1, 255, 3)
    return [frame, check]

def proper_correction(img, square_coordinates):
    square_coordinates=square_coordinates[np.lexsort((square_coordinates[:,0,1], square_coordinates[:,0,0]))]
    hight=line_length([square_coordinates[0, 0, :], square_coordinates[1,0,:]])
    width = line_length([square_coordinates[0, 0, :], square_coordinates[2, 0, :]])
    if hight>width:
        img=cv2.resize(img, None, None, fx=1, fy=width/hight, interpolation=cv2.INTER_LINEAR)
    elif hight<width:
        img = cv2.resize(img, None, None, fx=hight/width, fy=1, interpolation=cv2.INTER_LINEAR)
    return img

def header_decoder(img):
    #print(img[0,0,0])
    if img[0,0,0]!=127:
        return [None, None, None]
    values=[]
    values.append(img[0, 1, 0] * 0.00005)
    values.append(img[0, 2, 0] * 0.375)
    values.append(img[0, 3, 0] * 0.375)
    return values

class CvCamera(Image):
    def __init__(self, **kwargs):
        super(CvCamera, self).__init__(**kwargs)
        self.base = sys.argv[0][:sys.argv[0].rfind('/')]
        self.home = os.path.expanduser("~")

        self.camera = PiCamera(resolution=(1920, 1080), framerate=32)
        self.camera.iso = 100
        time.sleep(2)
        self.camera.shutter_speed = 4000
        self.camera.exposure_mode = 'off'
        self.camera.awb_mode = 'off'
        #red/blue
        self.camera.awb_gains = (1.65, 1.4)
        self.camera.brightness = 38
        self.rawCapture = np.empty((1088, 1920, 3), dtype=np.uint8)

        self.umbral=0.0011
        self.live=True
        self.alto=15
        self.largo=20
        self.buffer=5

    def set_live(self, on_live):
        self.live=on_live

    def set_alto(self, valor):
        self.buff = [[[], []], [[], []], [[], []], [[], []]]
        self.alto=valor

    def set_largo(self, valor):
        self.buff = [[[], []], [[], []], [[], []], [[], []]]
        self.largo=valor

    def set_umbral(self, valor):
        self.umbral=valor
    def set_exposure(self, value_parse):
        if value_parse=='black':
            self.camera.shutter_speed=4000
            self.camera.brightness = 40
        elif value_parse=='white':
            self.camera.shutter_speed = 8000
            self.camera.brightness = 38

    def get_live(self):
        return self.live

    def camera_play(self, dt):
        self.camera.capture(self.rawCapture, format="bgr", use_video_port=True)
        buf=self.rawCapture[:]
        if self.live:

            self.check, coords = perspective_check(buf)

            if self.check:
                vpoints = virtual_area(self.largo, self.alto, coords)
                for n in range(len(vpoints)):
                    self.buff[n][0].append(vpoints[n][0])
                    self.buff[n][1].append(vpoints[n][1])

            if len(self.buff[0][0]) == self.buffer:
                for n in range(len(self.buff)):
                    self.four_points[n][0]=int(round(sum(self.buff[n][0]) / self.buffer))
                    self.four_points[n][1]=int(round(sum(self.buff[n][1]) / self.buffer))
                    del(self.buff[n][0][0])
                    del(self.buff[n][1][0])

                pts = np.array(self.four_points, np.int32)

                buf, self.check = live_feed_roi(buf, self.umbral, area=pts)

                buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGBA)
                image_texture = Texture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='rgba')
                buf = cv2.flip(buf, 0)
                buf = buf.tostring()

                image_texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
                self.texture = image_texture

        if self.live == False and self.check:
            #self.camera.capture(self.rawCapture, format="bgr", use_video_port=True)
            buf = self.rawCapture[:]
            data = np.zeros((1, 1920, 3), np.uint8)
            data[0, 0, :] = 127
            data[0, 1, :] = int(self.umbral/0.00005)
            data[0, 2, :] = int(self.largo/0.375)
            data[0, 3, :] = int(self.alto/0.375)
            cv2.imwrite('/home/pi/temp.png', np.append(data,buf[:1081], axis=0))
            self.live=True
        else:
            self.live=True

    def cam_init(self):
        self.buff=[[[],[]],[[],[]],[[],[]],[[],[]]]
        self.four_points=[[[],[]],[[],[]],[[],[]],[[],[]]]
        self.update = Clock.schedule_interval(self.camera_play, 1 / 25)

    def cam_cancel(self):
        self.update.cancel()