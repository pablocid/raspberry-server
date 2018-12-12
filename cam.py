#import os
#import sys
import time
import threading
import socket
import numpy as np
import cv2
from detect import detect_markers
from picamera import PiCamera
from subprocess import check_output

BRIGHT=0
ISO=0
SHUTTER=0
AWB_GAINS=0
CONTRAST=0

ips = check_output('ifconfig')
if '192.168.50.4' in ips.decode("UTF-8"):
    BRIGHT = 25
    ISO = 100
    SHUTTER = 25000
    AWB_GAINS = (1.55, 1.5)
else:
    BRIGHT=38
    ISO = 200
    SHUTTER = 10000
    AWB_GAINS = (1.55, 1.4)
    CONTRAST=60
    

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

def img_check(img):
    frame=img.copy()

    w_a, h_a = frame.shape[1], frame.shape[0]

    new_x = 640 / img.shape[1]
    img = cv2.resize(img, None, None, fx=new_x, fy=new_x, interpolation=cv2.INTER_LINEAR)
    img_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    check = False
    img_roi = cv2.Canny(img_roi, 10, 100)
    M = np.ones((2, 2), np.uint8)
    img_roi = cv2.dilate(img_roi, M, iterations=1)

    #cv2.imshow('frame', img_roi)
    #cv2.waitKey(0)

    try:
        _, cnts, hierarchy = cv2.findContours(img_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        cnts, hierarchy = cv2.findContours(img_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    counter = 0
    major_area=0
    temp=0
    for c in cnts:
        c[:, 0, 0] = c[:, 0, 0] // new_x
        c[:, 0, 1] = c[:, 0, 1] // new_x
        x, y, w, h = cv2.boundingRect(c)
        if (w*h)>=(w_a*h_a)*0.6:
            if temp<(w * h):
                temp=(w * h)
                major_area=counter
        counter += 1
    if temp > 0 and len(cv2.approxPolyDP(cnts[major_area], 0.04*cv2.arcLength(cnts[major_area],True),True))!=4:
        return [check, 'no_square_background']

    counter=0
    for n in hierarchy[0]:
        if n[3]==major_area and counter!=major_area:
            temp=counter
        counter+=1
    major_area=temp

    temp=four_point_transform(frame, cv2.approxPolyDP(cnts[major_area], 0.04*cv2.arcLength(cnts[major_area],True),True)[:,0,:])
    temp = detect_markers(temp[int(temp.shape[0]*0.85):, int(temp.shape[1]*0.85):])

    try:
        if temp[0].id != 3116:
            return [check, 'wrong_marker']
    except:
        return [check, 'no_marker']

    #black=np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

    counter = 0
    temp=0
    for n in hierarchy[0]:
        if n[3] == major_area and counter != major_area:
            #cv2.drawContours(black, cnts, counter, 255, -1)
            temp+=1
        counter += 1
    #cv2.imshow('frame', black)
    #cv2.waitKey(0)
    if temp<=1:
        return [check, 'no_objects']
    return [True, 'ok']

class Cameraman():
    def __init__(self):
        self.camera = PiCamera(resolution=(1640, 1232), framerate=15)
        #self.camera.iso = ISO
        time.sleep(2)
        #self.camera.shutter_speed = SHUTTER
        #self.camera.exposure_mode = 'off'
        #self.camera.awb_mode = 'off'
        #self.camera.awb_gains = AWB_GAINS # red/blue
        #self.camera.brightness = BRIGHT
        #self.camera.contrast = CONTRAST
        self.rawCapture = np.empty((1232, 1664, 3), dtype=np.uint8)

        self.busy=True
        self.breaker=False
        self.photo_check=''

        self.in_screen()

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
                if not self.busy:
                    self.busy=True
                    msg=self.capture_full()
                    c.send(msg.encode('utf-8'))
                else:
                    c.send('busy'.encode('utf-8'))
            if a == 'preview':
                if not self.busy:
                    self.busy=True
                    msg=self.capture_preview()
                    c.send(msg.encode('utf-8'))
                else:
                    c.send('busy'.encode('utf-8'))
            else:
                c.send('duno bro'.encode('utf-8'))

    def in_screen(self):
        t1 = threading.Thread(target=self.server_call, )
        t1.start()
        self.busy = False
        print('Camera ready')
    def capture_preview(self):
        self.camera.capture(self.rawCapture, format="rgb", use_video_port=False)
        buf = cv2.cvtColor(self.rawCapture[:], cv2.COLOR_RGB2BGR)
        check, msg=img_check(buf)
        new_x = 640 / buf.shape[1]
        buf = cv2.resize(buf, None, None, fx=new_x, fy=new_x, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('/home/pi/temp.jpg', buf)
        self.busy=False
        return msg
    def capture_full(self):
        self.camera.capture(self.rawCapture, format="rgb", use_video_port=False)
        buf = cv2.cvtColor(self.rawCapture[:], cv2.COLOR_RGB2BGR)
        #check, msg=img_check(buf)
        cv2.imwrite('/home/pi/temp.png', buf)
        self.busy=False
        return 'done'
    def photo_precheck(self, np_image):
        pass

if __name__ == '__main__':
    Cameraman()
