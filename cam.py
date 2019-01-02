#import os
#import sys
import time
import threading
import socket
import numpy as np
import cv2
from detect import detect_markers, detect_markers_integrated
from subprocess import check_output
from fractions import Fraction
from functions import img_check, four_point_transform, contour_transform
BRIGHT=0
ISO=0
SHUTTER=0
AWB_GAINS=0
CONTRAST=0

try:
    from picamera import PiCamera
    ips = check_output('ifconfig')
    if '192.168.50.4' in ips.decode("UTF-8"):
        BRIGHT = 40
        ISO = 100
        SHUTTER = 18346
        AWB_GAINS = (Fraction(57, 32), Fraction(195, 128))

    else:
        BRIGHT=38
        ISO = 200
        SHUTTER = 10000
        AWB_GAINS = (Fraction(52, 32), Fraction(193, 128))
        CONTRAST=50
except:
    pass
    



class Cameraman():
    def __init__(self):
        self.camera = PiCamera(resolution=(1640, 1232), framerate=15)
        #self.camera.iso = ISO
        time.sleep(2)
        #self.camera.shutter_speed = SHUTTER
        #self.camera.exposure_mode = 'off'
        self.camera.awb_mode = 'off'
        self.camera.awb_gains = AWB_GAINS # red/blue
        #self.camera.brightness = BRIGHT
        self.camera.contrast = CONTRAST
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
                    self.busy = False
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
        buf = cv2.cvtColor(self.rawCapture[:, :1640], cv2.COLOR_RGB2BGR)
        new_x = 640 / buf.shape[1]
        cv2.imwrite('/home/pi/capture.png',buf)
        cv2.imwrite('/home/pi/preview.png', cv2.resize(buf, None, None, fx=new_x, fy=new_x, interpolation=cv2.INTER_LINEAR))
        contours, bg_cnt=img_check(buf)
        try:
            test=bg_cnt[0]
        except:
            return 'no_square_background'
        buf, persp_mtx = four_point_transform(buf, bg_cnt[:, 0, :])
        contours = contour_transform(contours, persp_mtx)
        markers, contours = detect_markers_integrated(buf, contours)
        if len(contours)==0:
            return 'no_objects'
        try:
            if markers[0].id!=3116:
                return 'wrong_marker'
        except:
            return 'no_marker'
        return 'ok'

    def capture_full(self):
        #print(self.camera.shutter_speed)
        #print(self.camera.awb_gains)
        #print(self.camera.brightness)
        #print(self.camera.contrast)
        #print(self.camera.exposure_speed)
        #print(self.camera.iso)
        #print(self.camera.analog_gain)
        #print(self.camera.digital_gain)
        #print(self.camera.contrast)

        #self.camera.capture(self.rawCapture, format="rgb", use_video_port=False)
        #buf = cv2.cvtColor(self.rawCapture[:, :1640], cv2.COLOR_RGB2BGR)

        #check, msg=img_check(buf)

        #cv2.imwrite('/home/pi/temp.png', buf)
        self.busy=False
        return 'done'
    def photo_precheck(self, np_image):
        pass

if __name__ == '__main__':
    Cameraman()
