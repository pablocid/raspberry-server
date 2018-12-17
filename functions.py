import numpy as np
import cv2
import math

def points_distance(pt1, pt2):
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

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

def roi_filter(mask, marker_center=None):
    if marker_center==None:
        marker_center=[0,0]
    try:
        _, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result=[]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cx=int(x+w/2)
        cy=int(y+h/2)
        if w*h>mask.shape[1]*mask.shape[0]*0.0001 and points_distance(marker_center, [cx, cy])>mask.shape[0]*0.09:
            result.append(c)
    return result


def img_check(img, analysis=False):
    frame=img.copy()
    result = []

    w_a, h_a = frame.shape[1], frame.shape[0]

    new_x = 875 / img.shape[1]
    img = cv2.resize(img, None, None, fx=new_x, fy=new_x, interpolation=cv2.INTER_LINEAR)
    #img_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1]

    check = False
    img_roi = cv2.bitwise_or(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1], 80, 100), cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 100))
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
    comparator=cv2.contourArea(cnts[major_area])

    counter=0
    for n in hierarchy[0]:
        if n[3]==major_area and counter!=major_area and cv2.contourArea(cnts[counter])>comparator*0.7:
            temp=counter
        counter+=1
    major_area=temp

    if temp > 0 and len(cv2.approxPolyDP(cnts[major_area], 0.04*cv2.arcLength(cnts[major_area],True),True))!=4:
        return [cnts, None, None]

    cnts[major_area]=cv2.approxPolyDP(cnts[major_area], 0.04*cv2.arcLength(cnts[major_area],True),True)
    counter = 0
    temp=0
    black = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    #cv2.drawContours(black, cnts, major_area, 255, -1)

    if analysis:
        for n in hierarchy[0]:
            if n[3] == major_area and counter != major_area:
                cv2.drawContours(black, cnts, counter, 255, -1)
                temp+=1
                #cv2.imshow('frame', black)
                #cv2.waitKey(0)
            counter += 1
        kernel = np.ones((5, 5), np.uint8)
        black=cv2.erode(black, kernel, iterations=1)
    #cv2.imshow('frame', black)
    #cv2.waitKey(0)
    return [cnts, major_area, black]