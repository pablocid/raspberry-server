import numpy as np
import cv2
import math

def points_distance(pt1, pt2):
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

import cv2
import numpy as np
import math
from  os import path

dir_path = path.dirname(path.realpath(__file__))
url_reference = path.join(dir_path, "reference.png")

color_reference = path.join(dir_path, "color_reference2.jpg")

def colorBalance(img, marker_coords):
    """
    Color correction based on a coloured template as objective. Be sure the input image shows the paper sheet covering
    the whole frame (paper sheet background only).

    The test file (test if the average BGR values of the color template are closer to the template found in the test
    image after the colorBalance)

    >>> reference = cv2.imread('color_reference2.jpg')
    >>> reference_values=[np.average(reference[:,:,n]) for n in range(3)]
    >>> image = cv2.imread('nts_cb_test.png')
    >>> image_values=[np.average(image[16:62, 22:68, n]) for n in range(3)]
    >>> difference=[[],[]]
    >>> for n in range(3):
    ...     difference[0].append(abs(reference_values[n]-image_values[n]))
    >>> image = colorBalance(image)
    >>> image_values=[np.average(image[16:62, 22:68, n]) for n in range(3)]
    >>> for n in range(3):
    ...     difference[1].append(abs(reference_values[n]-image_values[n]))
    ...     print(difference[0][n]>difference[1][n], end=',')
    True,True,True,

    :param img: numpy BGR array. The black square in the frame should measure 30x30 to 100x100 px.
    :return: Colours corrected image as a numpy BGR array.
    """
    colors = ['b', 'g', 'r']
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #search for the template

    template = cv2.imread(url_reference, 0)
    w, h = template.shape[::-1]
    meth = 'cv2.TM_CCOEFF'
    method = eval(meth)
    res = cv2.matchTemplate(img_gray, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img_pattern = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    #colour cuantization into 2 colours with k-means clustering

    Z = img_pattern.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    img_pattern = res.reshape((img_pattern.shape))

    #obtain the BGR values from the white area of the matched template

    h_p = img_pattern[0, 0]

    # obtain the BGR values from the black area of the matched template

    l_p = img_pattern[math.floor(img_pattern.shape[0] / 2), math.floor(img_pattern.shape[1] / 2)]
    if h_p.all() == l_p.all():
        h_p = img_pattern[img_pattern.shape[0] - 1, img_pattern.shape[1] - 1]



    img_pattern=[h_p,l_p]
    img_reference = cv2.imread(color_reference, 1)
    Z = img_reference.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    img_reference = res.reshape((img_reference.shape))

    h_r=img_reference[0, 0]
    l_r=img_reference[math.floor(img_reference.shape[0] / 2), math.floor(img_reference.shape[1] / 2)]
    img_reference=[h_r, l_r]
    lut = []
    for n in range(len(colors)):
        lut.append(np.zeros((256), np.uint8))
        lut[n][0:img_pattern[1][n]] = np.linspace(0, img_reference[1][n] - 1, img_pattern[1][n])   #comentar esta linea y la 58 para ver pixeles en la foto fuera del rango del patron
        lut[n][img_pattern[1][n]:img_pattern[0][n] + 1] = np.linspace(img_reference[1][n], img_reference[0][n],
                                                                      img_pattern[0][n] - img_pattern[1][n] + 1)
        lut[n][img_pattern[0][n] + 1:256] = np.linspace(img_reference[0][n] + 1, 255, 255 - img_pattern[0][n])   #comentar esta linea y la 55 para ver pixeles en la foto fuera del rango del patron
    img_ch = cv2.split(img)
    for n in range(len(colors)):
        img_ch[n] = cv2.LUT(img_ch[n], lut[n]).astype(np.uint8)
    return cv2.merge(img_ch)

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
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

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
    black = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    #cv2.drawContours(black, cnts, major_area, 255, -1)

    if analysis:
        for n in hierarchy[0]:
            if n[3] == major_area and counter != major_area:
                cv2.drawContours(black, cnts, counter, 255, -1)
                result.append(cnts[counter])
                #cv2.imshow('frame', black)
                #cv2.waitKey(0)
            counter += 1
    #cv2.imshow('frame', black)
    #cv2.waitKey(0)
    return [result, cnts[major_area], black]