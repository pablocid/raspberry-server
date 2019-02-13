import numpy as np
import cv2
import math
from  os import path, listdir

dir_path = path.dirname(path.realpath(__file__))

def points_distance(pt1, pt2):
    """
    Distance between two points
    :param pt1: x,y pair 1
    :param pt2: x,y pair 2
    :return: float
    """
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

def line_newPoint(point, length,rad):
    """
    Calculates the coordinates of a new point given an origin point, a length and an angle
    :param point: x,y pair origin point
    :param length: Distance to the new point
    :param rad: float. Angle of the new point to the origin point (in radians)
    :return: int tuple. Coordinates of the new point
    """
    x = int(point[0] + (length * math.cos(rad)))
    y = int(point[1] + (length * math.sin(rad)))
    return (int(x), int(y))

def contour_crop(img, cnt, background=False):
    """
    Crop an image to a contour
    :param img: numpy array image. Image to crop
    :param cnt: numpy contour. Selected contour to be cropped in the input image
    :param background: bool. True for return the output image without background (only the portion in the contour)
    :return: numpy array image. Input image cropped to the contour
    """
    temp = img[np.min(cnt[:, 0, 1]):np.max(cnt[:,0,1]), np.min(cnt[:, 0, 0]):np.max(cnt[:,0,0])]
    if background:
        return temp
    cnt[:, 0, 0] = cnt[:, 0, 0] - np.min(cnt[:, 0, 0])
    cnt[:, 0, 1] = cnt[:, 0, 1] - np.min(cnt[:, 0, 1])
    black=np.zeros((np.max(cnt[:,0,1]), np.max(cnt[:,0,0])), np.uint8)
    cv2.drawContours(black, [cnt], -1, 255, -1)
    return cv2.bitwise_and(temp, temp, mask=black)

def factor_calculator(markers_list, real_border=1):
    return (real_border*0.94) / (cv2.contourArea(markers_list[0].coordinates().reshape(4, 1, 2)) ** (1 / 2))

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
    """
    Edited four points transformation function from imutils package.
    :param image: numpy array image.
    :param pts: four pairs of x,y coordinates
    :return: List. Numpy array transformed image; perspective matrix.
    """
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
    return [warped, M]

def roi_filter(resolution_xy, contours):
    result=[]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cx=int(x+w/2)
        cy=int(y+h/2)
        if w*h>resolution_xy[0]*resolution_xy[1]*0.0001:
            result.append(c)
    return result

def contour_transform(contours_list, perspective_matrix):
    """
    Apply a perspective matrix to a list of contours.
    :param contours_list: List. Contours to be transformed
    :param perspective_matrix: perspective matrix from the getPerspectiveTransform of the OpenCV library
    :return: List. List of transformed contours
    """
    result=[]
    for cnt in contours_list:
        cnt = cv2.perspectiveTransform(cnt.reshape(1, len(cnt), 2).astype('float32'), perspective_matrix)
        result.append(cnt.reshape(len(cnt[0]), 1, 2).astype('int'))
    return result

def img_check(img):
    """
    Detects contours in the input image (BGR array) and looks for the white square background and the contours in it.
    :param img: BGR numpy array.
    :return: List with two elements. The first one is a list of contours located inside the white square. The second one
    the contour depicting the square background (None if the square can't be determined).
    """
    frame=img.copy()
    result = []

    w_a, h_a = frame.shape[1], frame.shape[0]

    new_x = 870 / img.shape[1]
    img = cv2.resize(img, None, None, fx=new_x, fy=new_x, interpolation=cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_roi = cv2.bitwise_or(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1], 80, 100), cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 100))
    M = np.ones((2, 2), np.uint8)
    img_roi = cv2.dilate(img_roi, M, iterations=1)

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

    try:
        cnts[major_area]
    except:
        return [cnts, None]

    if temp > 0 and len(cv2.approxPolyDP(cnts[major_area], 0.04*cv2.arcLength(cnts[major_area],True),True))!=4:
        return [cnts, None]

    cnts[major_area]=cv2.approxPolyDP(cnts[major_area], 0.04*cv2.arcLength(cnts[major_area],True),True)
    counter = 0

    for n in hierarchy[0]:
        if n[3] == major_area and counter != major_area and len(cnts[counter])>10:
            result.append(cnts[counter])
        counter += 1
    return [result, cnts[major_area]]