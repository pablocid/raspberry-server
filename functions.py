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

def tmpl_mask(img, tmpls_parsed, factor=1, cnts=None, prev_result=None, prev_header=None):
    try:
        cnts[0][0]
    except:
        cnts=[np.asarray([(0,0), (0, img.shape[0]), (img.shape[1], img.shape[0]), (img.shape[1], 0)]).reshape(4,1,2)]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    if prev_result==None:
        result=[]
    else:
        result=prev_result.copy()
    counter=0
    if prev_header==None:
        header=[]
    else:
        header=prev_header.copy()
    for cnt in cnts:
        if prev_result==None:
            result.append([])

        img_temp=contour_crop(img, cnt)
        for (name,tmpl_parsed) in tmpls_parsed:
            if counter==0:
                header.append(name+'_area')
            masked = ''
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
                c_mask = cv2.inRange(img_temp, lower, upper)
                if type(masked) == str:
                    masked = c_mask

                masked = cv2.add(masked, c_mask)
            if prev_result==None:
                result[-1].append(cv2.countNonZero(masked) * (factor ** 2))
            else:
                result[counter].append(cv2.countNonZero(masked) * (factor ** 2))
        counter+=1
    return [result, header]

def template_reader(directory):
    result=[]
    if directory[-1]!='/':
        directory+='/'
    for n in listdir(directory):
        if '.tsv' not in n:
            continue
        filess = open(directory+n, 'r')
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
        result.append([n[:-4], n_arr])
    return result

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
    color_reference = path.join(dir_path, "reference.png")
    colors = ['b', 'g', 'r']
    #search for the template
    img_pattern, _=four_point_transform(img, marker_coords)
    #colour cuantization into 2 colours with k-means clustering
    Z = img_pattern.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    img_pattern = res.reshape((img_pattern.shape))
    img_pattern = cv2.resize(img_pattern, (7,7), interpolation=cv2.INTER_LINEAR)
    #obtain the BGR values from the white area of the matched template
    h_p = img_pattern[3, 3]
    # obtain the BGR values from the black area of the matched template
    l_p = img_pattern[0,0]
    img_pattern=[h_p,l_p]
    img_reference = cv2.imread(color_reference, 1)
    h_r=img_reference[3, 3]
    l_r=img_reference[0,0]
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

def berry_shape(cnts, factor=1, prev_result=None, prev_header=None, fancy_output=None):
    """
    Calculates the equatorial and polar size of the berries by fitting an ellipse.
    :param cnts: List of contours that depicts the berries
    :param factor: value used to transform linear distance in pixels to a measurement unit. 1 by default
    :param prev_result: List. Previous data corresponding to each berry where the new data columns are going to be
    appended. If not given, returns a new list
    :param prev_header: List. Where the new data headers are going to be appended
    :param fancy_output: Numpy array image. Image to be edited with a visual output of the acquired data.
    :return: List. Data obtained for each berry; header for the data acquired.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if prev_header==None:
        header=['width', 'height', 'area']
    else:
        header=prev_header+['width', 'height', 'area']
    if prev_result==None:
        result=[]
    else:
        result=prev_result.copy()
    counter=0
    for cnt in cnts:
        area=cv2.contourArea(cnt)

        try:
            fancy_output[0]

            center, (MA, ma), angles = cv2.fitEllipse(cnt)
            cv2.ellipse(fancy_output, (center, (MA, ma), angles), (0,255,0), 1)
            # in angles, is the angle and start angle. We need the angle.
            point_a=line_newPoint(center, MA/2, (angles*(math.pi/180)))
            point_b=line_newPoint(center, ma/2, ((angles+90)*(math.pi/180)))

            cv2.circle(fancy_output, point_a, 5, (255, 255, 0), -1)
            cv2.circle(fancy_output, point_b, 5, (255, 0, 255), -1)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(fancy_output, 'width: ' + str(round(MA * factor, 2)), (x+w+2, y+h+2), font, 0.5,
                        (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(fancy_output, 'height: ' + str(round(ma * factor, 2)), (x+w+17, y+h+17), font, 0.5, (255, 0, 255), 1,
                        cv2.LINE_AA)


        except:
            _, (MA, ma), _ = cv2.fitEllipse(cnt)
        if prev_result==None:
            result.append([round(MA * factor, 2), round(ma * factor, 2), round(area*(factor**2), 2)])
        else:
            result[counter]=result[counter]+[round(MA * factor, 2), round(ma * factor, 2), round(area*(factor**2), 2)]
        counter+=1
    return [result, header]

def rachis_shape(cnts, factor=1, prev_result=None, prev_header=None, fancy_output=None):
    if prev_header==None:
        header=['width', 'height', 'area']
    else:
        header=prev_header+['width', 'height', 'area']
    if prev_result==None:
        result=[]
    else:
        result=prev_result.copy()
    counter=0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cnt in cnts:
        area=cv2.contourArea(cnt)
        try:
            fancy_output[0]
            temp_1, (width, height), temp_2 = cv2.minAreaRect(cnt)
            cv2.drawContours(fancy_output, [np.int0(cv2.boxPoints((temp_1, (width, height), temp_2)))], 0, (0, 255, 0),
                             2)
            x, y, w, h = cv2.boundingRect(cnt)
            box = cv2.boxPoints((temp_1, (width, height), temp_2))
            box = np.int0(box)
            cv2.circle(fancy_output,
                       (int(((box[1][0] - box[0][0]) / 2) + box[0][0]), int(((box[1][1] - box[0][1]) / 2) + box[0][1])),
                       5,
                       (255, 0, 255), -1)
            cv2.circle(fancy_output,
                       (int(((box[3][0] - box[0][0]) / 2) + box[0][0]), int(((box[3][1] - box[0][1]) / 2) + box[0][1])),
                       5,
                       (255, 255, 0), -1)
            cv2.putText(fancy_output, 'width: ' + str(round(width * factor, 2)), (x + w + 2, y + h + 2), font, 0.5,
                        (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(fancy_output, 'height: ' + str(round(height * factor, 2)), (x + w + 17, y + h + 17), font, 0.5,
                        (255, 0, 255), 1,
                        cv2.LINE_AA)
        except:
            _, (width, height), _=cv2.minAreaRect(cnt)
        if prev_result==None:
            result.append([width * factor, height * factor, area*(factor**2)])
        else:
            result[counter]=result[counter]+[width * factor, height * factor, area*(factor**2)]
        counter+=1
    return [result, header]
