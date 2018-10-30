from __future__ import print_function
from __future__ import division

try:
    import cv2
except ImportError:
    raise Exception('Error: OpenCv is not installed')

from numpy import array, rot90
import numpy as np
from ar_markers.coding import decode, extract_hamming_code
from ar_markers.marker import MARKER_SIZE, HammingMarker

BORDER_COORDINATES = [
    [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 0], [1, 6], [2, 0], [2, 6], [3, 0],
    [3, 6], [4, 0], [4, 6], [5, 0], [5, 6], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],
]

ORIENTATION_MARKER_COORDINATES = [[1, 1], [1, 5], [5, 1], [5, 5]]


def validate_and_turn(marker):
    # first, lets make sure that the border contains only zeros
    for crd in BORDER_COORDINATES:
        if marker[crd[0], crd[1]] != 0.0:
            raise ValueError('Border contians not entirely black parts.')
    # search for the corner marker for orientation and make sure, there is only 1
    orientation_marker = None
    for crd in ORIENTATION_MARKER_COORDINATES:
        marker_found = False
        if marker[crd[0], crd[1]] == 1.0:
            marker_found = True
        if marker_found and orientation_marker:
            raise ValueError('More than 1 orientation_marker found.')
        elif marker_found:
            orientation_marker = crd
    if not orientation_marker:
        raise ValueError('No orientation marker found.')
    rotation = 0
    if orientation_marker == [1, 5]:
        rotation = 1
    elif orientation_marker == [5, 5]:
        rotation = 2
    elif orientation_marker == [5, 1]:
        rotation = 3
    marker = rot90(marker, k=rotation)
    return marker


def detect_markers(img):
    """
    This is the main function for detecting markers in an image.

    Input:
      img: a color or grayscale image that may or may not contain a marker.

    Output:
      a list of found markers. If no markers are found, then it is an empty list.
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img.copy()

    new_x = 640 / img.shape[1]

    img = cv2.resize(img, None, None, fx=new_x, fy=new_x,
                              interpolation=cv2.INTER_LINEAR)

    width, height = img.shape
    img = cv2.Canny(img, 100, 255)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # We only keep the long enough contours
    min_contour_length = min(width, height) / 10
    # contours = [contour for contour in contours if cv2.arcLength(contour, True) > min_contour_length]
    warped_size = 49
    canonical_marker_coords = array(
        (
            (0, 0),
            (warped_size - 1, 0),
            (warped_size - 1, warped_size - 1),
            (0, warped_size - 1)
        ),
        dtype='float32')

    markers_list = []
    
    for contour in contours:
        if cv2.arcLength(contour, True) <= min_contour_length:
            continue
        approx_curve = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.01, True)
        if not (len(approx_curve) == 4 and cv2.isContourConvex(approx_curve)):
            continue
        
        approx_curve[:, 0, 0] = approx_curve[:, 0, 0] // new_x
        approx_curve[:, 0, 1] = approx_curve[:, 0, 1] // new_x
        sorted_curve = array(
            cv2.convexHull(approx_curve, clockwise=False),
            dtype='float32'
        )
        persp_transf = cv2.getPerspectiveTransform(sorted_curve, canonical_marker_coords)
        warped_img = cv2.warpPerspective(gray, persp_transf, (warped_size, warped_size))
        
        # do i really need to convert twice?
        if len(warped_img.shape) > 2:
            warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        else:
            warped_gray = warped_img
        
        _, warped_bin = cv2.threshold(warped_gray, 130, 255, cv2.THRESH_BINARY)
        
        marker = warped_bin.reshape(
            [MARKER_SIZE, warped_size // MARKER_SIZE, MARKER_SIZE, warped_size // MARKER_SIZE]
        )
        
        marker = marker.mean(axis=3).mean(axis=1)
        marker[marker < 130] = 0
        marker[marker >= 130] = 1
        
        try:
            marker = validate_and_turn(marker)
            hamming_code = extract_hamming_code(marker)
            marker_id = int(decode(hamming_code), 2)
            markers_list.append(HammingMarker(id=marker_id, contours=approx_curve))
        except ValueError:
            continue
    
    return markers_list
