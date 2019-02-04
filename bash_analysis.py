import cv2
from detect import detect_markers_integrated
from functions import roi_filter, colorBalance, template_reader, tmpl_mask, img_check, contour_transform, \
    four_point_transform, berry_shape, factor_calculator, rachis_shape
import time
import os
import numpy as np

def bash_analysis(folder, color_folder=None, keyword=None):
    """
    Hace uso de las funciones en functions.py para analizar las fotos de una carpeta. ¡¡¡Ver comentario en linea 67!!!
    :param folder: str. path to de folder containing the photos. The photos were previously checked by the Raspberry Pi.
    :param color_folder: str. path to the folder containing the tsv templates for color analysis. If not given, only
    shape analysis is going to be performed.
    :param keyword: List. Strings of keywords for filtering the photos to be analyzed in the given folder.
    :return: None. Writes a report.tsv file containing the data obtained from the photos.
    """

    initial = time.time()
    if folder[-1] != '/':
        folder += '/'
    final = open(folder + 'report.tsv', 'w')
    final.close()
    counter = 0
    for n in os.listdir(folder):
        # Filter the files in folder: must have a png extension and, if given, the at least one of the keywords.
        if n[-4:] != '.png' or n[0] == '.' or '_watcher.png' in n:
            continue
        keycheck = False
        if keyword != None:
            for k in keyword:
                if k in n:
                    keycheck = True
                    break
            if not keycheck:
                continue
        print(n)
        img = cv2.imread(folder + n)

        # detect the white square and the objects in it
        contours, bg_cnt = img_check(img)

        # transform the image to white square area. Also apply the transformation to the objects
        img, persp_mtx = four_point_transform(img, bg_cnt[:, 0, :])
        contours = contour_transform(contours, persp_mtx)

        # detect marker and delete it from contours
        markers, contours = detect_markers_integrated(img, contours)

        # (optional) calculate factor pixel to cm
        factor = factor_calculator(markers, real_border=5.0)

        # (optional) filter detected objects
        contours = roi_filter(img.shape[:2][::-1], contours)

        #img = colorBalance(img, markers[0].coordinates())

        # UNDER CONSTRUCTION: generate a visual output
        black = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(black, contours, -1, 255, -1)
        black=cv2.bitwise_and(img, img, mask=black)

        # prepare the header and a data collector variables
        header = ['file']
        result = []
        for c in contours:
            result.append([n])

        # (optional) shape analysis. IMPORTANT: change this function according to the photos to be analyze
        # result, header = berry_shape(contours, factor, prev_result=result, prev_header=header)
        result, header = berry_shape(contours, factor, prev_result=result, prev_header=header, fancy_output=black)

        if color_folder != None:
            # (optional) color balance
            #img = colorBalance(img, markers[0].coordinates())

            # (optional) color analysis
            templates = template_reader(color_folder)
            result, header = tmpl_mask(img, templates, factor, contours, prev_result=result, prev_header=header)

        cv2.imwrite(folder + n + '_watcher.png', black)
        if counter == 0:
            final = open(folder + 'report.tsv', 'a')
            final.write('\t'.join(header) + '\n')
            final.close()
        final = open(folder + 'report.tsv', 'a')
        final.write('\n'.join(['\t'.join(list(map(str, n))) for n in result]) + '\n')
        final.close()
        counter += 1
    print('Data saved in', folder + 'report.tsv')
    print('\ntime:', time.time() - initial)