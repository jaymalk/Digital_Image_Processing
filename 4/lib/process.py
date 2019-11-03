

import cv2 as cv
import numpy as np
from subprocess import run

from mio import *
from align import *
from contour_map import *


global _template, _mask
_template = None
_mask = None


def process_image(_img: np.ndarray, _name: str = 'default'):
    # First aligning the image
    _imga = image_alignment(_img, _template)
    # Adaptive Histogram Equalisation
    _imgadap = adaptive_hist_equalise(_imga)
    # Segmenting fields using the mask
    _imgm = np.where(_mask == 0, 255, _imgadap)
    write('omask'+_name, _imgm)
    # Working on individual segments, for characters
    # Greyscaling
    _imgmg = cv.cvtColor(_imgm, cv.COLOR_BGR2GRAY)
    # Thresholding
    _imgmg[_imgmg>127] = 255
    # Inverting Scale
    _im = 255-_imgmg
    write('final_'+_name, _im)
    # Now, labelling the characters
    _contour, _homography = cv.findContours(_im, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    _msk = np.zeros(_imgmg.shape)
    for _cnt in _contour:
        (_x, _y, _w, _h) = cv.boundingRect(_cnt)
        if _w < 50 and 50 > _h > 10:
            cv.rectangle(_imgadap, (_x, _y), (_x+_w, _y+_h), (0,0,255), 1)
            cv.rectangle(_msk, (_x, _y), (_x+_w, _y+_h), 255, -1)
    write('lab_'+_name, _imgadap)
    write('char_'+_name, np.where(_msk == 0, 0, _imgmg))