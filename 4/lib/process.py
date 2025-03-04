

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
    write('align'+_name, _imga)
    _imga = cv.medianBlur(_imga, 3)
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


if __name__ == "__main__":
    # Reading global variables
    _template = read('./__main.jpg', False)
    _mask = read('./best.png', False)

    # Working on all the images
    _path = '../all-images/'
    for _n in ['booklets/']:
        _p = _path+_n
        for _i in run(['ls', _p], capture_output=True).stdout.decode().rstrip().split():
            _img = read(_p+_i, False)
            # _img = cv.cvtColor(_img, cv.COLOR_GRAY2BGR)
            process_image(_img, _n[:-1]+'_'+_i)
