

import cv2 as cv
import numpy as np
# from skimage import exposure

from align import image_alignment
from mio import *


# Histogram Equaliser (self)
def hist_equalise(_img: np.ndarray, _BW: bool=False) -> np.ndarray:
    # If black and white, simply work with the image
    if _BW:
        return cv.equalizeHist(_img)
    # Else, work on the V component of HSV
    else:
        _hsv = cv.cvtColor(_img, cv.COLOR_BGR2HSV)
        _hsv[:,:,2] = cv.equalizeHist(_hsv[:,:,2])
        return cv.cvtColor(_hsv, cv.COLOR_HSV2BGR)

# Adaptive Histogram Equalisation
def adaptive_hist_equalise(_img: np.ndarray, _BW: bool=False):
    # Creating the CLAHE component
    _clahe = cv.createCLAHE(clipLimit=0.3, tileGridSize=(40,40))
    # WORK on illumination if BGR
    if not _BW:
        # Converting to LAB
        _ilab = cv.cvtColor(_img, cv.COLOR_BGR2LAB)
        _ilab[:,:,0] = _clahe.apply(_ilab[:,:,0])
        # Getting the image back
        return cv.cvtColor(_ilab, cv.COLOR_LAB2BGR)
    # If black and white, simply work with the image
    else:
        return _clahe.apply(_img)


# Finding and applying contours to the image
def _contour_map(_img, _n=''):
    # Get the canny-edges
    _ed = cv.Canny(_img, 80, 160)
    # Dilating the image get the edges strengthened
    _ed = cv.dilate(_ed, np.ones((4,4)))
    write('dilate_'+_n, _ed)
    # Use the edges to get the countors
    _contour, _homography = cv.findContours(_ed, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # Setting up the segmenting mask
    _mask = np.zeros(_img.shape)
    # Getting the image with contours applied
    for _cnt in _contour:
        (_x, _y, _w, _h) = cv.boundingRect(_cnt)
        if (130>_w>100 and 2.7<_w/_h<3.3) or (160<_w<190 and 2.5<_w/_h<5.5):
            cv.rectangle(_img, (_x, _y), (_x+_w, _y+_h), (0,0,255), 2)
            cv.rectangle(_mask, (_x, _y), (_x+_w, _y+_h), (255,255,255), -1)
    # Return the images and the contours
    return _img, _mask, _contour


# The process for segmentation
def segmentation(_img, _template, _n='', _eq=False):
    # Aligning the image
    _img_aligned = image_alignment(_img, _template)
    # Eroding the image
    _imge = cv.erode(_img_aligned, np.ones((2,2)))
    # Getting the histogram equlaised
    if _eq:
        _img_eq = adaptive_hist_equalise(_imge)
    else:
        _img_eq = _imge
    # Getting the contours mapped
    _img_rect, _mask, _c = _contour_map(_img_eq, _n)
    write('rect'+_n, _img_rect)
    # Creating the masked-feature image
    _img_feat = np.where(_mask==0, 0, _img_aligned)
    write('rmask'+_n, _img_feat)
    # Returnting all the features
    return _img_feat, _mask

 