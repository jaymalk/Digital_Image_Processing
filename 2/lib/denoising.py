#!/anaconda3/bin/python

import sys
import cv2 as cv
import numpy as np
from mio import read, write, trace
from noise import add_noise


# Calculating noisy image PSNR
# @param _img : original image (without noise)
# @param _nimg : noisy image
def calculate_psnr(_img: np.array, _nimg: np.array):
    try:
        # Converting to float
        _img = np.array(_img, dtype=np.float)
        _nimg = np.array(_nimg, dtype=np.float)
        # Calculating MSE
        _sqdiff = (_img-_nimg)**2
        _mse = _sqdiff.sum()/_sqdiff.size
        # Calculating PSNR (dB)
        _psnr = 10*np.log(255*255/_mse)
        return _psnr
    except:
        trace()


# Applying blur to noisy image
# @param _img : image input (noisy)
# @param _size : size of the filter
# @param _type : type of the filter (ME mean) (MD median) (GS gaussian)
def blur(_img: np.array, _size: int, _type: str = 'ME'):
    try:
        # Mean
        if _type == 'ME':
            _bimg = cv.blur(_img, (_size, _size))
        # Median
        elif _type == 'MD':
            _bimg = cv.medianBlur(_img, _size)
        # Gaussian
        elif _type == 'GS':
            _bimg = cv.GaussianBlur(_img, (_size, _size), 0)
        # Error
        else:
            raise ValueError('Invalid filter type ' + _type)
        # Return
        return _bimg
    except:
        trace()


# Finding the image with highest PSNR and respective filter size
# @param _img : original image
# @param _nimg : noisy image
# @param _tp : type of filter (in sync with blur)
def find_best_size(_img: np.array, _nimg: np.array, _tp: str = 'ME'):
    try:
        # Setting default as best
        _bst = _nimg
        _psnr = calculate_psnr(_bst, _img)
        _fsz = 1
        # Looping over sizes
        for _sz in range(3, 50, 2):
            _bimg = blur(_nimg, _sz, _tp)
            _psnri = calculate_psnr(_bimg, _img)
            # Better found
            if _psnri < _psnr:
                _bst = _bimg
                _fsz = _sz
                _psnr = _psnri
        # Return best triplet
        return (_bst, _fsz, _psnr)
    except:
        trace()
