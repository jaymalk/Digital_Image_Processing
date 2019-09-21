# Snippet for elegant FFT and IFFT using numpy (2 dimensional)

import numpy as np
import cv2 as cv


# Fast fourier transform (shifted)
# param _img : original input image (a 2D matrix)
# param _shp : shape tuple
# NOTE -> _shp should be fft optimal
def FFT(_img: np.ndarray, _shp: tuple, _shft: bool = True, _c_pad: bool = True) -> np.ndarray:
    _x = -_img.shape[0]+_shp[0]
    _y = -_img.shape[1]+_shp[1]
    # Getting padding
    _x1 = _x//2
    _x2 = (_x+1)//2
    _y1 = _y//2
    _y2 = (_y+1)//2
    # Padding the image
    if _c_pad:
        _img_pad = cv.copyMakeBorder(_img, _x1, _x2, _y1, _y2, cv.BORDER_CONSTANT)
    else:
        _img_pad = cv.copyMakeBorder(_img, 0, _x, 0, _y, cv.BORDER_CONSTANT)
    # Calculating the FFT
    _img_ft = np.fft.fft2(_img_pad)
    # Transforming (if offered)
    if _shft:
        _img_ft = np.fft.fftshift(_img_ft)
    # Return
    return _img_ft

# Inverse fast fourier transform
# param _img : original input image (a 2D matrix)
# param _shft : whether shift is considered
def IFFT(_img: np.ndarray, _shft: bool = True) -> np.ndarray:
    # Transforming (if shift)
    if _shft:
        _img_ft = np.fft.ifftshift(_img)
    # Calculating the IFFT
    _img_rec = np.fft.ifft2(_img_ft)
    # Return
    return _img_rec


# Ideal DFT Pad (for cv.dft)
def cvdft_pad(_img: np.ndarray):
    _x = cv.getOptimalDFTSize(_img.shape[0]) - _img.shape[0]
    _y = cv.getOptimalDFTSize(_img.shape[1]) - _img.shape[1]
    return cv.copyMakeBorder(_img, 0, _x, 0, _y, cv.BORDER_CONSTANT)