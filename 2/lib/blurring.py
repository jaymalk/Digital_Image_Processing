
import sys
import numpy as np
import cv2 as cv
from mio import read, write, show, trace
from noising import add_noise


# Function for getting disc-shaped h(x, y)
# @param n : radius of kernel/disc
# @param r : ratio of kernel/disc radius (default 1)
def disc(n: int, r: int = 1):
    try:
        _m = np.ones((2*n+1, 2*n+1))
        for i in range(n+1):
            for j in range(n+1):
                if ((i-n)**2 + (j-n)**2)**0.5 > n/r:
                    _m[i, j] = 0
                    _m[2*n-i, j] = 0
                    _m[2*n-i, 2*n-j] = 0
                    _m[i, 2*n-j] = 0
        return _m/_m.sum()
    except:
        trace()


# Blurring the image with disc
# @param _img : input image
# @param _n : size of the kernel
def disc_blur(_img: np.array, _n: int):
    try:
        _nimg = cv.filter2D(_img, cv.CV_8U, disc(_n, 3))
        if _nimg is None:
            raise ValueError('Convolution unsucessful.')
        return _nimg
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


# Function for blurring an image aroung the edges 
# This is done so as to reduce the ringing effect induced while recovering the image in fourier domain
# @param _img: image to be blurred
# @param _d: depth og blurring
def blur_edge(_img: np.ndarray, _d: int = 31, _bw: bool = False): 
    # Getting frame dimensions
    _h, _w  = _img.shape[:2] 
    # Padding the image (with rotational wrap)
    __pad = cv.copyMakeBorder(_img, _d, _d, _d, _d, cv.BORDER_WRAP) 
    # Getting the blur component
    _img_blur = cv.GaussianBlur(__pad, (2*_d+1, 2*_d+1), -1)[_d:-_d,_d:-_d] 
    # Getting the merging constants
    _y, _x = np.indices((_h, _w)) 
    __dist = np.dstack([_x, _w-_x-1, _y, _h-_y-1]).min(-1)
    # The constants 
    _c = np.minimum(np.float32(__dist)/_d, 1.0) 
    # Checking if BW
    if _bw:
        return _img*_c + _img_blur*(1-_c)
    # Temporary matrix
    _temp = np.ones(_img.shape) 
    # Applying on domains
    for i in [0,1,2]: 
        _temp[:,:,i] *= _img[:,:,i]*_c 
        _img_blur[:,:,i] *= (1-_c)
    # Returning the image 
    return _temp + _img_blur                                                   



# Main
if __name__ == "__main__":
    try:
        _nimg = disc_blur(read(sys.argv[1]), int(sys.argv[2]))
        _nimg = add_noise(_nimg, 'G', 20)
        write(sys.argv[3], _nimg)
    except:
        trace()