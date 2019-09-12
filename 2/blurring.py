
import sys
import numpy as np
import cv2 as cv
from mio import read, write, show, trace
from noise import add_noise


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


# Main
if __name__ == "__main__":
    try:
        _nimg = disc_blur(read(sys.argv[1]), int(sys.argv[2]))
        _nimg = add_noise(_nimg, 'G', 20)
        write(sys.argv[3], _nimg)
    except:
        trace()