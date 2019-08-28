
import numpy as np
import cv2 as cv
from scipy import signal 
import sys
import matplotlib.pyplot as plt
from rescaling import *                                                                                  

def show(_img, title='Image'): 
    cv.imshow(title,_img) 
    cv.waitKey(0) 
    cv.destroyAllWindows() 
    cv.waitKey(1)                 

def apply_gaussian_filter(_img, _std):
    # Inner Function
    def gaussian_k(n, al):
        pts = np.arange((1-n)/2, (n+1)/2, 1)
        _x, _y = np.meshgrid(pts, pts)
        _gk = (_x**2 + _y**2)/(al**2)
        _gk = (1/(np.math.pi * (al**2))) * (np.exp(-_gk))
        return _gk
    _sz = max(_img.shape)
    _sz = _sz + 0 if _sz%2 == 1 else _sz+1
    _gsk = gaussian_k(_sz, _std)
    # Padding the image
    # Getting size H
    _m = _sz - _img.shape[0]
    _m1 = _m//2
    if _m%2 == 1:
        _m2 = _m1+1
    else:
        _m2 = _m1
    # Getting size V
    _n = _sz - _img.shape[1]
    _n1 = _n//2
    if _n%2 == 1:
        _n2 = _n1+1
    else:
        _n2 = _n1
    _img_p = cv.copyMakeBorder(_img, _m1, _m2, _n1, _n2, cv.BORDER_CONSTANT, value=0) 
    # Going into the fourier domain
    _img_f = np.fft.fftshift(np.fft.fft2(_img_p))
    _gsk_f = np.fft.fftshift(np.fft.fft2(_gsk))
    # Convolution (**Multiplication)
    _new_f = _img_f*_gsk_f
    # Inverse Transform
    _new_p = np.fft.ifft2(_new_f)
    _new_p = np.fft.ifftshift(_new_p)
    # Removing Padding
    _new = _new_p[_m1+1:_sz-_m2+1, _n1+1:_sz-_n2+1]
    # Return real values
    # return np.real(_new)
    return np.abs(_new)


def getV(_img, s, a, phi):
    al1 = 1/(2*(2**0.5))
    al2 = 1.6*al1
    V1 = apply_gaussian_filter(_img, al1*s)
    V2 = apply_gaussian_filter(_img, al2*s)
    V = (V1-V2)/(abs(V1)+((pow(2, phi)*a)/(s**2)))
    return V

def reinhard_map(_img, best_only = False):
    def lum_map(_img, _ol, _nl):
        _ni = np.zeros(_img.shape)
        _ni[:,:,0] = _img[:,:,0]*((_nl/(_ol+0.0001))**0.75)
        _ni[:,:,1] = _img[:,:,1]*((_nl/(_ol+0.0001))**0.75)
        _ni[:,:,2] = _img[:,:,2]*((_nl/(_ol+0.0001))**0.75)
        return _ni
    _img = gamma_crr(_img)
    al1 = 1/(2*(2**0.5))
    _lum = scale(_img[:,:,0]*0.299 + _img[:,:,1]*0.587 + _img[:,:,0]*0.114, 255)
    phi = 10
    a = 0.72
    _x = []
    _s = []
    for s in np.arange(0.1, 10, 0.1):
        _V = getV(_lum, s, a, phi)
        V1 = abs(apply_gaussian_filter(_lum, al1*s))
        _ld = _lum/(1+V1)
        best = scale(lum_map(_img, _lum, _ld), 255)
        if not best_only:
            write('./imgs/v1_'+str(round(s, 2))+'.jpg', best)
        if((best_only) and (_V.sum() < 0.5)): break
        _x.append(_V.sum())
        _s.append(s)
    if not best_only: 
        plt.plot(_s, _x)
        plt.show()
    if best_only: return best

if __name__ == '__main__':
    img = read(sys.argv[1])
    try:
        best = sys.argv[2]
        write('best.jpg', reinhard_map(img, True)) if best.startswith('T') else reinhard_map(img)
    except IndexError:
        reinhard_map(img)