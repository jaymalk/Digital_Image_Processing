'''
Library for deblurring images...
Contains various techniques, such as Wiener Filter, Blind Deconvolution...
'''

import numpy as np
import cv2 as cv

from fft import FFT, IFFT
from mio import trace

# Weiner Filtering...
# param _img : noisy/blurry image
# param _fest : estimate of original image
# param _h : estimate of deblurring function (PSF)
# param _n : estimate of noise distribution
# param _BW : whether the image is black and white
def weiner_(_img: np.ndarray, _h: np.ndarray, _fest: np.ndarray = None, _n: np.ndarray = None, _BW: bool = True):
    try:
        # Getting the optimal DFT size
        __shp = _img.shape
        '''
        Getting fourier transforms (all needed)
        Sensing
        g = h*f + n (Real domain)
        G = HF + N (Fourier Domain)
        & weiner filter...
        H* / (abs(H)**2 + K),
        where K can be either of |N(u,v)|/|F(u,v)| or |N|/|F| (over complete matrix)
        & recovered image  GH*
        '''
        H = FFT(_h, __shp)
        G = FFT(_img, __shp)
        if _n is None:
            assert(_fest == None)
            K = 0.0001
        else:
            N = FFT(_n, __shp)
            F_ = FFT(_fest, __shp)
            K = (abs(N)**2).sum()/(abs(F_)**2).sum()
        # Getting the inverse filter
        H_w = (np.conj(H)) * (1/(abs(H)**2 + K))
        # Getting the image back...
        F_rec = G * H_w
        # Coming back to spatial domain
        f_rec = np.real(IFFT(F_rec))
        return np.fft.fftshift(f_rec)
    except:
        trace()
