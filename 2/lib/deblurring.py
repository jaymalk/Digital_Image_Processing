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
def weiner_(_img: np.ndarray, _h: np.ndarray, SNR: float = 0.00000001, _BW: bool = True):
    try:
        # Getting the optimal DFT size
        __shp = _img.shape[:2]
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

        # If the image is colored (BGR)
        if not _BW:
            __temp = np.zeros(_img.shape)
            for _i in [0,1,2]:
                __temp[:,:,_i] = weiner_(_img[:,:,_i], _h, SNR, True)
            return __temp
        # else
        H = FFT(_h, __shp)
        G = FFT(_img, __shp)
        K = SNR
        # Getting the inverse filter
        H_w = (np.conj(H)) * (1/(abs(H)**2 + K))
        # Getting the image back...
        F_rec = G * H_w
        # Coming back to spatial domain
        f_rec = np.real(IFFT(F_rec))
        return np.fft.fftshift(f_rec)
    except:
        trace()
