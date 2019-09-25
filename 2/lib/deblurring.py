'''
Library for deblurring images...
Contains various techniques, such as Wiener Filter, Blind Deconvolution...
'''

import numpy as np
import cv2 as cv

from fft import FFT, IFFT
from mio import show, trace

from denoising import total_variation_norm

# Weiner Filtering... I
# param _img : noisy/blurry image
# param _fest : estimate of original image
# param SNR: snr value estimate for the image
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

# Weiner Filtering...
# param _img : noisy/blurry image
# param f_ : estimate of original image (should be the same dim as _img)
# param _h : estimate of deblurring function (PSF)
# param _noise : estimate of noise distribution
# param _BW : whether the image is black and white
def weiner(_img: np.ndarray, _h: np.ndarray, _noise: np.ndarray, f_: np.ndarray, _BW: bool = True):
    try:
        _h /= _h.sum()
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
                __temp[:,:,_i] = weiner_(_img[:,:,_i], _h, _noise, f_[:,:,_i], True)
            return __temp
        # else
        H = FFT(_h, __shp)
        G = FFT(_img, __shp)
        N = FFT(_noise, __shp)
        F_ = FFT(f_, __shp)        
        # Getting the inverse filter
        H_w = (np.conj(H)) * (1/(abs(H)**2 + ((abs(N)**2)/(abs(F_)**2))))
        # Getting the image back...
        F_rec = G * H_w
        # Coming back to spatial domain
        f_rec = np.real(IFFT(F_rec))
        return np.fft.fftshift(f_rec)
    except:
        trace()



# Trying for edge preserving deblurring
def edge_preserving_deblurring(_img: np.ndarray, _h: np.ndarray, _lam: float = 1, _BW: bool = True, __show: bool = False):
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
                __temp[:,:,_i] = edge_preserving_deblurring(_img[:,:,_i], _h, _lam=_lam, _BW=True)
            return __temp
        # else
        H = FFT(_h, __shp)
        G = FFT(_img, __shp)
        # Getting TV norm
        _norm, _grad = total_variation_norm(_img)
        #SHOW
        if __show:
            show(_grad)
        # Getting TV FFT
        _L = FFT(_grad, _img.shape)
        #SHOW
        if __show:
            show(abs(_L)/abs(_L).max())
        # Getting the inverse filter
        H_w = (np.conj(H)) * (1/(abs(H)**2 + _lam/abs(_L)))
        # Getting the image back...
        F_rec = G * H_w
        # Coming back to spatial domain
        f_rec = np.real(IFFT(F_rec))
        return np.fft.fftshift(f_rec)
    except:
        trace()
