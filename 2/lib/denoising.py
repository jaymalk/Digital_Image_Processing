#!/anaconda3/bin/python

import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mio import read, write, trace, show
from noising import add_noise
from blurring import blur

__eps = np.finfo(np.float32).eps

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

# Finding the image with highest PSNR and respective filter size
# @param _img : original image
# @param _nimg : noisy image
# @param _tp : type of filter (in sync with blur)
def find_best_size(_img: np.array, _nimg: np.array, _tp: str = 'ME', _plt: bool = False):
    try:
        # Setting default as best
        _bst = _nimg
        _psnr = calculate_psnr(_bst, _img)
        _fsz = 1
        # If plot is to be shown
        if _plt:
            _szl = []
            _psnrl = []
        # Looping over sizes
        for _sz in range(3, 50, 2):
            _bimg = blur(_nimg, _sz, _tp)
            _psnri = calculate_psnr(_bimg, _img)
            # Plotting
            if _plt:
                _szl.append(_sz)
                _psnrl.append(_psnri)
            # Better found
            if _psnri > _psnr:
                _bst = _bimg
                _fsz = _sz
                _psnr = _psnri
        # Showing plot
        if _plt:
            plt.plot(_szl, _psnrl)
            plt.show()
        # Return best triplet
        return (_bst, _fsz, _psnr)
    except:
        trace()


# ! TOTAL VARIATION DENOISING

def total_variation_norm(_img: np.ndarray):
    # Derivative along x-axis
    _Dx = _img - np.roll(_img, -1, axis=1)
    # Derivative along y-axis
    _Dy = _img - np.roll(_img, -1, axis=0)
    # Calculating L2 norm of the gradient
    _gradn = np.sqrt(_Dx**2 + _Dy**2 + __eps)
    _norm = _gradn.sum()
    # Calculating the desired gradient
    _c = 0.5/_gradn
    _Dx, _Dy = 2*_c*_Dx, 2*_c*_Dy # Scaling accordingly
    # Gradient
    _grad = _Dx+_Dy
    _grad[:,1:] -= _Dx[:,:-1]
    _grad[1:] -= _Dy[:-1]
    # Returning the pair (_norm, _grad)
    return _norm, _grad


def tv_denoise(_img: np.ndarray, _lmbda: float = 0.001):
    # Setting up
    # Keeping the original image copy
    _org = _img.copy()

    # Function working as our image iterator
    def next_(_img: np.ndarray, _grad: np.ndarray, _step: float = 0.001):
        return _img - _grad*_step
    
    # Function working as evaluator
    def eval_(_img: np.ndarray):
        # Total variation parameters
        _TV_loss, _TV_grad = total_variation_norm(_img)
        # Absolute (L2) parameters
        _L2_grad = _img - _org
        _L2_loss = (_L2_grad**2).sum()
        # Total _grad and _loss
        _grad = _TV_grad + _lmbda*_L2_grad
        _loss = _TV_loss + _lmbda*_L2_loss
        # Return
        return _loss, _grad

    # Loop parameters
    loop_loss = np.inf
    # Main loop
    while True:
        _loss, _grad = eval_(_img)
        if _loss > loop_loss:
            break
        loop_loss = _loss
        _img = next_(_img, _grad)
    
    # Returning the final created image
    return _img