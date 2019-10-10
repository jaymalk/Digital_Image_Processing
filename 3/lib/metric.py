# Library for psnr calculation

import numpy as np
from skimage.measure import compare_ssim as ssim

from mio import trace

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


# Calculate the SSIM b/w two images
# @param _img : original image (without noise)
# @param _nimg : noisy image
def calculate_ssim(_img: np.array, _nimg: np.array):
    try:
        return ssim(_img, _nimg)
    except:
        trace()