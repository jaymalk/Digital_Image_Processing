# Simple library for getting the gaussian and laplacian of an image


import numpy as np

from mio import *


# Gaussian
def gaussian_special(_img: np.ndarray, _lvl: int):
    # Asserting, provided levels are viable
    assert(pow(2, _lvl) <= np.min(_img.shape))
    # Pyramid List
    _g = []
    # LOOP
    for _ in range(_lvl):
        # Appending the image
        _g.append(_img.copy())
        # Reducing the image, by 2
        _img = (_img[::2,::2] + _img[1::2,::2] + _img[::2,1::2] + _img[1::2,1::2])/4
    # Final addition
    _g.append(_img.copy())
    # Return
    return _g

# Laplacian
def laplacian_special(_img: np.ndarray, _lvl: int):
    # Getting the gaussian
    _g = gaussian_special(_img, _lvl)
    # Pyramid List
    _l = []
    # LOOP
    for _i in range(1, len(_g)):
        # Expanding an image
        _e = _g[_i-1].copy()
        # Subtracing the values
        _e[::2, ::2] -= _g[_i]
        _e[1::2, ::2] -= _g[_i]
        _e[::2, 1::2] -= _g[_i]
        _e[1::2, 1::2] -= _g[_i]
        # Appending the copy
        _l.append(_e)
    # Final addition
    _l.append(_g[-1])
    # Return
    return _l

# Recreating from the laplacian
def laplacian_create(_l):
    # Getting the starting image
    _imgr = _l[-1]
    # LOOP
    for _i in range(-1, -len(_l), -1):
        # Getting the current image
        _e = _l[_i-1].copy()
        # Adding thee low-image
        _e[::2, ::2] += _imgr
        _e[1::2, ::2] += _imgr
        _e[::2, 1::2] += _imgr
        _e[1::2, 1::2] += _imgr
        # Setting the variable right
        _imgr = _e
    # Return the image
    return _imgr
