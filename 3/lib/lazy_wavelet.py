

import numpy as np
from typing import Tuple

# Lazy wavelet transform for 1D
def lazy_wavelet_reduce(_arr: np.ndarray, _len: int = None, _recurse: bool = True) -> np.ndarray:
    '''
    Important assumption, length must be of the form 2^n+1
    '''
    # Checking the length
    if _len is None: _len = len(_arr)
    elif _len <= 2: return _arr
    # Assertion on length
    assert((_len-2)&(_len-1) == 0)
    # Creating a temporary array to work with
    __temp = np.zeros(_arr.shape)
    __temp[:] = _arr
    # Performing the lazy transform on the array
    for _i in range(1, _len, 2):
        __temp[_i] -= (1/2)*(__temp[_i-1]+__temp[_i+1])
    # Performing the final transform
    for _i in range(2, _len-1, 2):
        __temp[_i] += (1/4)*(__temp[_i-1]+__temp[_i+1])
    __temp[0] += (0.5)*(__temp[1])
    __temp[-1] += (0.5)*(__temp[-2])
    if _recurse:
        # Setting the array
        __temp[:_len] = np.concatenate((__temp[:_len:2], __temp[1:_len:2]))
        # Recursively calling the next section
        return lazy_wavelet_reduce(__temp, (_len+1)//2)
    else: return __temp

# Inveres Lazy Wavelet
def lazy_wavelet_inverse(_arr: np.ndarray, _len: int = None, _recurse: bool = True) -> np.ndarray:
    '''
    Important assumpition, length must be of the form 2^n+1
    '''
    # Checking the length
    if _len is None: _len = len(_arr)
    elif _len <= 2: return _arr
    # Assertion on length
    assert((_len-2)&(_len-1) == 0)
    # Performing operations on lower level first
    if _recurse: _arr = lazy_wavelet_inverse(_arr, (_len+1)//2)
    # Creating a temporary array to work with
    __temp = np.zeros(_arr.shape)
    __temp[:] = _arr
    # Resetting the array tilt
    for i in range(_len//2 + 1):
        __temp[2*i] = _arr[i]
    for i in range(_len//2):
        __temp[2*i+1] = _arr[(_len+1)//2 + i]
    # Recreating the array back 
    # First the, previously, final transform
    __temp[-1] -= (0.5)*(__temp[-2])
    __temp[0] -= (0.5)*(__temp[1])
    for _i in range(2, _len-1, 2):
        __temp[_i] -= (1/4)*(__temp[_i-1]+__temp[_i+1])
    # Second, reversing the lazy transform
    for _i in range(1, _len, 2):
        __temp[_i] += (1/2)*(__temp[_i-1]+__temp[_i+1])
    # Return the array
    return __temp


# Applying the transform to 2D images (Standard)
def _lazy_transform(_img: np.ndarray, _shp: Tuple[int, int]) -> np.ndarray:
    '''
    Lazy transform on 2D images. 
    Note: both the dimensions must be of the form 2^n+1
    '''
    # Getting the dimensions
    __l1 = _shp[0]
    __l2 = _shp[1]
    # Returing if boundary dimensions
    if __l1 <= 2 or __l2 <= 2: return
    # Working on the first dimension (0)
    for _i in range(__l1):
        _img[_i,:__l2] = lazy_wavelet_reduce(_img[_i], __l2, False)
    # Working on the second dimension (1)
    for _i in range(__l2):
        _img[:__l1,_i] = lazy_wavelet_reduce(_img[:,_i], __l1, False)
    # Resetting the complete image (quadrant-wise)
    _img[:__l1,:__l2] = np.r_[np.c_[_img[0:__l1:2, 0:__l2:2], _img[1:__l1:2, 0:__l2:2]], np.c_[_img[0:__l1:2, 1:__l2:2], _img[1:__l1:2, 1:__l2:2]]]
    # Recursing on the main quadrant
    _lazy_transform(_img, ((__l1+1)//2, (__l2+1)//2))