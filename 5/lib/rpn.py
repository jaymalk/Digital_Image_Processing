# Texture synthesis using random-phase-noise description of the original image

import numpy as np
from typing import Tuple
from array import ArrayType

from fft import *

def uniform_random_phase(_shp: Tuple[int, int]) -> ArrayType:
    '''
        TODO: DOC-STRING
    '''
    # Creating a random distribution
    _upn = np.random.uniform(-np.pi, np.pi, _shp)
    # Adding constraints
    # ODDNESS
    _1, _2 = _shp[0]//2, _shp[1]//2
    for _i in range(_shp[0]):
        for _j in range(_shp[1]):
            _upn[_1-_i-1][_2-_j-1] = -_upn[_i][_j]
    # BOUNDARY CONDITIONS
    _upn[0,0] = np.pi if np.random.choice([True, False]) else 0
    _upn[0,0] = np.pi if np.random.choice([True, False]) else 0
    _upn[0,0] = np.pi if np.random.choice([True, False]) else 0
    _upn[0,0] = np.pi if np.random.choice([True, False]) else 0
    # Return the UPN
    return _upn


def random_phase_noise(_img: ArrayType, _upn: ArrayType = None, _clr: bool = False) -> ArrayType:
    '''
        TODO: DOC-STRING
    '''

    # Setting the UPN
    if _upn is None:
        _U = uniform_random_phase(_img.shape[:2])
    else:
        _U = _upn

    # Working individually if colored image
    if _clr:
        _imgc = np.zeros(_img.shape, dtype=np.uint8)
        for _i in [0,1,2]:
            _imgc[:,:,_i] = random_phase_noise(_img[:,:,_i], _U)
        return _imgc
    
    # Image is B/W
    # Getting the FFT-ABS Value
    _X = abs(FFT(_img, _img.shape, False))

    # Adding the phase noise
    _N = _X * np.exp(_U*1j);

    # Getting the image back
    _imgn = np.abs(IFFT(_N, False))

    # Converting the type, and clipping the values
    _imgn -= _imgn.min()
    _imgn = np.where(_imgn > 255, 255, _imgn)
    _imgn = np.where(_imgn < 0, 0, _imgn)
    _imgn = np.array(np.floor(_imgn), np.uint8)

    # Return the random phase image
    return _imgn