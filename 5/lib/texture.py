# Library for texture synthesis, with various methods

from typing import Tuple, List
from array import ArrayType
import numpy as np
from itertools import repeat

from steerable import *
from mio import *
from histogram import *
from fft import *

# Texture synthesis using steerable pyramids
def texture_synthesis_steerable(_img: ArrayType, K: int, Q: int, _iter: int = 5) -> ArrayType:
    # Creating image steerable pyramids
    _p = pyramids_fast(_img, K, Q, _cvt=False)
    # Creating the noise
    _n = np.random.normal(0.5, 0.4, _img.shape)
    # Matching the histogram
    _n = match_hist(_n, _img)
    # Loop
    for _ in range(_iter):
        # Creating the steerable pyramids
        _pn = pyramids_fast(_n, K, Q, _cvt=False)
        # Matching histograms
            # H0
        _pn[0][0] = match_hist(_pn[0][0], _p[0][0])
            # Final LOW
        _pn[-1][0] = match_hist(_pn[-1][0], _p[-1][0])
            # Orientations
        for i in range(K):
            for j in range(Q):
                _pn[i+1][j] = match_hist(_pn[i+1][j], _p[i+1][j])
        # Recreating the image
        _n = recreate_fast(_pn)
        # Matching the histograms
        _n = match_hist(_n, _img)
    # Return the image
    return _n