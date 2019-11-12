# Library for constructing the steerable pyramids of an image

from typing import Tuple, List
from array import ArrayType
import numpy as np

from mio import *
from fft import *

# Global lambda for calculating the distance
global _d
_d = lambda x, y: np.sqrt(x**2 + y**2)
# Global lambda for calculating the theta
global _theta
_theta = lambda x, y: (np.arctan(y/x) if x != 0 else np.pi/2)
# Global lambda for calculating the factorial
global _fact
_fact = lambda x: (1 if x <= 0 else x*_fact(x-1))


# Creating the Low-Pass Filters
def low_pass(_shp: Tuple[int, int], _scl: float = 1) -> ArrayType:
    # Creating the matrix
    _l = np.zeros(_shp, dtype=complex)
    # Getting the parameters
    _m, _n = _shp
    # Getting the mid ranges
    _mx, _my = _m//2, _n//2
    # Setting the points
    for _i in range(_m): 
        for _j in range(_n): 
            # Getting the distance
            _r = _d(-(_i-_mx)/_m, -(_j-_my)/_n)
            _r *= (2*np.pi) 
            # Scaling the radius
            _r *= _scl
            # Checking the case
            if _r <= np.pi/4: 
                _l[_i,_j] = 1 
            elif _r <= np.pi/2: 
                _l[_i,_j] = np.cos((np.pi/2)*np.log2(4*_r/np.pi))
    # Return the filter
    return _l


# Creating the High-Pass Filters
def high_pass(_shp: Tuple[int, int], _scl: float = 1) -> ArrayType:
    # Creating the matrix
    _h = np.ones(_shp, dtype=complex)
    # Getting the parameters
    _m, _n = _shp
    # Getting the mid ranges
    _mx, _my = _m//2, _n//2
    # Setting the points
    for _i in range(_m): 
        for _j in range(_n): 
            # Getting the distance
            _r = _d(-(_i-_mx)/_m, (_j-_my)/_n)
            _r *= (2*np.pi) 
            # Scaling the radius
            _r *= _scl
            # Checking the case
            if _r <= np.pi/4: 
                _h[_i,_j]= 0
            elif _r <= np.pi/2: 
                _h[_i,_j] = np.cos((np.pi/2)*np.log2(2*_r/np.pi))
    # Return the filter
    return _h


# Creating oriented-band-pass-filters
def oriented(_shp: Tuple[int, int], Q: int) -> List[ArrayType]:
    # Setting the alpha, normalizer
    _alpha = pow(2, Q-1) * _fact(Q-1) / np.sqrt(Q * _fact(2*(Q-1)))  
    # Getting the parameters
    _m, _n = _shp
    # Getting the mid ranges
    _mx, _my = _m//2, _n//2
    # List
    _list = []
    # Looping to get all the filters
    for _q in range(Q):
        # The image
        P = np.zeros(_shp) 
        # Get angle parameters
        _1 = np.pi*_q/Q 
        _2 = np.pi*(_q-Q)/Q  
        # LOOP
        for _i in range(_m):  
            for _j in range(_n):  
                # Get theta
                _t = _theta(-(_i-_mx)/_m, (_j-_my)/_n)
                # Check case
                if abs(_t-_1) <= np.pi/2: 
                    P[_i,_j] += _alpha * (np.cos(_t-_1)**(Q-1)) 
                elif abs(_t-_2) <= np.pi/2: 
                    P[_i,_j] += _alpha * (np.cos(_t-_2)**(Q-1)) 
        # Appending
        _list.append(P)                                         
    # Return the list
    return _list


# Creating the steerable pyramids of an image
def pyramids(_img: ArrayType, K: int, Q: int, _cvt: bool = True) -> List[List[ArrayType]]:
    # Pyramid list
    _p = [[]]
    # Converting the image to float[0-1]
    if _cvt:
        _img = np.array(_img, np.float)/255
    # Getting the filters
    _h0 = high_pass(_img.shape, 1/2)
    _l0 = low_pass(_img.shape, 1/2)
    # Storing the high-pass image
    _p[0].append(np.real(IFFT(FFT(_img, _img.shape)*_h0)))
    # Starting with low-pass image
    _low = np.real(IFFT(FFT(_img, _img.shape)*_l0))
    # LOOP
    for _ in range(K):
        # LOW-PASS
        _LF = FFT(_low, _low.shape)
        # Layer list
        _layer = []
        # GETTING THE BANDS
        _G = oriented(_low.shape, Q)
        _H = high_pass(_low.shape)
        # GETTING THE ORIENTED FILTERS, LOOP
        for _i in range(Q):
            _I = _LF*_G[_i]*_H
            _layer.append(np.real(IFFT(_I)))
        # Low-passing further
        _L = low_pass(_low.shape)
        _low = np.real(IFFT(_L*_LF))
        # Downsampling by 2
        _low = _low[::2,::2]
        # Adding _layer to _pyramids
        _p.append(_layer)
    # Adding the latest low to pyramids
    _p.append([_low])
    # Returning the pyramids
    return _p