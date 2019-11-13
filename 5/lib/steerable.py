# Library for constructing the steerable pyramids of an image

from typing import Tuple, List
from array import ArrayType
import numpy as np

from mio import *
from fft import *
from fft import __pad

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
                if abs(_t-_2) <= np.pi/2: 
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
        _low = (_low[::2,::2] + _low[::2,1::2] + _low[1::2,::2] + _low[1::2,1::2])/4
        # Adding _layer to _pyramids
        _p.append(_layer)
    # Adding the latest low to pyramids
    _p.append([_low])
    # Returning the pyramids
    return _p


# Recreating the image from pyramids
def recreate(_p: List[List[ArrayType]]) -> ArrayType:
    # Getting the parameters
    K = len(_p)-2; assert(K > 0)    # Atleast 1-scale in the image
    Q = len(_p[1])
    # Getting the first low-scale
    _low = _p[-1][0]
    # LOOP, with oriented versions
    for _i in range(K):
        # Up-sampling the present LOW
        _low_up = np.zeros((_low.shape[0]*2, _low.shape[1]*2))
        _low_up[::2,::2] = _low
        _low_up[1::2,::2] = _low
        _low_up[::2,1::2] = _low
        _low_up[1::2,1::2] = _low
        # Going to Fourier Domain
        _LF = FFT(_low_up, _low_up.shape)
        # Convolving with a low-pass band
        _L = low_pass(_LF.shape)
        _LF = _LF*_L
        # Working with the oriented versions now
        _G = oriented(_LF.shape, Q)
        _H = high_pass(_LF.shape)
        # Looping over oriented filters
        for _j in range(Q):
            _I = FFT(_p[-_i-2][_j], _p[-_i-2][_j].shape) * _H * _G[_j]
            _LF += _I
        # Going back to spatial-domain
        _low = np.real(IFFT(_LF))
    # Now, finally processing the _low with _l0, and then adding _h0 processing high-pass
    _l0 = low_pass(_low.shape, 1/2)
    _LF *= _l0
    _h0 = high_pass(_low.shape, 1/2)
    _HF = _h0 * _p[0][0]
    # Adding the get the image back
    _I = _LF + _HF
    return np.real(IFFT(_I))


# ----------------------------------------
# FASTER ROUTINES
# ----------------------------------------


# Faster Low-Pass Filter
def low_pass_fast(_shp: Tuple[int, int], _scl: float = 1) -> ArrayType:
    # Creating the matrix
    _l = np.zeros(_shp, dtype=complex)
    # Getting the parameters
    _m, _n = _shp
    # Getting the mid ranges
    _mx, _my = _m//2, _n//2
    # Getting the mesh
    _x, _y = np.meshgrid(np.arange(-_mx, _mx), -np.arange(-_my+1, _my+1))
    # Setting the distance
    _r = 2 * np.pi * np.sqrt((_x/_m)**2 + (_y/_n)**2)
    # Scaling the mesh
    _r *= _scl
    # Setting values using the mesh
    _l[_r <= np.pi/4] = 1
    _l[(np.pi/4 < _r)*(_r <= np.pi/2)] = np.cos((np.pi/2) * np.log2(4 * _r[(np.pi/4 < _r)*(_r <= np.pi/2)] / np.pi))
    # Returning the mesh
    return _l


# Faster High-Pass Filter
def high_pass_fast(_shp: Tuple[int, int], _scl: float = 1) -> ArrayType:
    # Creating the matrix
    _h = np.ones(_shp, dtype=complex)
    # Getting the parameters
    _m, _n = _shp
    # Getting the mid ranges
    _mx, _my = _m//2, _n//2
    # Getting the mesh
    _x, _y = np.meshgrid(np.arange(-_mx, _mx), -np.arange(-_my+1, _my+1))
    # Setting the distance
    _r = 2 * np.pi * np.sqrt((_x/_m)**2 + (_y/_n)**2)
    # Scaling the mesh
    _r *= _scl
    # Setting values using the mesh
    _h[_r <= np.pi/4] = 0
    _h[(np.pi/4 < _r)*(_r <= np.pi/2)] = np.cos((np.pi/2) * np.log2(2 * _r[(np.pi/4 < _r)*(_r <= np.pi/2)] / np.pi))
    # Returning the mesh
    return _h


# Getting the oriented filters faster
def oriented_fast(_shp: Tuple[int, int], Q: int) -> List[ArrayType]:
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
        # Getting the mesh grid
        _x, _y = np.meshgrid(np.arange(-_mx, _mx), -np.arange(-_my+1, _my+1))
        # Setting the angles
        _t = np.arctan((_y/_n)/(_x/_m + np.finfo(float).eps))
        # Setting the vales with angles
        P[abs(_t-_1) <= np.pi/2] += _alpha * (np.cos(_t[abs(_t-_1) <= np.pi/2]-_1)**(Q-1)) 
        P[abs(_t-_2) <= np.pi/2] += _alpha * (np.cos(_t[abs(_t-_2) <= np.pi/2]-_2)**(Q-1)) 
        # Appending
        _list.append(P)                                         
    # Return the list
    return _list


# Getting the pyramids fast
def pyramids_fast(_img: ArrayType, K: int, Q: int, _cvt: bool = True) -> List[List[ArrayType]]:
    # Image parameters
    _m, _n = _img.shape
    # Pyramid list
    _p = [[]]
    # Converting the image to float[0-1]
    if _cvt:
        _img = np.array(_img, np.float)/255
    # Getting the filters
    _h0 = high_pass_fast(_img.shape, 1/2)
    _l0 = low_pass_fast(_img.shape, 1/2)
    # Getting image FFT
    _IF = FFT(_img, _img.shape)
    # Storing the high-pass image
    _p[0].append(np.real(IFFT(_IF*_h0)))
    # Starting with low-pass image
    _LF = _IF*_l0
    # LOOP
    for _ in range(K):
        # Layer list
        _layer = []
        # GETTING THE BANDS
        _G = oriented_fast(_LF.shape, Q)
        _H = high_pass_fast(_LF.shape)
        # GETTING THE ORIENTED FILTERS, LOOP
        for _i in range(Q):
            _I = _LF*_G[_i]*_H
            _layer.append(np.real(IFFT(_I)))
        # Low-passing further
        _L = low_pass_fast(_LF.shape)
        _LF = _L*_LF
        # Downsampling by 2
        _m /= 2; _n /= 2
        _LF = _LF[int(_m//2):int(_m+_m//2),int(_n//2):int(_n+_n//2)]
        # Adding _layer to _pyramids
        _p.append(_layer)
    # Adding the latest low to pyramids
    _p.append([np.real(IFFT(_LF))])
    # Returning the pyramids
    return _p

def recreate_fast(_p: List[List[ArrayType]]) -> ArrayType:
    # Getting the parameters
    K = len(_p)-2; assert(K > 0)    # Atleast 1-scale in the image
    Q = len(_p[1])
    # Getting the first low-scale
    _LF = FFT(_p[-1][0], _p[-1][0].shape)
    # LOOP, with oriented versions
    for _i in range(K):
        # Up-sampling the present LOW
        _m, _n = _LF.shape[0], _LF.shape[1]
        _LFU = np.zeros((_m*2, _n*2), dtype=complex)
        _LFU[int(_m//2):int(_m+_m//2),int(_n//2):int(_n+_n//2)] = _LF
        # Resize
        _LF = _LFU
        # Convolving with a low-pass band
        _L = low_pass_fast(_LF.shape)
        _LF = _LF*_L
        # Working with the oriented versions now
        _G = oriented_fast(_LF.shape, Q)
        _H = high_pass_fast(_LF.shape)
        # Looping over oriented filters
        for _j in range(Q):
            _I = FFT(_p[-_i-2][_j], _p[-_i-2][_j].shape) * _H * _G[_j]
            _LF += _I
    # Now, finally processing the _low with _l0, and then adding _h0 processing high-pass
    _l0 = low_pass_fast(_LF.shape, 1/2)
    _LF *= _l0
    _h0 = high_pass_fast(_LF.shape, 1/2)
    _HF = _h0 * _p[0][0]
    # Adding the get the image back
    _I = _LF + _HF
    return np.real(IFFT(_I))