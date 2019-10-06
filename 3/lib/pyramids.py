'''
Program for evaluating various kinds of pyramids
'''

import numpy as np
import sys
from typing import List, Tuple

from mio import trace


# Reduce Function
def reduce_(_arr: np.ndarray, _a: float = 0.4) -> np.ndarray:
    '''
    Function for reducing the image, each dimension by 2.\n
    @param _arr : input array (2D)\n
    @param _a (optional) : the parameter for reduction satisfying : a+2*b+2*c=1 && b=0.25
    '''
    # Setting parameters
    _b = 0.25
    _c = (0.5-_a)/2
    # Shrinking first dimension
    _len = (_arr.shape[0]+1)//2
    __temp = np.ones((_len, _arr.shape[1]))
    # Working with the starting index
    __temp[0] = _a*_arr[0] + 2*_arr[1]*_b + 2*_arr[2]*_c
    # Working with intermediate index
    for _i in np.arange(2, _arr.shape[0]-2, 2):
        __temp[_i//2] = _c*_arr[_i+2] + _b*_arr[_i+1] + _a*_arr[_i] + _b*_arr[_i-1] + _c*_arr[_i-2]
    # Working with ending index
    if _len%2==1: __temp[_len-1] = _a*_arr[-1] + 2*_arr[-2]*_b + 2*_arr[-3]*_c
    else: __temp[_len-1] = _a*_arr[-2] + _arr[-3]*_b + _arr[-1]*_b + 2*_arr[-4]*_c
    # Shrinkng other dimension
    _len = (__temp.shape[1]+1)//2
    __temp2 = np.ones((__temp.shape[0], _len))
    # Working with starting index
    __temp2[:,0] = _a*__temp[:,0] + 2*__temp[:,1]*_b + 2*__temp[:,2]*_c
    # Working with intermediate index
    for _i in np.arange(2, __temp.shape[1]-2, 2):
        __temp2[:,_i//2] = _c*__temp[:,_i+2] + _b*__temp[:,_i+1] + _a*__temp[:,_i] + _b*__temp[:,_i-1] + _c*__temp[:,_i-2]
    # Working with ending index
    if _len%2 == 1: __temp2[:,_len-1] = _a*__temp[:,-1] + 2*__temp[:,-2]*_b + 2*__temp[:,-3]*_c
    else: __temp2[:,_len-1] = _a*__temp[:,-2] + __temp[:,-3]*_b + __temp[:,-1]*_b + 2*__temp[:,-4]*_c
    # Returning the array
    return __temp2


# Gaussian Pyramids
def gaussian_pyramid(_img: np.ndarray, _a: float = 0.4) -> List[np.ndarray]:
    '''
    Function that returns the gaussian pyramid of the input 2D array\n
    @param _img : input array (2D np.ndarray)\n
    @param _a (optional) : the parameter for reduction satisfying : a+2*b+2*c=1 && b=0.25\n
    @return List[np.ndarray] : list of np.ndarrays (pyramids)
    '''
    # Creating the pyramid
    _gaussian = [_img]
    # Loop
    while True:
        # Reduce
        _new = reduce_(_gaussian[-1], _a)
        # Store
        _gaussian.append(_new)
        # Check
        if _new.shape[0] <= 3 or _new.shape[1] <= 3:
            break
    # Returning the pyramid
    return _gaussian


# Expand function
def expand_(_arr: np.ndarray, _a: float = 0.4) -> np.ndarray:
    '''
    Function for expanding the image, each dimension by 2.\n
    @param _arr : input array (2D)\n
    @param _a (optional) : the parameter for reduction satisfying : a+b+c=1 && b=0.5
    '''
    # Setting parameters
    _b = 0.25
    _c = (0.5-_a)/2
    # Doubling the values
    _a, _b, _c = 2*_a, 2*_b, 2*_c
    # Expanding first dimension
    __temp = np.zeros((2*_arr.shape[0], _arr.shape[1]))
    # Setting initial coordinate
    __temp[2] += _c*_arr[0]
    __temp[1] += _b*_arr[0]
    __temp[0] += _a*_arr[0]
    # Setting intermediate coordinates
    for i in range(1, _arr.shape[0]-1):
        __temp[2*i+2] += _c*_arr[i]
        __temp[2*i+1] += _b*_arr[i]
        __temp[2*i] += _a*_arr[i]
        __temp[2*i-1] += _b*_arr[i]
        __temp[2*i-2] += _c*_arr[i]
    # Setting ending coordinate
    __temp[-1] = 2*_b*_arr[-1]
    __temp[-2] += (_a+_c)*_arr[-1]
    __temp[-3] += _b*_arr[-1]
    __temp[-4] += _c*_arr[-1]
    # Expanding second dimension
    __temp2 = np.zeros((2*_arr.shape[0], 2*_arr.shape[1]))
    # Setting initial values
    __temp2[:,2] += _c*__temp[:,0]
    __temp2[:,1] += _b*__temp[:,0]
    __temp2[:,0] += _a*__temp[:,0]
    # Setting intermediate coordinates
    for i in range(1, _arr.shape[1]-1):
        __temp2[:,2*i+2] += _c*__temp[:,i]
        __temp2[:,2*i+1] += _b*__temp[:,i]
        __temp2[:,2*i] += _a*__temp[:,i]
        __temp2[:,2*i-1] += _b*__temp[:,i]
        __temp2[:,2*i-2] += _c*__temp[:,i]
    # Setting eending coordinates
    __temp2[:,-1] = 2*_b*__temp[:,-1]
    __temp2[:,-2] += (_a+_c)*__temp[:,-1]
    __temp2[:,-3] += _b*__temp[:,-1]
    __temp2[:,-4] += _c*__temp[:,-1]
    # Returning the array
    return __temp2


# Laplacian Pyramids
def laplacian_pyramids(_img: np.ndarray, _a: float = 0.4) -> List[np.ndarray]:
    '''
    Function that returns the laplacian pyramid of the input 2D array\n
    @param _img : input array (2D np.ndarray)\n
    @param _a (optional) : the parameter for reduction satisfying : a+2*b+2*c=1 && b=0.25\n
    @return List[np.ndarray] : list of np.ndarrays (pyramids)
    '''
    # Creating the laplacian
    _laplacian = []
    # Creating the gaussian
    __g = gaussian_pyramid(_img, _a)
    # Loop
    for _i in range(len(__g)-1):
        _exp = expand_(__g[_i+1], _a)
        if __g[_i].shape[0]%2 == 1: _exp = _exp[:-1]
        if __g[_i].shape[1]%2 == 1: _exp = _exp[:,:-1]
        assert(_exp.shape == __g[_i].shape)
        _laplacian.append(__g[_i] - _exp)
    # Adding the last gaussian
    _laplacian.append(__g[-1])
    # Returning 
    return _laplacian


# Recreating the image from pyraamids
def recreate_(_lap: List[np.ndarray], _a: float = 0.4) -> np.ndarray:
    '''
    Recreating the original image from laplacian pyramids\n
    @param _lap : laplacian pyramid\n
    @param _a (optional) : the parameter for reduction satisfying : a+2*b+2*c=1 && b=0.25\n
    @return np.ndarrray, the original 2D image
    '''
    # First scalling coefficient
    _mn = _lap[-1]
    # Looping to get features
    for _i in range(1,len(_lap)):
        # Getting the feature
        _k = _lap[-1-_i]
        # Expanding the present image
        _exp = expand_(_mn, _a)
        if _k.shape[0]%2 == 1: _exp = _exp[:-1]
        if _k.shape[1]%2 == 1: _exp = _exp[:,:-1]
        # Inducing the feature
        _mn = _k + _exp
    # Return image
    return _mn


# Laplacian blending
def blend_(_img1: np.ndarray, _img2: np.ndarray, _blnd: np.ndarray, _a: float = 0.4) -> np.ndarray:
    '''
    Blending two images using a distribution mask, via laplacian blending\n
    Important: sizes of all the parameters must be the same\n
    @param _img1 : First image\n
    @param _img2 : Second image\n
    @param _blnd : Blender mask\n
    @return np.ndarray, the blended image
    '''
    try:
        assert(_img1.shape == _img2.shape == _blnd.shape)
        # Creating pyramids
        _l1 = laplacian_pyramids(_img1, _a)
        _l2 = laplacian_pyramids(_img2, _a)
        _g = gaussian_pyramid(_blnd, _a)
        # Starting with base blend
        _mn = _l1[-1]*(_g[-1]) + _l2[-1]*(1-_g[-1])
        # Looping on blending features
        for _i in range(1, len(_l1)):
            # Blending features
            _k = _l1[-1-_i]*(_g[-1-_i]) + _l2[-1-_i]*(1-_g[-1-_i])
            # Expanding the present blend
            _exp = expand_(_mn)
            if _k.shape[0]%2 == 1: _exp = _exp[:-1]
            if _k.shape[1]%2 == 1: _exp = _exp[:,:-1]
            # Inducing the feature
            _mn = _k + _exp
        # Return final blend
        return _mn
    # Assertion Error
    except AssertionError:
        print(f"Assertion failed, all images must be of the same size.", file=sys.stderr)
    # General Exception
    except:
        trace()


# Laplacian Denoise
def pyramid_denoise(_img: np.ndarray, _ctof: float, _rng: int = -1):
    __l = laplacian_pyramids(_img)
    __n = len(__l) if _rng == -1 else _rng
    for _i in range(__n):
        __l[-_i-1] = np.where(abs(__l[-1-_i]) < _ctof, 0, __l[-1-_i])
    return recreate_(__l)
