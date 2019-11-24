

import numpy as np
from typing import Tuple, List
from array import ArrayType
import sys

from mio import *
from gaussian import *

# Getting the neighbourhood of a point
def nb(_shp: Tuple[int, int], _pt: Tuple[int, int], _arr: ArrayType, _tp: bool = True) -> ArrayType:
    # Getting the point
    _x, _y = _pt
    # Getting the shape
    _m, _n = _shp
    # Creating the neighbourhood
    _nb = np.zeros(_shp)
    # Filling, acc. to type
    if _tp:
        if _x+1 >= _m and _y >= _n:
            _nb = _arr[_x-_m+1:_x+1, _y-_n:_y]
        elif _x+1 >= _m:
            _nb[:, _n-_y:_n] = _arr[_x-_m+1:_x+1, 0:_y]
        elif _y >= _n:
            _nb[_m-_x-1:_m, :] = _arr[0:_x+1, _y-_n:_y]
        else:
            _nb[_m-_x-1:_m, _n-_y:_n] = _arr[0:_x+1, 0:_y]
    else:
        if _y+_n >= _arr.shape[1] and _x < _m:
            _nb[_m-_x:_m, :_arr.shape[1]-_y] = _arr[0:_x, _y:_arr.shape[1]]
        elif _y+_n > _arr.shape[1]:
            _nb[:, :_arr.shape[1]-_y] = _arr[_x-_m:_x, _y:_arr.shape[1]]
        elif _x < _m:
            _nb[_m-_x:_m, :] = _arr[0:_x, _y:_y+_n]
        else:
            _nb = _arr[_x-_m:_x, _y:_y+_n]
    # Returning the neighbourhood
    return _nb

# Creating the neighbourhood of a template
def create_nb(_template: ArrayType, _shp: Tuple[int, int] = (4, 4), _tp: bool = True):
    # Gettng the shape
    _m, _n = _template.shape
    # Creating the lambda
    _l = lambda x: nb(_shp, (x//_m, x%_n), _template, _tp)
    # Creating the values
    _x = []
    for i in range(_template.size):
        _x.append(_l(i))
    _x = np.array(_x)
    # Reshaping the values
    _x = _x.reshape((_m, _n, _shp[0], _shp[1]))
    # Returning the values
    return _x

# Matching the best neighbourhoods
def match_nb(_nb: ArrayType, _template_nbs: ArrayType):
    # Creating matchings
    _match = np.linalg.norm(np.linalg.norm(_template_nbs-_nb, axis=3), axis=2)
    # Getting the argmin
    _i = np.argmin(_match)
    # Returning the best match index
    return (_i//_template_nbs.shape[0], _i%_template_nbs.shape[1])

# Matching the best neighbourhoods
def match_nb_linear(_nb: ArrayType, _template_nbs: ArrayType, _rng: int = 0):
    # Creating matchings
    _match = np.linalg.norm(np.linalg.norm(_template_nbs-_nb, axis=2)[:, _rng:], axis=1)
    # Getting the argmin
    _i = np.argmin(_match)
    # Returning the best match index
    return _i

# GLOBAL CONSTANTS
global SIZEX, SIZEY, SHORT, LONG, FINAL, CLIP
SIZEX = 96
SIZEY = 96
LONG = 20
SHORT = 10
FINAL = 150
CLIP = 2000

# Final Runner
if __name__ == "__main__":
    # Read
    x = read_special(sys.argv[1], 'GIF')
    # Clipping
    x = x[:SIZEY, :SIZEX].astype(np.float)/255


    # Horizontal Elongation

    # Parameters
    _shp = (SHORT, LONG)
    # Creating template-neighbourhood
    _xt = create_nb(x, _shp)
    # Linearising the data
    _p = np.unique(np.random.randint(SIZEY, SIZEX*SIZEY, CLIP))
    xx = x[_p%SIZEY, _p//SIZEY]
    _xxt = _xt[_p%SIZEY, _p//SIZEY]
    # Creating the new image
    nx = np.zeros((SIZEY, FINAL))
    nx[:SIZEY, :SIZEX] = x

    # Creating the working lambda
    _i = SIZEX
    __cl = np.vectorize(lambda j: (xx[match_nb_linear(nb(_shp, (j, _i), nx), _xxt)] if j >= SHORT else xx[match_nb_linear(nb(_shp, (j, _i), nx), _xxt, SHORT-j)]))
    _r = np.arange(SIZEY-1, -1, -1, dtype=int)
    # Looping with lambda
    while _i < FINAL:
        print(f'C: {_i}', flush=True)
        nx[_r, _i] = __cl(_r)
        _i += 1


    # Semi-Vertical Elongation

    # Parameters
    _shp = (LONG, SHORT)
    # Creating template-neighbourhood
    _xt = create_nb(x, _shp, False)
    # Linearising the data
    _p = np.unique(np.random.randint(SIZEY, SIZEY*(SIZEX-SHORT), CLIP))
    xx = x[_p%SIZEY, _p//SIZEY]
    _xxt = _xt[_p%SIZEY, _p//SIZEY]
    # Creating the new image
    hx = np.zeros((FINAL, FINAL))
    hx[:SIZEX] = nx

    # Creating working lambda
    _i = SIZEX
    __cl = np.vectorize(lambda j: (xx[match_nb_linear(nb(_shp, (_i, j), hx, False), _xxt)]))
    _r = np.arange(FINAL-SHORT, -1, -1, dtype=int)
    # Looping with lambda
    while _i < FINAL:
        print(f'R: {_i}', flush=True)
        hx[_i, _r] = __cl(_r)
        _i += 1


    # Semi-Horizontal Elongation

    # Parameters
    _shp = (SHORT, LONG)
    # Creating template-neighbourhood
    _xt = create_nb(x, _shp)
    # Linearising the data
    _p = np.unique(np.random.randint(SIZEY, SIZEX*SIZEY, CLIP))
    xx = x[_p%SIZEY, _p//SIZEY]
    _xxt = _xt[_p%SIZEY, _p//SIZEY]

    # Creating the working lambda
    _i = FINAL-SHORT+1
    __cl = np.vectorize(lambda j: (xx[match_nb_linear(nb(_shp, (j, _i), hx), _xxt)]))
    _r = np.arange(SIZEY, FINAL, dtype=int)
    # Looping with lambda
    while _i < FINAL:
        print(f'C2: {_i}', flush=True)
        hx[_r, _i] = __cl(_r)
        _i += 1

    # Writing the image
    try:
        write(sys.argv[2], hx*512)
    except IndexError:
        write('final.png', hx*512)