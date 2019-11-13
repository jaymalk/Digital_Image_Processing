# Library for histogram related functions

import numpy as np


# Histogram Matching
def match_hist(_src, _temp):
    # Setting the parameters
    _tp = _src.dtype
    _shp = _src.shape
    # Ravelling both source and template
    _s = _src.ravel()
    _t = _temp.ravel()
    # Getting the countsn
    _sval, _inv, _scnt = np.unique(_s, return_inverse=True, return_counts=True)
    _tval, _tcnt = np.unique(_t, return_counts=True)
    # Creating the source CDF
    _cdf_s = np.cumsum(_scnt).astype(np.float64)
    _cdf_s /= _cdf_s[-1]
    # Creating the template CDF
    _cdf_t = np.cumsum(_tcnt).astype(np.float64)
    _cdf_t /= _cdf_t[-1]
    # Matching the values, for getting the map
    interp_t_values = np.interp(_cdf_s, _cdf_t, _tval)
    interp_t_values = interp_t_values.astype(_tp)
    # Return the mapped image
    return interp_t_values[_inv].reshape(_shp)