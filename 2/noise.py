#!/anaconda3/bin/python

import sys
import cv2 as cv
import numpy as np
from mio import write, read, trace


# Function to add noise to an image
# @param _img : original image
# @param _type : type of the noise
def add_noise(_img: np.array, _type: str, _rng: int = 50) -> np.array:
    try:
        # Salt noise
        if _type == 'S':
            _n_img = np.where(np.random.random(_img.shape) > 0.80, 255, _img)
        # Pepper noise
        elif _type == 'P':
            _n_img = np.where(np.random.random(_img.shape) > 0.80, 0, _img)
        # Salt and pepper noise
        elif _type == 'SP':
            _n_img = np.where(np.random.random(_img.shape) > 0.9, 255, _img)
            _n_img = np.where(np.random.random(_img.shape) > 0.9, 0, _n_img)
        # Gaussian Noise
        elif _type == 'G':
            _n_img = _img + np.random.normal(0, _rng, _img.shape)
            _n_img = np.where(_n_img < 0, 0, _n_img)
            _n_img = np.where(_n_img > 255, 255, _n_img)
            _n_img = np.array(_n_img, dtype=np.uint8)
        # Uniform Noise
        elif _type == 'U':
            _n_img = _img + np.random.uniform(-_rng/2, _rng/2, _img.shape)
            _n_img = np.where(_n_img < 0, 0, _n_img)
            _n_img = np.where(_n_img > 255, 255, _n_img)
            _n_img = np.array(_n_img, dtype=np.uint8)
        # Error (Inavlid)
        else:
            raise ValueError('No available noise type ' + _type)
        # Return
        return _n_img
    except:
        trace()

# Main Module
# ARG
# read_name : argv[1]
# write_name : argv[1]
# noise_type : argv[1]
if __name__ == "__main__":
    try:
        # Read Name
        _rname = sys.argv[1]
        # Write Name
        _wname = sys.argv[2]
        # Noise  Type
        _noise_type = sys.argv[3]
        # Setting Range
        try:
            _rng = int(sys.argv[4])
        except:
            _rng = 50
        # Completing the task
        write(_wname, add_noise(read(_rname), _noise_type, _rng))
    except:
        trace()
