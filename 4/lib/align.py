

import cv2 as cv
import numpy as np
from subprocess import run


from mio import *

global TOTAL_FEATURES
TOTAL_FEATURES = 14000
BEST_PERCENT = 0.005


def image_alignment(_img_to_align: np.ndarray, _img_template: np.ndarray, _gray: bool = False) -> np.ndarray:
    # Checking if image is in gray_scale or colored format
    if not _gray:
        _g1 = cv.cvtColor(_img_to_align, cv.COLOR_BGR2GRAY)
        _g2 = cv.cvtColor(_img_template, cv.COLOR_BGR2GRAY)
    else:
        _g1 = _img_to_align.copy()
        _g2 = _img_template.copy()
    # Getting keypoints and descriptors for the images
        # Creating the ORB-Descriptor Instance
    __orb = cv.ORB_create(TOTAL_FEATURES)
        # Getting the features
    _kp1, _dp1 = __orb.detectAndCompute(_g1, None)
    _kp2, _dp2 = __orb.detectAndCompute(_g2, None)
    # Matching the features
    _matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
    _all_matches = _matcher.match(_dp1, _dp2, None)
    # Filtering the best matches
        # First sorting on the basis of feature distances
    _all_matches.sort(key=lambda x: x.distance, reverse=False)
        # Now getting the best matches complying with BEST_PERCENT
    _cnt = int(len(_all_matches) * BEST_PERCENT)
    _final = _all_matches[:_cnt]
    # Getting the positions of all the good matches
    _p1 = np.zeros((len(_final), 2), dtype=np.float32)
    _p2 = np.zeros((len(_final), 2), dtype=np.float32)
    # Extracting the positions
    for _i, _m in enumerate(_final):
        _p1[_i,:] = _kp1[_m.queryIdx].pt
        _p2[_i,:] = _kp2[_m.trainIdx].pt
    # Finding Homography
    _hom, _ = cv.findHomography(_p1, _p2, cv.RANSAC)
    # Applying homographic transform (affine)
    _h, _w = _img_template.shape[:2]
    _alinged_img = cv.warpPerspective(_img_to_align, _hom, (_w, _h))
    # Return the aligned image
    return _alinged_img

# _template = read('./__main.jpg', False)
# _mask = read('./best.png', False)
# _path = '../all-images/booklets/'
# for _i in run(['ls', _path], capture_output=True).stdout.decode().rstrip().split():
#     _p = _path+_i
#     _img = read(_p)
#     _img = cv.cvtColor(_img, cv.COLOR_GRAY2BGR)
#     TOTAL_FEATURES = 14500
#     try:
#         m = image_alignment(_img, _template)
#         o = np.where(_mask == 0, 0, m)
#         show(o)
#     except:
#         pass