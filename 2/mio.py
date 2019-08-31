#!/anaconda3/bin/python

import sys
import cv2 as cv
import numpy as np


# Exception trace printer
def trace():
    # Exception objects
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print("Exception occured.")
    # Type
    print("Type : " + exc_type.__name__)
    # Line no. of exception
    print("Line no. : " + str(exc_tb.tb_lineno))
    # Exception contains message (optional)
    if exc_obj:
        print("Message : " + str(exc_obj))
    # Exiting the code
    sys.exit(2)


# Image viewer
# @param _img : image to display
# @param _title : window name
def show(_img, title='Image'):
    cv.imshow(title, _img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)


# Reading an image (B/W)
# @param _name : relative path of the file
# @param _bw : whether the image is b/w or rgb
def read(_name: str, _bw: bool = True) -> np.array:
    try:
        # Read
        _img = cv.imread(_name, 0) if _bw else cv.imread(_name)
        # Image invalid
        if _img is None:
            raise FileNotFoundError('No image with name ' + _name)
        # Return
        return _img
    except:
        trace()


# Writing an image (B/W)
# @param _name : relative write path
# @param _img : image to be written
def write(_name: str, _img: np.array) -> None:
    try:
        # Write
        _img = cv.imwrite(_name, _img)
    except:
        trace()
