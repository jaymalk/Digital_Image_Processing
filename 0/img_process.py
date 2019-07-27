# Assignment 0
# Processing an image to produce an image with averaged rows

# Uses 'python 3.x'
# Dependencies 'sys' 'cv2' 'numpy' 'argparse'

# Rajbir Malik
# 2017CS10416

import numpy as np
import cv2 as cv
import argparse
import sys


# Function for command line arguments
def set_parser():
    # Creating a parser
    parser = argparse.ArgumentParser(
            description="Assignment-0 (Command Line Arguments)")
    # Adding -in
    parser.add_argument('-in', action="store", dest="_in", default=None, type=str, help='Input-file relative path.')
    # Adding -out
    parser.add_argument('-out', action="store", dest="_out", default=None, type=str, help='Output-file relative path.')
    return parser


# Processing Function
def process(img):
    img = np.array(img, dtype=np.uint64)
    # Slow, single line solution
    new_img = []
    row_len = img.shape[1]
    for row in img:
        avg = row.sum(axis=0)/row_len
        new_img.append([avg]*row_len)
    return np.array(new_img)


# Main
if __name__ == '__main__':

    # Setting up parser
    parser = set_parser()

    # Main program
    try:

        # Parsing
        parse = parser.parse_args(sys.argv[1:])

        # Reading
        if parse._in:
            img = cv.imread(parse._in)
        else:
            img = cv.imread(input('Enter relative file path : '))

        # Function (Processing)
        img = process(img)

        # Writing
        if parse._out:
            img = cv.imwrite(parse._out, img)
        else:
            print('Image written at relative path ./final.jpeg')
            cv.imwrite('./final.jpeg', img)

    # Catching exceptions (while reading, writing...)
    except Exception as _e:
        print(_e.__doc__())



