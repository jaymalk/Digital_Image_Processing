# Program for reading HDR images and mapping them to a JPEG

import sys
import cv2 as cv
import imageio
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

# Read an HDR image
def read(name : str):
    try:
        return imageio.imread(name, format='HDR-FI')
    except:
        trace()

# Write an jpeg image
def write(name : str, img : np.array):
    try:
        cv.imwrite(name, img[:,:,::-1])
    except:
        trace()

# Gamma Correction
def gamma_crr(img : np.array, gm : float = 1/2.2):
    try:
        return np.power(img, gm)
    except:
        trace()

# Linear Scaling
def scale(img : np.array, mx : int, mn : int = 0):
    try:
        img_scl = mn + ((mx-mn)/(img.max()-img.min())) * (img - img.min())
        return img_scl
    except:
        trace()

# Logarithmic-Luminance Scaling
def log_lum(img : np.array, base : int = 2, bmx : float = 3, bmn : float = 1, al : float = 0.3):
    try:   
        _delta = 0.00001
        img = scale(img, 255)
        # Seperating RGB Components
        imgr = np.array(img[:,:,0])         
        imgb = np.array(img[:,:,2])         
        imgg = np.array(img[:,:,1])
        # Getting Luminance
        lum = imgr*0.299 + imgg*0.587 + imgb*0.114 
        # Going to log-domain
        log_lum = np.log(lum+_delta)/np.log(base)
        # Scaling in log-domain
        log_lum = scale(log_lum, bmx, bmn)
        # Getting out of log-domain
        new_lum = np.power(base, log_lum) - _delta
        new_lum = scale(new_lum, 255, 1)
        # Getting new RGB components 
        imgr = new_lum*((imgr/(lum+_delta))**al) 
        imgb = new_lum*((imgb/(lum+_delta))**al) 
        imgg = new_lum*((imgg/(lum+_delta))**al)
        # Creating new image
        new_img = cv.merge([imgr, imgg, imgb])
        # Returning image
        return scale(new_img, 255)
    except:
        trace()

# Main
# Completes Part-1 of Assignment
if __name__ == '__main__':
    try:
        # Reading image and correcting it with gamma correction
        img = gamma_crr(read(sys.argv[1]))
        # Linearly Scaling
        write('img_max__ign.jpg', scale(img, 255, 0)) # Fitting all in range (lower values are lost)
        write('img_middle__ign.jpg', scale(img, 4000, 0)) # Middle fitting (decent)
        write('img_min__ign.jpg', scale(img, 10000, 0)) # Fitting lower values (higher are lost)
        # Log Scaling with various parameters
        write('log_set1__ign.jpg', log_lum(gamma_crr(img, 2.2), base=2, bmn=1, bmx=3, al=0.45)) # Setting - 1 (Default)
        write('log_set2__ign.jpg', log_lum(img, base=10, bmn=0.1, bmx=1.4, al=0.9)) # Setting - 2
        write('log_set3__ign.jpg', log_lum(gamma_crr(img, 2.2), base=2, bmn=0.5, bmx=4.5, al=0.45)) # Setting - 3
    except:
        trace()
