/*
 * Basic Image Input/Output Functions
 */


#ifndef __IO__
#define __IO__

#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/*
 * Displaying image on a window
 * @param _img : image to be displayed
 * @param _name : window name (default "Image")
 */
void show_image(cv::Mat& _img, std::string _name = "Image") {
    // Creating the window
    cv::namedWindow(_name);
    // Displaying the image
    cv::imshow(_name, _img);
    // Waiting for key input
    cv::waitKey(0);
    // Destroying the window
    cv::destroyWindow(_name);
}


/* 
 * Loading image (using relative path)
 * @param _name : relative path of the image to be uploaded
 * @param _img : const reference to loading array
 */
void load_image(std::string _name, cv::Mat& _img, bool _make_float = false) {
    try {
        // Loading the image on the reference
        _img = cv::imread(_name, 0);
        // Exiting if image doesn't exist
        if(_img.empty())
            throw std::runtime_error("File Doesn't exist\nImage is empty\nExiting...\n");
        // Checking if conversion needed
        if(_make_float) {
            _img.convertTo(_img, CV_32F);
            _img /= 255;
        }
    }
    catch(const std::runtime_error& e) {
        // Image doesn't exist
        std::cerr << e.what();
        exit(1);
    }
}


/* 
 * Saving image (at relative path)
 * @param _name : relative path where the image is saved
 * @param _img : const reference to img array
 */
void save_image(std::string _name, cv::Mat& _img) {
    try {
        // Saving the image on the reference
        cv::imwrite(_name, 255*_img);
    }
    catch(const std::runtime_error& e) {
        // Image doesn't exist
        std::cerr << e.what();
        exit(1);
    }
}


#endif
