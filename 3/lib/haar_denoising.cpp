/*
 * Performing denoising by clipping components from haar-transform of the image
 */


#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "image_io.hpp"
#include "haar.hpp"


using namespace std;
using namespace cv;


/*
 * Denoising using the haar-transformed matrix (removing small quantities)
 * @param _matrix : transform matrix
 * @param _k : threshold
 * @param _smth : smooth removal / hard removal
 */
void denoise(Mat& _matrix, float _k, bool smooth = true) {
    int _col, _row;
    _row = _matrix.size().height;
    _col = _matrix.size().width;
    // Thresholding the values
    // Hard
    for(size_t i{}; i<_row; i++)
        for(size_t j{}; j<_col; j++)
            if(smooth) {
                if(_matrix.at<float>(i, j) > _k) _matrix.at<float>(i, j) -= _k;
                if(_matrix.at<float>(i, j) < -_k) _matrix.at<float>(i, j) -= -_k;
            }
            else {
                if(abs(_matrix.at<float>(i, j)) < _k)
                    _matrix.at<float>(i, j) = 0;
            }
}


// Main
int main(int argc, char const *argv[])
{
    // Reader
    string rdpath;
    float _k;
    bool _c;
    switch(argc) {
        case 1: {
            cout << "Enter file path: ";
            cin >> rdpath;
            cout << "Enter cutoff: ";
            cin >> _k;
            cout << "Enter denoising type [0/1]: ";
            cin >> _c;
            break;
        }
        case 2: {
            rdpath = argv[1];
            cout << "Enter cutoff: ";
            cin >> _k;
            cout << "Enter denoising type [0/1]: ";
            cin >> _c;
            break;
        }
        case 3: {
            rdpath = argv[1];
            _k = stof(argv[2]);
            cout << "Enter denoising type [0/1]: ";
            cin >> _c;
            break;
        }
        case 4: {
            rdpath = argv[1];
            _k = stof(argv[2]);
            _c = (bool)stoi(argv[3]);
            break;
        }
        default: {
            cerr << "Invalid arguments\n";
            exit(1);
        }
    }

    //!!!!!!!!!!!!!!!
    // Containers
    Mat _img;

    // Loading and processing
    load_image(rdpath, _img);
    _img.convertTo(_img, CV_32F);
    _img /= 255;

    //! Process
    // Transform
    haar_transform(_img, _img.size().height, _img.size().width);
    // Denosing
    denoise(_img, _k, _c);
    // Inverse
    inverse_haar_transform(_img, _img.size().height, _img.size().width);
    // Saving
    save_image("./" + to_string(_c) + "_"+to_string(_k)+".jpg", _img);

    return 0;
}
