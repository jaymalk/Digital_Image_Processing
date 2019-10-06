/*
 * Library for lossy compression/decompression of image files
 */

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "image_io.hpp"
#include "haar.hpp"

void write_haar(Mat& img) {
    img *= 1000;
    int *_tab = new int[2001]();
    for(size_t i{}; i<img.size().height; i++)
        for(size_t j{}; j<img.size().width; j++)
            _tab[(int)round(img.at<float>(i, j))+1000]++;
    for(int i=0; i<2001; i++) cout << i-1000 << ":" << _tab[i] << "\n";
    free(_tab);
}

void cutoff_reduce(Mat& img, int k) {
    // Setting the cutoff based on k and maximum
    float _cut = (float)k/100.0;
    for(size_t i{}; i<img.size().height; i++)
        for(size_t j{}; j<img.size().width; j++) {
            if(abs(img.at<float>(i, j)) < _cut)
                img.at<float>(i, j) = 0;
        }
}

// ! RUN LENGTH ENCODING

vector<pair<uchar, int>> runlength_encode(Mat& img) {
    // First attempt
    float _max = img.at<float>(0,0);
    img /= (_max+1);
    img *= 128;
    img += 128;
    img.convertTo(img, CV_8UC1);
    // Coding now
    vector<pair<uchar, int>> _code;
    uchar _cval = img.at<uchar>(0,0);
    int _run = 1;
    for(size_t i{}; i<img.size().height; i++)
        for(size_t j{}; j<img.size().width; j++)
            if(img.at<uchar>(i,j) == _cval) {
                _run++;
            }
            else {
                _code.push_back({_cval, _run});
                _cval = img.at<uchar>(i,j);
                _run = 1;
            }
    return _code;
}

void write_run_length(Size _size, vector<pair<uchar, int>> _code, string _fname = "./save.bin") {
    // Opening the binary file
    ofstream out(_fname, ios::binary | ios::out);
    // Writing the size of the image
    short _ht = _size.height, _wd = _size.width;
    out.write(reinterpret_cast<const char *>(&_ht), sizeof(short));
    out.write(reinterpret_cast<const char *>(&_wd), sizeof(short));
    // Writing the length of the encoded scheme
    int _len = _code.size();
    out.write(reinterpret_cast<const char *>(&_len), sizeof(int));
    // Writing the run length encoded image
    for(auto& _p : _code)
        out.write(reinterpret_cast<const char *>(&_p), sizeof(_p));
    // Closing the file
    out.close();
}

int main(int argc, char const *argv[])
{
    string path = argv[1];
    int k = stoi(argv[2]);
    Mat img;
    // Loading the image
    load_image(path, img, true);
    // Taking the haar transform of the image
    haar_transform(img, img.size().height, img.size().width);
    // Thresholding the image
    cutoff_reduce(img, k);
    // Taking the inverse transform
    inverse_haar_transform(img, img.size().height, img.size().width);
    // Taking the de-normalised transform
    haar_transform(img, img.size().height, img.size().width, false);
    // Creating a run length encoded form of the image
    vector<pair<uchar, int>> _encoded = runlength_encode(img);
    // Writing the binary image
    write_run_length(img.size(), _encoded);
    // Return
    return 0;
}
