/*
 * Library of functions for Lifting Wavelet-Transforms
 */

#ifndef __LIFT__
#define __LIFT__

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/*
 *
 *
 */
Mat lazy_wavelet_reduce(Mat _arr, short _len) {
    
    // Checking the assumption, the _len is a power of 2
    assert((_len & (_len-1)) == 0);
    
    // Base case
    if (_len <= 2) return _arr;

    // Creating a temporary memory container
    float *__temp = new float[_len];
    for(int _i=0; _i < _len; _i++) __temp[_i] = _arr.at<float>(0, _i);

    // Performing the lazy transform on the array
    for(int _i=1; _i < _len-1; _i+=2)
        __temp[_i] -= (0.5)*(__temp[_i-1]+__temp[_i+1]);
    __temp[_len-1] -= (0.5)*(__temp[_len-2]+__temp[0]);

    // Performing the final transform
    for(int _i=2; _i < _len; _i+=2)
        __temp[_i] += (0.25)*(__temp[_i-1]+__temp[_i+1]);
    __temp[0] += (0.25)*(__temp[1]+__temp[_len-1]);

    // Setiing the values in _arr
    for(int _i=0; _i < _len; _i++) _arr.at<float>(0, _i) = __temp[_i];

    // Return the created array
    return _arr;
}


/*
 *
 * 
 */
Mat lazy_wavelet_inverse(Mat _arr, short _len) {
    
    // Checking the assumption, the _len is a power of 2
    assert((_len & (_len-1)) == 0);
    
    // Base case
    if (_len <= 2) return _arr;

    // Creating a temporary memory container
    float *__temp = new float[_len];
    for(int _i=0; _i < _len; _i++) __temp[_i] = _arr.at<float>(0, _i);

    // Reversing the final transform
    __temp[0] -= (0.25)*(__temp[1]+__temp[_len-1]);
    for(int _i=2; _i < _len; _i+=2)
        __temp[_i] -= (0.25)*(__temp[_i-1]+__temp[_i+1]);

    // Reversing the lazy transform
    __temp[_len-1] += (0.5)*(__temp[_len-2]+__temp[0]);
    for(int _i=1; _i < _len-1; _i+=2)
        __temp[_i] += (0.5)*(__temp[_i-1]+__temp[_i+1]);
    
    // Setiing the values in _arr
    for(int _i=0; _i < _len; _i++) _arr.at<float>(0, _i) = __temp[_i];

    // Return the created array
    return _arr;
}


/*
 *
 * 
 */
void lazy_transform(Mat& _img, Size _shp, bool _scl = false) {
    float __c = 1;
    if(_scl) __c = 1.4;

    // Getting the dimensions
    short __l1 = _shp.height, __l2 = _shp.width;

    // Base case
    if (__l2 <= 2 || __l1 <= 2) return;

    // Working for the first dimension
    for(int i=0; i<__l1; i++)
        lazy_wavelet_reduce(_img.row(i), __l1).copyTo(_img.row(i));;
    
    // Image transpose to work with columns as rows
    transpose(_img, _img);

    // Working for the second dimension
    for(int i=0; i<__l2; i++)
        lazy_wavelet_reduce(_img.row(i), __l2).copyTo(_img.row(i));

    // Back transpose to recover the image
    transpose(_img, _img);

    // Reorganising the matrix
    Mat __temp;
    _img.copyTo(__temp);
    for(int i=0; i<__l1; i++)
        for(int j=0; j<__l2; j++) {
            if(!(i%2 || j%2)) { _img.at<float>(i/2, j/2) = __c*__temp.at<float>(i, j); };
            if(i%2 && !(j%2)) { _img.at<float>(__l1/2 + (i-1)/2, j/2) = __temp.at<float>(i, j); };
            if(!(i%2) && j%2) { _img.at<float>(i/2, __l2/2 + (j-1)/2) = __temp.at<float>(i, j); };
            if ( i%2 && j%2 ) { _img.at<float>(__l1/2 + (i-1)/2, __l2/2 + (j-1)/2) = __temp.at<float>(i, j); };
        }
    
    // Calling the recursion
    lazy_transform(_img, Size(__l1/2, __l2/2), _scl);
}


/*
 *
 * 
 */
void lazy_inverse(Mat& _img, Size _shp, bool _scl = false) {
    float __c = 1;
    if(_scl) __c = 1.4;

    // Getting the dimensions
    short __l1 = _shp.height, __l2 = _shp.width;

    // Base case
    if (__l2 <= 2 || __l1 <= 2) return;

    // Working on the lower levels of recursion
    lazy_inverse(_img, Size(__l1/2, __l2/2), _scl);

    // Reorganising the matrix
    Mat __temp;
    _img.copyTo(__temp);
    for(int i=0; i<__l1; i++)
        for(int j=0; j<__l2; j++) {
            if (i<__l1/2 && j<__l2/2)  { _img.at<float>(2*i, 2*j) = __temp.at<float>(i, j)/__c; };
            if(i>=__l1/2 && j<__l2/2)  { _img.at<float>((i-__l1/2)*2+1, 2*j) = __temp.at<float>(i, j); };
            if(i<__l1/2 && j>=__l2/2)  { _img.at<float>(2*i, (j-__l2/2)*2+1) = __temp.at<float>(i, j); };
            if(i>=__l1/2 && j>=__l2/2) { _img.at<float>((i-__l1/2)*2+1, (j-__l2/2)*2+1) = __temp.at<float>(i, j); };
        }

    // Image transpose to work with columns as rows
    transpose(_img, _img);

    // Working for the second dimension
    for(int i=0; i<__l2; i++)
        lazy_wavelet_inverse(_img.row(i), __l1).copyTo(_img.row(i));

    // Back transpose to recover the image
    transpose(_img, _img);

    // Working for the first dimension
    for(int i=0; i<__l1; i++)
        lazy_wavelet_inverse(_img.row(i), __l2).copyTo(_img.row(i));;
    
}

#endif