/*
 * Library of functions for Haar-Transforms
 */

#ifndef __HAAR__
#define __HAAR__

#include <opencv2/opencv.hpp>

#define sq2 sqrt(2)

using namespace std;
using namespace cv;


float __scale = sqrt(2);

/*
 * Haar-Transform of a matrix, given a matrix and len's, give its transform
 * @param _matrix : input matrix (it is the one that is modified)
 * @param _row : row length (preferrably even)
 * @param _col : col length (preferrably even)
 */
void haar_transform(Mat& _matrix, int _row, int _col, bool normalised = true) {
    float _c = 1;
    if(normalised)
        _c = sq2;
    // Base case for the recursion
    if(_row <= 1 || _col <= 1) return;
    // Local Variables
    float _avg;
    // Working row wise
    if(_row%2)
        for(size_t i{}; i<_col; i++)
            _matrix.at<float>(_row-1, i) = 0;
    _row -= _row%2;
    for(size_t i{}; i<_row; i+=2)
        for(size_t j{}; j<_col; j++) {
            _avg = (_matrix.at<float>(i, j)+_matrix.at<float>(i+1, j))/2;
            _matrix.at<float>(i+1, j) = _c*(_matrix.at<float>(i, j)-_avg);
            _matrix.at<float>(i, j) = _c*_avg;
        }
    // Working column wise
    if(_col%2)
        for(size_t i{}; i<_row; i++)
            _matrix.at<float>(i, _col-1) = 0;
    _col -= _col%2;
    for(size_t j{}; j<_col; j+=2)
        for(size_t i{}; i<_row; i++) {
            _avg = (_matrix.at<float>(i, j)+_matrix.at<float>(i, j+1))/2;
            _matrix.at<float>(i, j+1) = _c*(_matrix.at<float>(i, j)-_avg);
            _matrix.at<float>(i, j) = _c*_avg;
        }
    //! Organising the matrix
    // Copying the matrix first
    Mat _temp = Mat::zeros(_matrix.size(), _matrix.type());
    _matrix.copyTo(_temp);
    // Setting the values
    for(size_t i{}; i<_row; i++)
        for(size_t j{}; j<_col; j++) {
            if(!(i%2 || j%2)) { _matrix.at<float>(i/2, j/2) = _temp.at<float>(i, j); };
            if(i%2 && !(j%2)) { _matrix.at<float>(_row/2 + (i-1)/2, j/2) = _temp.at<float>(i, j); };
            if(!(i%2) && j%2) { _matrix.at<float>(i/2, _col/2 + (j-1)/2) = _temp.at<float>(i, j); };
            if ( i%2 && j%2 ) { _matrix.at<float>(_row/2 + (i-1)/2, _col/2 + (j-1)/2) = _temp.at<float>(i, j); };
        }
    // Recursivly working on the quadrant
    haar_transform(_matrix, _row/2, _col/2, normalised);
}

/*
 * Recreating the image from its haar transform.
 * @param _matrix : input matrix (it is the one that is modified)
 * @param _row : row length (preferrably even)
 * @param _col : col length (preferrably even)
 */
void inverse_haar_transform(Mat& _matrix, int _row, int _col, bool normalised = true) {
    // Base case for recursion
    if(_row <= 1 || _col <= 1) return;
    // First creating the first quadrant completely
    inverse_haar_transform(_matrix, _row/2, _col/2, normalised);
    // Assuming the first quadrant is created
    //! Reorganising the matrix
    // Copying the matrix first
    Mat _temp = Mat::zeros(_matrix.size(), _matrix.type());
    _matrix.copyTo(_temp);
    // Setting the values
    for(size_t i{}; i<_row-_row%2; i++)
        for(size_t j{}; j<_col-_col%2; j++) {
            if (i<_row/2 && j<_col/2)  { _matrix.at<float>(2*i, 2*j) = _temp.at<float>(i, j); };
            if(i>=_row/2 && j<_col/2)  { _matrix.at<float>((i-_row/2)*2+1, 2*j) = _temp.at<float>(i, j); };
            if(i<_row/2 && j>=_col/2)  { _matrix.at<float>(2*i, (j-_col/2)*2+1) = _temp.at<float>(i, j); };
            if(i>=_row/2 && j>=_col/2) { _matrix.at<float>((i-_row/2)*2+1, (j-_col/2)*2+1) = _temp.at<float>(i, j); };
        }
    // Recreating from columns
    _col -= _col%2;
    for(size_t j{}; j<_col; j+=2)
        for(size_t i{}; i<_row; i++) {
            if(normalised) {
                _matrix.at<float>(i, j) = (_matrix.at<float>(i, j) + _matrix.at<float>(i, j+1))/sq2;
                _matrix.at<float>(i, j+1) = _matrix.at<float>(i, j) - sq2*_matrix.at<float>(i, j+1);
            }
            else {
                _matrix.at<float>(i, j) = (_matrix.at<float>(i, j) + _matrix.at<float>(i, j+1));
                _matrix.at<float>(i, j+1) = _matrix.at<float>(i, j) - 2*_matrix.at<float>(i, j+1);
            }
        }
    // Recreating from rows
    _row -= _row%2;
    for(size_t i{}; i<_row; i+=2)
        for(size_t j{}; j<_col; j++) {
            if(normalised) {
                _matrix.at<float>(i, j) = (_matrix.at<float>(i, j) + _matrix.at<float>(i+1, j))/sq2;
                _matrix.at<float>(i+1, j) = _matrix.at<float>(i, j) - sq2*_matrix.at<float>(i+1, j);
            }
            else {
                _matrix.at<float>(i, j) = (_matrix.at<float>(i, j) + _matrix.at<float>(i+1, j));
                _matrix.at<float>(i+1, j) = _matrix.at<float>(i, j) - 2*_matrix.at<float>(i+1, j);
            }
        }
}

#endif