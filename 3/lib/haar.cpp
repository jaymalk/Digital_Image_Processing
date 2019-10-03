

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


float __scale = sqrt(2);


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
void load_image(std::string _name, cv::Mat& _img) {
    try {
        // Loading the image on the reference
        _img = cv::imread(_name, 0);
        // Exiting if image doesn't exist
        if(_img.empty())
            throw std::runtime_error("File Doesn't exist\nImage is empty\nExiting...\n");
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


/*
 * Haar-Transform of a matrix, given a matrix and len's, give its transform
 * @param _matrix : input matrix (it is the one that is modified)
 * @param _row : row length (preferrably even)
 * @param _col : col length (preferrably even)
 */
void haar_transform(Mat& _matrix, int _row, int _col) {
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
            _matrix.at<float>(i+1, j) = _matrix.at<float>(i, j)-_avg;
            _matrix.at<float>(i, j) = _avg;
        }
    // Working column wise
    if(_col%2)
        for(size_t i{}; i<_row; i++)
            _matrix.at<float>(i, _col-1) = 0;
    _col -= _col%2;
    for(size_t j{}; j<_col; j+=2)
        for(size_t i{}; i<_row; i++) {
            _avg = (_matrix.at<float>(i, j)+_matrix.at<float>(i, j+1))/2;
            _matrix.at<float>(i, j+1) = _matrix.at<float>(i, j)-_avg;
            _matrix.at<float>(i, j) = _avg;
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
    haar_transform(_matrix, _row/2, _col/2);
}

/*
 * Recreating the image from its haar transform.
 * @param _matrix : input matrix (it is the one that is modified)
 * @param _row : row length (preferrably even)
 * @param _col : col length (preferrably even)
 */
void inverse_haar_transform(Mat& _matrix, int _row, int _col) {
    // Base case for recursion
    if(_row <= 1 || _col <= 1) return;
    // First creating the first quadrant completely
    inverse_haar_transform(_matrix, _row/2, _col/2);
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
            _matrix.at<float>(i, j) = _matrix.at<float>(i, j) + _matrix.at<float>(i, j+1);
            _matrix.at<float>(i, j+1) = _matrix.at<float>(i, j) - 2*_matrix.at<float>(i, j+1);
        }
    // Recreating from rows
    _row -= _row%2;
    for(size_t i{}; i<_row; i+=2)
        for(size_t j{}; j<_col; j++) {
            _matrix.at<float>(i, j) = _matrix.at<float>(i, j) + _matrix.at<float>(i+1, j);
            _matrix.at<float>(i+1, j) = _matrix.at<float>(i, j) - 2*_matrix.at<float>(i+1, j);
        }
}

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
