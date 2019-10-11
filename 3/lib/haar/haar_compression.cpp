/*
 * Library for lossy compression/decompression of image files
 */

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "../image_io.hpp"
#include "haar.hpp"


// ! COMPRESSION PART

void reset_cut(Mat& img, int& _ct) {
    // Getting the cutoff for top _ct%
    int *_tab = new int[10001]();
    double *_mx = new double, *_mn = new double;
    // Getting the image maximum
    minMaxLoc(abs(img), _mn, _mx);
    img *= 5000;
    img /= (*_mx);
    // Filling up the matrix
    for(size_t i{}; i<img.size().height; i++)
        for(size_t j{}; j<img.size().width; j++)
            _tab[(int)round(img.at<float>(i, j))+5000]++;
    // Getting the cutoff
    long _total = (img.size().height * img.size().width * _ct) / 100;
    long _cnt = _tab[5000];
    for(int i=1; i<=5000; i++) {
        _cnt += _tab[5000-i];
        _cnt += _tab[5000+i];
        if(_cnt >= _total) {
            cout << _ct << " " << i << "\n";
            cout << _cnt << " " << _total << "\n";
            _ct = (int)((*_mx) * ((float)i/5000.0));
            break;
        }
    }
    // Resetting the image
    img /= 5000.0;
    img *= (*_mx);
    // Freeing space
    free(_tab);
    free(_mx);
    free(_mn);
}


// Thresholding
void cutoff_reduce(Mat& img, int _ct) {
    // reset_cut(img, _ct);
    // Setting the cutoff based on k and maximum
    float _cut = (float)_ct/100.0;
    for(size_t i{}; i<img.size().height; i++)
        for(size_t j{}; j<img.size().width; j++) {
            if(abs(img.at<float>(i, j)) < _cut)
                img.at<float>(i, j) = 0;
        }
}


// Run length encoder
vector<pair<uchar, int>> runlength_encode(Mat& img) {
    // First attempt
    float _max = img.at<float>(0,0);
    img *= 128;
    img += 128;
    img.convertTo(img, CV_8UC1);
    // Coding now
    vector<pair<uchar, int>> _code;
    uchar _cval = img.at<uchar>(0,0);
    int _run = 0;
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
    _code.push_back({_cval, _run});
    return _code;
}

// Writing to the file
void write_run_length(Size _size, vector<pair<uchar, int>> _code, string _fname) {
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
    for(size_t i{}; i<_len; i++)
        out.write(reinterpret_cast<const char *>(&(_code[i])), sizeof(pair<uchar, int>));
    // Closing the file
    out.close();
}


// Encoder
void encode(string path, int _cut, string _fname = "./save.bin")
{
    Mat img;
    // Loading the image
    load_image(path, img, true);
    // Taking the haar transform of the image
    haar_transform(img, img.size().height, img.size().width);
    // Thresholding the image
    cutoff_reduce(img, _cut);
    // Taking the inverse transform
    inverse_haar_transform(img, img.size().height, img.size().width);
    // Taking the de-normalised transform
    haar_transform(img, img.size().height, img.size().width, false);
    // Creating a run length encoded form of the image
    vector<pair<uchar, int>> _encoded = runlength_encode(img);
    // Writing the binary image
    write_run_length(img.size(), _encoded, _fname);
}


// ! DECOMPRESSION PART

// Reader
void read_run_length(string _fname, Size& _sz, vector<pair<uchar, int>>& _decode) {
    // Opening the binary file
    ifstream in(_fname, ios::binary | ios::in);
    // Reading the matrix size
    short _h, _w;
    in.read(reinterpret_cast<char *>(&_h), sizeof(short));
    in.read(reinterpret_cast<char *>(&_w), sizeof(short));
    // Reading the length of encoding
    int _l;
    in.read(reinterpret_cast<char *>(&_l), sizeof(int));
    // Reading the complete encoding in the vector
    pair<uchar, int> _p;
    for(int i=0; i<_l; i++) {
        in.read(reinterpret_cast<char *>(&_p), sizeof(pair<uchar, int>));
        _decode.push_back(_p);
    }
    // Setting the size
    _sz = Size(_h, _w);
    // Closing the file
    in.close();
}

// Recreator
void recreate_image(Mat& _img, vector<pair<uchar, int>>& _decode) {
    // Local variables
    uchar _val = 0; int _run = 0;
    // Looping on all values of the matrix
    for(int i = _img.size().height-1; i>=0; i--)
        for(int j = _img.size().width-1; j>=0; j--) {
            if(_run == 0) {
                _val = _decode.back().first;
                _run = _decode.back().second;
                _decode.pop_back();
            }
            _img.at<float>(i,j) = (static_cast<float>(_val)-128)/128.0;
            _run--;
        }
    // The decoded data must be completely used
    assert(_decode.empty() && (_run == 0));
}

// Decoder
void decode(string _bname, string _sname) {
    // Local variables
    Mat _img; Size _sz; vector<pair<uchar, int>> _decode{};
    // Reading from the file (binary)
    read_run_length(_bname, _sz, _decode);
    // Creating the matrix
    _img = Mat::zeros(_sz, CV_32F);
    // Recreating the transform from the image 
    recreate_image(_img, _decode);
    // Taking the inverse-transform of the image (denormalised)
    inverse_haar_transform(_img, _img.size().height, _img.size().width, false);
    // Saving the image
    save_image(_sname, _img);
}

int main(int argc, char const *argv[]) {
    // Local varaibles
    string _fname, _bname, _sname; int _case, _cut;
    // Reading command line arguments
    switch(argc) {
        case 1: {
            cout << "Enter protocol [decode (0) or encode (1)] : ";
            cin >> _case;
            if(_case) {
                cout << "Enter image path: ";
                cin >> _fname;
            }
            else {
                cout << "Enter binary path: ";
                cin >> _bname;
            }
            cout << "Enter save-path: ";
            cin >> _sname;
            if(_case) {
                cout << "Enter cutoff (0-100): ";
                cin >> _cut;
                assert(abs(_cut)<=100);
            }
            break;
        }
        case 2: {
            _case = stoi(argv[1]);
            if(_case) {
                cout << "Enter image path: ";
                cin >> _fname;
            }
            else {
                cout << "Enter binary path: ";
                cin >> _bname;
            }
            cout << "Enter save-path: ";
            cin >> _sname;
            if(_case) {
                cout << "Enter cutoff (0-100): ";
                cin >> _cut;
                assert(abs(_cut)<=100);
            }
            break;
        }
        case 3: {
            _case = stoi(argv[1]);
            if(_case) _fname = argv[2];
            else _bname = argv[2];
            cout << "Enter save-path: ";
            cin >> _sname;
            if(_case) {
                cout << "Enter cutoff (0-100): ";
                cin >> _cut;
                assert(abs(_cut)<=100);
            }
            break;
        }
        case 4: {
            _case = stoi(argv[1]);
            if(_case) _fname = argv[2];
            else _bname = argv[2];
            _sname = argv[3];
            if(_case) {
                cout << "Enter cutoff (0-100): ";
                cin >> _cut;
                assert(abs(_cut)<=100);
            }
            break;
        }
        case 5: {
            _case = stoi(argv[1]);
            assert(_case);
            _fname = argv[2];
            _sname = argv[3];
            _cut = stoi(argv[4]);
            assert(abs(_cut) <= 100);
            break;
        }
        default: {
            cout << "Incomplete Arguments. Exiting\n";
            exit(1);
        }
    }
    // Calling for service
    if(_case) encode(_fname, _cut, _sname);
    else decode(_bname, _sname);
}
