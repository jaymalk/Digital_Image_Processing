

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <opencv2/opencv.hpp>

#include "image_io.hpp"

using namespace std;
using namespace cv;

#define loopx(_var, _start, _end, _step) for(int _var = _start; _var<_end; _var+=_step)

ofstream __f("data.txt");

/*
 * Neighbour-Hood Class
 * Keeps track of neighbourhood features, and extraction and comparison
 */
class NB {

//  •••••••       Typical NB Shape
//  •••••••        Parameters: rows, cols
//  •••••••
//  •••••••P

// Static Members
static short _r, _c;

// Private
private:
    // Value at the point
    float _val;
    // Detail Containers
    vector<float> _data;

// Public
public:

/*
 * **************
 * Constructors
 * **************
 */


    /*
     * Empty Constructor.
     * Use carefully !!!
     */
    NB() {};


    /*
     * Constructor
     * @param x, y: Point coordinates
     * @param _img: Source image
     * @param _type: NB type (col/row)
     */
    NB(short x, short y, Mat& _img, bool _type=true) {
        // Setting the value
        _val = _img.at<float>(y, x);
        // Setting the vector
        _data = vector<float>(_r*_c, 0.0f);
        // Creating the data
        if(_type)
            for(int i=0; i<_r; i++) {
                // If invalid row, ignore
                if(y-i < 0) break;
                // Working on complete columns
                for(int j=1; j<=_c; j++) {
                    // If invalid, ignore
                    if(x-j < 0) break;
                    _data[_c*i + j - 1] = _img.at<float>(y-i, x-j);
                }
            }
        else
            for(int i=1; i<=_r; i++) {
                // If invalid row, ignore
                if(y-i < 0) break;
                // Working on complete columns
                for(int j=0; j<_c; j++) {
                    // If invalid, ignore
                    if(x+j > _img.cols) break;
                    _data[_r*j + i - 1] = _img.at<float>(y-i, x+j);
                }
            }
    }


/*
 * **************
 * Setters
 * **************
 */

    /*
     * Setting the parameters (STATIC)
     * @param r, b, c: Class Parameters
     * 
     * NOTE: MUST BE CALLED ONLY ONCE
     */
    static void set_class(short r, short c) {
        _r = r;
        _c = c;
    }

/*
 * **************
 * Getters
 * **************
 */

    /*
     * Getting the parameters, shape
     * @param x: parameter type
     */
    static int shape(short x) {
        switch(x) {
            case 0: return _r;
            case 1: return _c;
            default: throw new runtime_error("Invalid Type. [NB.shape]");
        }
    }

    /*
     * Getting the data.
     */
    vector<float>& data() {
        return _data;
    }

    /*
     * Getting the value.
     */
    float value() {
        return _val;
    }

/*
 * **************
 * NB Comparison
 * **************
 */

    float comparison(NB& other, int x, bool _type=true) {
        float _scr;
        for(size_t i{}; i<_data.size(); i++) {
                if((i >= x*_c && _type) || (i >= x*_r && !_type)) break;
                _scr += (pow(_data[i] - other.data()[i], 4)*1000);
        }
        return _scr;
    }

/*
 * **************
 * NB Comparison
 * **************
 */

    void _print(ostream& _out = cout) {
        loopx(i, 0, _data.size(), 1) _out << setprecision(4) << setw(7) << _data[i];
        _out << "\n";
    }
};

// Initialsing the static parameters, NB
short NB::_r, NB::_c;


/*
 *
 */
void nb_construction(Mat& _base, Mat& _template, int _step) {
    // Adding template
    loopx(i, 0, _template.cols, 1)
        loopx(j, 0, _template.rows, 1)
            _base.at<float>(j, i) = _template.at<float>(j, i);

    // Working on the image
        NB _nb; float _df, _val; int _l;

    // Template neighbourhoods
    vector<NB> _temp;
    // Getting all the neighbourhoods of first type
    loopx(i, NB::shape(1)+1, _template.cols, _step)
        loopx(j, NB::shape(0), _template.rows, _step)
            _temp.push_back(NB(i, j, _template));

    // Working on template column wise
    loopx(i, _template.cols, _base.cols, 1) {
        loopx(j, 0, _template.rows, 1) {
            _nb = NB(i, j, _base);
            _df = __DBL_MAX__;
            _l = NB::shape(0);
            if(j < _l) _l = j+1;
            loopx(k, 0, _temp.size(), 1) {
                _val = _nb.comparison(_temp[k], _l);
                if(_val < _df) {
                    _df = _val;
                    _base.at<float>(j, i) = _temp[k].value();
                }
                cout << _val;
            }
        }
    }

    // Clearing the template
    _temp.clear();
    // Getting all the neighbourhoods of first type
    loopx(i, NB::shape(1)+1, _template.cols, _step)
        loopx(j, 0, _template.rows - NB::shape(0), _step)
            _temp.push_back(NB(i, j, _template, false));

    // Working on template column wise
    loopx(j, _template.rows, _base.rows, 1) {
        loopx(i, 0, _base.cols, 1) {
            _nb = NB(i, j, _base, false);
            _df = __DBL_MAX__;
            _l = NB::shape(1);
            if(j+_l > _base.cols) _l = _base.cols - j;
            loopx(k, 0, _temp.size(), 1) {
                _val = _nb.comparison(_temp[k], _l, false);
                if(_val < _df) {
                    _df = _val;
                    _base.at<float>(j, i) = _temp[k].value();
                }
                cout << _val;
            }
        }
    }
}


int main(int argc, char const *argv[])
{
    Mat _t, _img;
    _img = Mat::zeros(192, 192, CV_32F);
    load_image(argv[1], _t, true);
    NB::set_class(32, 32);
    nb_construction(_img, _t, stoi(argv[2]));
    save_image("c.png", _img);
    return 0;
}
