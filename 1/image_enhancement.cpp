#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
#include <exception>

/*
 * Various image enhancement operations on image.
 *      Available operations include
 *          - Spatial Enhancement (Filters)
 *          - Histogram Equalisation
 *          - Half-Toning
 */


// ! FILTER BASED FUNCTIONS

// ! Filters available to be applied
// String giving complete detail of filters available
std::string __filter_details = "Available Filters\navgi   : Average Filter I  (Uniform Through Out) (Parameter : Size)\navgii  : Average Filter II (Emphasized Average)\nshpi   : Sharpening Filter I  (N4)\nshpii  : Sharpening Filter II (N8)\nbsti   : Boosting Filter I (N4) (Parameter : Boost)\nbstii  : Boosting Filter II (N8) (Parameter : Boost)\n";

/*
 * Averaging Filter I (Simple averaging filter)
 * @param _filter : filter object placeholder
 * @param _size : size of the filter (must be postive odd number) (default 3)
 */
void average_filter_i(cv::Mat& _filter, int _size = 3) {
    try {
        if((_size-1)%2 || _size < 3)
            throw new std::runtime_error("Error, _size must be an odd greater than 1. [average_filter_i]");
        float _cnt = 1.0/(_size*_size);
        _filter = cv::Mat(_size, _size, CV_32FC1, cv::Scalar(_cnt));
    }
    catch(...) {
        std::cout << "Exception [average_filter_i]\n";
    }
}

/*
 * Averaging Filter II (Averaging filter with different emphasis to N4 & ND)
 * @param _filter : filter object placeholder
 */
void average_filter_ii(cv::Mat& _filter) {
    float _[3][3] = {{1/16.0, 2/16.0, 1/16.0}, {2/16.0, 4/16.0, 2/16.0}, {1/16.0, 2/16.0, 1/16.0}};
    _filter = cv::Mat(3, 3, CV_32FC1);
    std::memcpy(_filter.data, _, 3*3*sizeof(float));
}

/*
 * Boosting Filter I (Considering N4)
 * @param _boost : amount of boost to the orginal pixel (>= 1) (default 1)
 * @param _filter : filter object placeholder
 */
void boosting_filter_i(cv::Mat& _filter, float _boost = 1) {
    if(_boost < 1)
        throw new std::runtime_error("Error, _boost must be greater than 1. [boosting_filter_ii]");
    float _[3][3] = {{0, -1, 0}, {-1, 4+_boost, -1}, {0, -1, 0}};
    _filter = cv::Mat(3, 3, CV_32FC1);
    std::memcpy(_filter.data, _, 3*3*sizeof(float));
}

/*
 * Boosting Filter II (Considering N8)
 * @param _boost : amount of boost to the orginal pixel (>= 1) (default 1)
 * @param _filter : filter object placeholder
 */
void boosting_filter_ii(cv::Mat& _filter, float _boost = 1) {
    if(_boost < 1)
        throw new std::runtime_error("Error, _boost must be greater than 1. [boosting_filter_ii]");
    float _[3][3] = {{-1, -1, -1}, {-1, 8+_boost, -1}, {-1, -1, -1}};
    _filter = cv::Mat(3, 3, CV_32FC1);
    std::memcpy(_filter.data, _, 3*3*sizeof(float));
}

/*
 * Prewitt Filter
 * @param _filter : filter object placeholder
 * @param _ornt : vertical/horizontal orientations ('H' or 'V', only) (default H)
 */
void prewitt(cv::Mat& _filter, char _ornt = 'H') {
    // Setting arrays
    float _H[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    float _V[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    // Setting filter
    _filter = cv::Mat(3, 3, CV_32FC1);
    // Applying filter
    if(_ornt == 'H')
        std::memcpy(_filter.data, _H, 3*3*sizeof(float));
    else if(_ornt == 'V')
        std::memcpy(_filter.data, _V, 3*3*sizeof(float));
    else
        // Exception thrown if orientation is invalid
        throw new std::runtime_error("Invalid orientation. [prewitt]");
}

/*
 * Sobel Filter
 * @param _filter : filter object placeholder
 * @param _ornt : vertical/horizontal orientations ('H' or 'V', only) (default H)
 */
void sobel(cv::Mat& _filter, char _ornt = 'H') {
    // Setting arrays
    float _H[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    float _V[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    // Setting filter
    _filter = cv::Mat(3, 3, CV_32FC1);
    // Applying filter
    if(_ornt == 'H')
        std::memcpy(_filter.data, _H, 3*3*sizeof(float));
    else if(_ornt == 'V')
        std::memcpy(_filter.data, _V, 3*3*sizeof(float));
    else
        // Exception thrown if orientation is invalid
        throw new std::runtime_error("Invalid orientation. [sobel]");
}

// ! Filter application method

/*
 * Applying filter to the image.
 * @param _img : original image matrix
 * @param _filter : filter matrix
 * @param _new_img : image matrix of new image (after filter application)
 * @param _med : median filter. (applied explicitly)
 */
void apply_filter(cv::Mat& _img, cv::Mat& _filter, cv::Mat& _new_img, bool _med = false) {
    try {
        // Getting attributes
        int imlen_x = _img.size().width, imlen_y = _img.size().height;
        int flen = (_med)?(7):(_filter.size().width);
        int pad = (flen-1)/2;
        // Creating new image
        _new_img = cv::Mat::zeros(imlen_y, imlen_x, CV_8UC1);
        // Applying filter on new image
        try {
            for(int i=pad; i<imlen_y-pad; i++)
                for(int j=pad; j<imlen_x-pad; j++) {
                    // Filter Convolution starts
                    float val = 0, temp, prod;
                    uchar *_tab;
                    // If median is to be found, initialise array
                    if(_med)
                        _tab = (uchar *)malloc(sizeof(uchar)*flen*flen);
                    for(int k=0; k<flen; k++)
                        for(int l=0; l<flen; l++) {
                            if(_med) {
                                // Adde value in array
                                _tab[l*flen+k] = _img.at<uchar>(i+k-pad, j+l-pad);
                            }
                            else {
                                // Convolve
                                prod = (float)(_filter.at<float>(k, l)*(_img.at<uchar>(i+k-pad, j+l-pad)));
                                val += prod;
                            }
                        }
                    // If median, find and put 
                    if(_med) {
                        std::sort(_tab, _tab+flen*flen);
                        _new_img.at<uchar>(i, j) = _tab[flen*flen/2];
                    }
                    // Else, assign the convolution value
                    else
                        if(val <= 0)
                            _new_img.at<uchar>(i, j) = (uchar)0;
                        else
                            _new_img.at<uchar>(i, j) = ((uchar)std::min(val, 255.0F));
                }
        }
        catch(...) {
            std::cerr << "Inner Exception [apply_filter]\n";
        }
    }
    catch(...) {
            std::cerr << "Outer Exception [apply_filter]\n";
    }
}

// ! HISTOGRAM BASED METHODS

// ! Transformation maps
/*
 * Transformation function based on CDF
 * @param _img : source image file
 * @param _map : mapping array pointer, will be updated
 */
void CDF_map(cv::Mat& img, int* _map) {

    // Setting value count (PDF)
    for(int i=0; i<img.size().width; i++)
        for(int j=0; j<img.size().height; j++)
            _map[(uint)img.at<uchar>(j, i)] += 1;

    // Initialising histogram min/max
    int min = INT32_MAX, max = 0;

    // Setting CDF
    for(int i=1; i<256; i++) {
        _map[i] += _map[i-1];
        if(_map[i-1]!=0 && _map[i-1]<min)
            min = _map[i-1];
        if(_map[i]>max)
            max = _map[i];
    }
    // Mapping CDF to range
    for(int i=0; i<256; i++) {
        if(_map[i] != 0)
            _map[i] = std::round(((float)(_map[i] - min)/(float)(max-min))*(220));
    }
}

/*
 * Transformation function T(r) = pow(r, gamma)/pow(255, gamma)
 * @param _map : mapping array pointer, will be updated
 * @param gamma : the correction value (0 < () < 4)
 */
void gamma_correction_map(int * _map, float gamma = 1/2.2) {
    if(gamma <= 0 || gamma > 4)
        throw new std::runtime_error("Error, gamma must be in the range (0, 4]. [gamma_correction_map]");
    float max = pow(255, gamma);
    for(size_t i{}; i<256; i++)
        _map[i] = static_cast<int>(255*(pow(i, gamma)/max));
}

// ! Histogram mapping function 
/*
 *  Histogram equalisation on complete image. Uses CDF as the default transformation function.
 * @param _img : source image file
 * @param _new_img : modified image location
 * @param _map : transformation map (array)
 */
void global_histogram_equilisation(cv::Mat& _img, cv::Mat& _new_img, int* _map = NULL) {
    // Checking if map exists
    if(_map == NULL) {
        _map = (int *)calloc(256, sizeof(int));
        // If map doesnt exist use CDF map
        CDF_map(_img, _map);
    }
    // New image matrix
    _new_img = cv::Mat::zeros(_img.size().height, _img.size().width, 0);
    // Assigning the new matrix equalised value
    for(int i=0; i<_img.size().width; i++)
        for(int j=0; j<_img.size().height; j++)
            _new_img.at<uchar>(j, i) = (uchar)_map[(int)_img.at<uchar>(j, i)];
}


// ! RGB -> Greyscale

/*
 * Getting the luminance of the image (RGB)
 * @param _img : original image (BGR)
 * @param _lum : luminance matrix (to be created)
 */
void get_luminance(cv::Mat& _img, cv::Mat& _lum) {
    try {
        _lum = cv::Mat(_img.size(), CV_8UC1);
        for(int i=0; i<_img.size().width; i++)
            for(int j=0; j<_img.size().height; j++) {
                cv::Vec3b v = _img.at<cv::Vec3b>(j, i);
                float l = 0.2990f*static_cast<float>(v[2]) + 0.5870f*static_cast<float>(v[1]) + 0.1140f*static_cast<float>(v[0]);
                _lum.at<uchar>(j, i) = static_cast<uchar>(l);
            }
    }
    catch(const std::runtime_error& e) {
        // Exception
        std::cerr << e.what();
        exit(1);
    }
}

// ! GreyScale * RGB -> RGB

/*
 * Mapping a different luminance on the present image
 * @param _img : original image
 * @param _lum : new luminance map
 */
void change_luminance(cv::Mat& _img, cv::Mat& _lum, float gm = 0.4) {
    try {
        for(int i=0; i<_img.size().width; i++)
            for(int j=0; j<_img.size().height; j++) {
                float _nl = static_cast<float>(_lum.at<uchar>(j, i));
                cv::Vec3b v = _img.at<cv::Vec3b>(j, i);
                float _ol = 0.2990f*static_cast<float>(v[2]) + 0.5870f*static_cast<float>(v[1]) + 0.1140f*static_cast<float>(v[0]);
                v[0] = static_cast<uchar>(std::min(254.0f, static_cast<float>(1+v[0]) * pow(_nl/_ol , gm)));
                v[1] = static_cast<uchar>(std::min(254.0f, static_cast<float>(1+v[1]) * pow(_nl/_ol , gm)));
                v[2] = static_cast<uchar>(std::min(254.0f, static_cast<float>(1+v[2]) * pow(_nl/_ol , gm)));
                cv::Vec3b _df = v - _img.at<cv::Vec3b>(j, i);
                if(_df[0] < 100 && _df[1] < 100 && _df[2] < 100)
                    _img.at<cv::Vec3b>(j, i) = v;
            }
    }
    catch(const std::runtime_error& e) {
        // Exception
        std::cerr << e.what();
        exit(1);
    }
}

// ! BASIC FUNCTIONS

/* 
 * Loading image (using relative path)
 * @param _name : relative path of the image to be uploaded
 * @param _img : const reference to loading array
 */
void load_image(std::string _name, cv::Mat& _img) {
    try {
        // Loading the image on the reference
        _img = cv::imread(_name);
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
        cv::imwrite(_name, _img);
    }
    catch(const std::runtime_error& e) {
        // Image doesn't exist
        std::cerr << e.what();
        exit(1);
    }
}

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


int main(int argc, const char** argv) {
    // String holders
    std::string rdpth, svpth, prc;
    float gmv1, gmv2;
    bool gm1=false, gm2=false;
    // Assigning parameters
    switch(argc) {
        case 1: {
            // Nothing is entered
            std::cout << "Enter image path : ";
            std::cin >> rdpth;
            std::cout << "Enter save path : ";
            std::cin >> svpth;
            std::cout << "Enter the process (hist-eq, avg, boost, sbl, pwt) : ";
            std::cin >> prc;
            break;
        }
        case 2: {
            rdpth = argv[1];
            std::cout << "Enter save path : ";
            std::cin >> svpth;
            std::cout << "Enter the process (hist-eq, avg, boost, sbl, pwt) : ";
            std::cin >> prc;
            break;
        }
        case 3: {
            rdpth = argv[1];
            svpth = argv[2];
            std::cout << "Enter the process (hist-eq, avg, boost, sbl, pwt) : ";
            std::cin >> prc;
            break;
        }
        case 4: {
            rdpth = argv[1];
            svpth = argv[2];
            prc = argv[3];
            break;
        }
        case 5: {
            rdpth = argv[1];
            svpth = argv[2];
            prc = argv[3];
            if(prc[0] == 'g') {
                gm1 = true;
                try {
                    gmv1 = std::stof(argv[4]);
                }
                catch(...) {
                    std::cerr << "Error while reading [gmv1]\nExiting...\n";
                    exit(1);
                }
            }
            else {
                gm2 = true;
                try {
                    gmv2 = std::stof(argv[4]);
                }
                catch(...) {
                    std::cerr << "Error while reading [gmv2]\nExiting...\n";
                    exit(1);
                }
            }
            break;
        }
        case 6: {
            if(argv[3][0] != 'g') {
                std::cerr << "5 Inputs only for Gamma\nExiting...\n";
                exit(1);
            }
            rdpth = argv[1];
            svpth = argv[2];
            prc = argv[3];
            gm1 = true;
            gm2 = true;
            try {
                    gmv1 = std::stof(argv[4]);
                    gmv2 = std::stof(argv[5]);
            }
            catch(...) {
                std::cerr << "Error while reading [gmv1 or gmv2]\nExiting...\n";
                exit(1);
            }
            break;
        }
        default: {
            std::cerr << "Invalid CL Inputs. [Max 5]\nExiting...\n";
            exit(1);
        }
    }
    // Containers
    cv::Mat img, lum, new_lum, filter;
    // Loading image
    load_image(rdpth, img);
    // Showing image
    show_image(img);
    // Creating Luminousity Matrix
    get_luminance(img, lum);
    // Showing Luminance
    show_image(lum);

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // APPLICATION LAYER
    switch(prc[0]) {
        case 'h' : // Histogram Equilisation
        {
            global_histogram_equilisation(lum, new_lum);
            break;
        }
        case 'b' : // Boosting Filter
        {
            boosting_filter_i(filter);
            apply_filter(lum, filter, new_lum);
            break;
        }
        case 'u' : // Unsharp Masking
        {
            average_filter_i(filter);
            apply_filter(lum, filter, new_lum);
            new_lum = lum + (lum - new_lum);
            break;
        }
        case 'a' : // Average Filter
        {
            average_filter_i(filter);
            apply_filter(lum, filter, new_lum);
            break;
        }
        case 's' : // Sobel Operator
        {
            sobel(filter, 'H');
            apply_filter(lum, filter, new_lum);
            new_lum = lum + new_lum;
            break;
        }
        case 'm' : // Median Filter
        {
            apply_filter(lum, filter, new_lum, true);
            break;
        }
        case 'g' : // Gamma Correction
        {
            // break;
            lum.convertTo(lum, CV_32F);
            if(gm1)
                cv::pow(lum, gmv1, new_lum);
            else
                cv::pow(lum, 1/2.2, new_lum);
            new_lum.convertTo(new_lum, CV_8U);
            break;
        }
        default : {std::cout << "No viable option.\nExiting...\n"; exit(2);}
    }
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // Showing new luminance
    show_image(new_lum);
    // Applying new luminance to image
    if(gm2)
        change_luminance(img, new_lum, gmv2);
    else
        change_luminance(img, new_lum);
    // Showing new image
    show_image(img);
    // Saving image
    save_image(svpth, img);
}