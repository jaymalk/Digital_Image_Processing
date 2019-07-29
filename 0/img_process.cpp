#include <opencv2/opencv.hpp>
#include <string>
#include <exception>

/*
! Program for reading an image, processing it and writing a new image formed by averaging the rows.

* Parameters
    * @param argv[1] -> Relative path of input image
    * @param argv[2] -> Relative path of output image
*/

// Main
int main(int argc, char** argv) {
    // Input and output name strings
    std::string _in, _out{"final.jpeg"} ;
    // Naming state
    switch(argc) {
        // No cli-args
        case 1: {
            std::cout << "Enter the input file name : ";
            std::cin >> _in;
            break;
            }
        // Input name args
        case 2: {
            _in = argv[1];
            break;
            }
        // Input and output name args
        case 3: {
            _in = argv[1];
            _out = argv[2];
            break;
            }
        //excess args
        default : {
            std::cout << "Only 2 arguments accepted. (_in & _out)\n";
            _in = argv[1];
            _out = argv[2];
            break;
            }
    }
    // Image variable
    cv::Mat img;
    try {
        // Loading the image
        img = cv::imread(_in,1);
        // Exiting if image doesn't exist
        if(img.empty())
            throw std::runtime_error("File Doesn't exist\nImage is empty\n");
    } 
    catch(const std::runtime_error& e) {
        // Image doesn't exist
        std::cerr << e.what();
    } 
    catch(...) {
        // Unknown error in reading (exit code 1)
        std::cerr << "Problem in reading image.\nExiting...\n";
        exit(1);
    }

    // New image matrix
    cv::Size_ size = img.size();
    cv::Mat new_img = cv::Mat::zeros(size, img.type());

    // Processing the image row wise
    for(size_t i {}; i < size.height; i++) {
        int bgr[] {0, 0, 0};
        // Summing the row
        for(size_t j {}; j < size.width; j++) {
            uchar* x = img.ptr<uchar>(i, j);
            bgr[0] += (int)x[0];
            bgr[1] += (int)x[1];
            bgr[2] += (int)x[2];
        }
        // Averaging values
        bgr[0] /= size.width;
        bgr[1] /= size.width;
        bgr[2] /= size.width;
        // Setting the row of new image
        for(size_t j {}; j < size.width; j++) {
            new_img.at<cv::Vec3b>(i, j) = cv::Vec3b((uchar)bgr[0], (uchar)bgr[1], (uchar)bgr[2]);
        }
    }
    // Writing the new created image
    cv::imwrite(_out, new_img);
    return 0;
}
