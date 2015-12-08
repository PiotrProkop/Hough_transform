//
// Created by pprokop on 12/5/15.
//


#ifndef HOUGH_TRANSFORM_HOUGH_H
#define HOUGH_TRANSFORM_HOUGH_H
#define GROUP_SIZE 16
#include "../include/AMDSDKUtil/SDKBitMap.hpp"
#include <CL/cl.h>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/core/core.hpp"
//#include "../include/utils.h"

using namespace appsdk;
using namespace cv;
using namespace cv::ocl;
class Hough {

public:
    SDKBitMap inputBitmap;   /**< Bitmap class object */
    uchar4* pixelData;       /**< Pointer to image data */
    cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */
    cl_uint width;                      /**< Width of image */
    cl_uint height;                     /**< Height of image */
    cl_uint width_original;
    cl_uint height_original;
    cl_uchar4* inputImageData;          /**< Input bitmap data to device */
    cl_uchar4* outputImageData;         /**< Output from device */
    cl_uchar4* accumulator;
    //cl_uchar4* num_lines;
    cl_program program;
    cl_int ret;
    cl_device_id device_id;
    cl_context context;
    cl_platform_id platform_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_kernel kernel;
    cl_command_queue command_queue;
    cl_mem memobj;
    cl_mem inputImageBuffer;
    cl_mem outputImageBuffer;
    cl_mem thetaBuffer;
    size_t blockSizeX;                  /**< Work-group size in x-direction */
    size_t blockSizeY;


    int readInputImage(std::string inputImageName);
    int writeOutputImage(std::string outputImageName);
    void createKernel(char*,const char*);
    void cleanUp();
    void setupCL(char* fileName);
    void run();
    void enqueueKernel(char*, const char*, bool=false);
    void greyScale();
    void gauss();
    void sobel();
    void max();
    void hyst();
    void swapBuffers();
    void houghTransform();



    Hough()
            : inputImageData(NULL),
              outputImageData(NULL)

    {
        pixelSize = sizeof(uchar4);
        pixelData = NULL;
        program = NULL;
        device_id = NULL;
        context = NULL;
        command_queue = NULL;
        memobj = NULL;
        blockSizeX = GROUP_SIZE;
        blockSizeY = 1;

    }

};
#endif //HOUGH_TRANSFORM_HOUGH_H
