//
// Created by pprokop on 12/5/15.
//

#include "Hough.hpp"
#include "utils.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/core/core.hpp"
#include <vector>
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
#define MAX_LINES 1000
#define MUL_UP(a, b) ((a)/(b)+1)*(b)

using namespace cv;
using namespace cv::ocl;

int Hough::readInputImage(std::string inputImageName)
{
    // load input bitmap image
    inputBitmap.load(inputImageName.c_str());

    // error if image did not load
    if (!inputBitmap.isLoaded())
    {
        std::cout << "Failed to load input image!";
        return 1;
    }


    // get width and height of input image
    height_original = inputBitmap.getHeight();
    width_original = inputBitmap.getWidth();
    height = ((height_original) / GROUP_SIZE) * GROUP_SIZE;
    width = ((width_original) / GROUP_SIZE) * GROUP_SIZE;
    // allocate memory for input & output image data
    inputImageData = (cl_uchar4*)malloc(width_original * height_original * sizeof(cl_uchar4));

    outputImageData = (cl_uchar4*)malloc(width_original * height_original * sizeof(cl_uchar4));

    pixelData = inputBitmap.getPixels();
    if (pixelData == NULL)
    {
        std::cout << "Failed to read pixel Data!";
        return 1;
    }

    // Copy pixel data into inputImageData
    memcpy(inputImageData, pixelData, width_original * height_original * pixelSize);


    // initialize the data to NULL
    //memset(verificationOutput, 0, width_original * height_original * pixelSize);

    return 0;
}


int
Hough::writeOutputImage(std::string outputImageName)
{
    // copy output image data back to original pixel data
    memcpy(pixelData, outputImageData,
           width_original * height_original * pixelSize);

/*    memcpy(pixelData, accumulator,
           accu_width * accu_height * pixelSize);*/


    //inputBitmap.height = height;
    //inputBitmap.width = width;
    // write the output bmp file
    if (!inputBitmap.write(outputImageName.c_str()))
    {
        std::cout << "Failed to write output image!";
        return 1;
    }

    return 0;
}

void
Hough::createKernel(char* fileName, const char* name)
{
    FILE *fp;
    char *source_str;
    size_t source_size;
    int ret;

/* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                        (const size_t *)&source_size, &ret);
    utils::handleError(ret);
/* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    utils::handleError(ret);


/* Create OpenCL Kernel */
    kernel = clCreateKernel(program, name, &ret);
    utils::handleError(ret);
    free(source_str);
}

void
Hough::cleanUp()
{
    ret = clReleaseProgram(program);
    utils::handleError(ret);
    ret = clFlush(command_queue);
    utils::handleError(ret);
    ret = clFinish(command_queue);
    utils::handleError(ret);
    ret = clReleaseKernel(kernel);
    utils::handleError(ret);
    ret = clReleaseMemObject(inputImageBuffer);
    utils::handleError(ret);
    ret = clReleaseMemObject(outputImageBuffer);
    utils::handleError(ret);
    ret = clReleaseMemObject(thetaBuffer);
    utils::handleError(ret);
    ret = clReleaseCommandQueue(command_queue);
    utils::handleError(ret);
    ret = clReleaseContext(context);
    utils::handleError(ret);
}

void
Hough::setupCL(char* fileName)
{
    size_t valueSize;
    char* value;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    utils::handleError(ret);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, &ret_num_devices);
    utils::handleError(ret);
    //tutaj printuje naczym puszczam
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, value, NULL);
    std::cout << value << std::endl;
/* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    utils::handleError(ret);

/* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    utils::handleError(ret);

/* Create Memory Buffer */
//    readInputImage("../images/coins.bmp");
    readInputImage(fileName);
    inputImageBuffer =  clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                                                        width_original * height_original * pixelSize, inputImageData, &ret);
    utils::handleError(ret);
    outputImageBuffer = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, width_original * height_original * pixelSize,0,&ret );
    utils::handleError(ret);
    thetaBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, width_original * height_original * pixelSize, 0, &ret);
    utils::handleError(ret);


}

void
Hough::enqueueKernel(char* fileName, const char* name, bool theta)
{
    createKernel(fileName, name);
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImageBuffer);
    utils::handleError(ret);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImageBuffer);
    utils::handleError(ret);
    if (theta)
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &thetaBuffer);
    size_t globalThreads[] = { width, height };
    size_t localThreads[] = { blockSizeX, blockSizeY };
    cl_event event;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2,NULL,globalThreads,localThreads, 0, NULL, &event );
    utils::handleError(ret);

}
void
Hough::run()
{
    cl_event event;
    ret = clEnqueueReadBuffer(command_queue, outputImageBuffer, CL_TRUE, 0,
                              width * height * pixelSize, outputImageData, 0, NULL, &event);
    /*ret = clEnqueueReadBuffer(command_queue, accuBuffer, CL_TRUE, 0,
                              accu_width * accu_height * pixelSize, accumulator, 0, NULL, &event);*/
    utils::handleError(ret);
    //ret = clFlush(command_queue);
    ret = clWaitForEvents(1, &event);
    utils::handleError(ret);
    writeOutputImage("../output/Canny.bmp");

}
void
Hough::swapBuffers()
{
    std::swap(inputImageBuffer,outputImageBuffer);
}
void
Hough::greyScale()
{
    char greyscale[] = "../kernels/GreyScale_Kernels.cl";
    Hough::enqueueKernel(greyscale, "greyscale_filter");
    Hough::swapBuffers();
}

void
Hough::gauss()
{
    char gaussian[] = "../kernels/Gaussian_Kernels.cl";
    Hough::enqueueKernel(gaussian, "gaussian_filter");
    Hough::swapBuffers();
}
void
Hough::sobel()
{
    char sobel[] = "../kernels/SobelFilter_Kernels.cl";
    Hough::enqueueKernel(sobel,"sobel_filter",true);
    Hough::swapBuffers();
}

void
Hough::max()
{
    char max[] = "../kernels/Max_Kernels.cl";
    Hough::enqueueKernel(max, "Max_filter",true);
    Hough::swapBuffers();
}

void
Hough::hyst()
{
    char hysteresis[] = "../kernels/Hysteresis_Kernels.cl";
    Hough::enqueueKernel(hysteresis, "Hyst_filter");
}
void
Hough::houghTransform(){

    Mat src = imread("../output/Canny.bmp",1);
    //cleanUp();
    Mat src_gray;
    Mat out(src.rows,src.cols,CV_8UC3, Scalar(0,0,0));

    cvtColor( src, src_gray, CV_BGR2GRAY );
    std::vector<Vec3f> circles;

    GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );

    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        circle( out, center, 3, Scalar(0,255,0), -1, 8, 0 );
        circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        circle( out, center, radius, Scalar(0,0,255), 3, 8, 0 );
        circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }


    imwrite("../output/Hough.bmp", out);
    imwrite("../output/Hough_stacked.bmp", src);


}


