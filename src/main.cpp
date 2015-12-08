#include <stdio.h>
#include <stdlib.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "Hough.hpp"
#include <stdio.h>

int main(int argc, char** argv)
{

if(argc != 2) {
    std::cout << " First parameter is location of image" << std::endl;
    exit(1);
}

char* fileName = argv[1];


Hough houghTransform;
houghTransform.setupCL(fileName);
houghTransform.greyScale();
houghTransform.gauss();
houghTransform.sobel();
houghTransform.max();
houghTransform.hyst();
houghTransform.run();
houghTransform.houghTransform();

houghTransform.cleanUp();


return 0;
}






