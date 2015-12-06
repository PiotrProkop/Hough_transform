#include <stdio.h>
#include <stdlib.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "Hough.hpp"
//#define MEM_SIZE (128)
//#define MAX_SOURCE_SIZE (0x100000)
//#define GROUP_SIZE 16

//using namespace appsdk;

int main()
{




 

char greyscale[] = "./GreyScale_Kernels.cl";
char gaussian[] = "./Gaussian_Kernels.cl";
char sobel[] = "./SobelFilter_Kernels.cl";
char max[] = "./Max_Kernels.cl";
char hysteresis[] = "./Hysteresis_Kernels.cl";


/* Get Platform and Device Info */



Hough houghTransform;
houghTransform.setupCL();
houghTransform.greyScale();
houghTransform.gauss();
houghTransform.sobel();
houghTransform.max();
houghTransform.hyst();
houghTransform.run();
houghTransform.cleanUp();


return 0;
}






