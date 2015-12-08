#!/bin/bash

mkdir -p ../bin
mkdir -p ../output
g++ ../src/main.cpp ../src/Hough.cpp -L ../../opencv/lib -I ../include -lOpenCL -lopencv_ocl -lopencv_highgui -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_imgproc -lopencv_objdetect -std=gnu++0x -o ../bin/hough
