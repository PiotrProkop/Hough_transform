#!/bin/bash

mkdir -p ../bin
g++ ../src/main.cpp ../src/Hough.cpp -lOpenCL -std=gnu++0x -o ../bin/hough
