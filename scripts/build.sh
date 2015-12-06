#!/bin/bash

mkdir -p ../bin
mkdir -p ../output
g++ ../src/main.cpp ../src/Hough.cpp -lOpenCL -std=gnu++0x -o ../bin/hough
