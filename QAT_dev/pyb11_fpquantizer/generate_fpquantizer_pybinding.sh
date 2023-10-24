#!/bin/bash
c++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) fpquantizer.cpp -o fpquantizer$(python3-config --extension-suffix)