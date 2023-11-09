#!/bin/bash
c++ -O3 -Wall -shared -std=c++20 -fPIC $(python3 -m pybind11 --includes) ulp.cpp -o ulp$(python3-config --extension-suffix)