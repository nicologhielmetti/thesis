#!/bin/bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) exp.cpp -o exp$(python3-config --extension-suffix)