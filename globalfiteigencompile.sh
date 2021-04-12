#!/bin/bash
g++ -fopenmp -o globalfiteigen globalfiteigen.cpp `root-config --cflags --libs` -march=native -O3
# g++ -fopenmp -o globalfiteigen globalfiteigen.cpp `root-config --cflags --libs` -march=znver2 -O3
