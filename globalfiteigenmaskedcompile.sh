#!/bin/bash
g++ -fopenmp -o globalfiteigenmasked globalfiteigenmasked.cpp `root-config --cflags --libs` -march=native -O3
