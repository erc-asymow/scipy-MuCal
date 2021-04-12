#!/bin/bash
# root -l -b -q fillmatricesrdf.c+ && ./globalfiteigencompile.sh && ./globalfiteigen && root -l -b -q applycorrections.c+ && python runFitsGunScaleFull.py
export XRD_PARALLELEVTLOOP=16
root -l -b -q fillmatricesrdf.c+ && ./globalfiteigencompile.sh && ./globalfiteigen && root -l -b -q applycorrections.c+ && python runFitsGunScaleFull.py
