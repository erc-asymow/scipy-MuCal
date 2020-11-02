import os
import multiprocessing

ncpu = multiprocessing.cpu_count()

import cProfile
import mtprof
import pstats

#import ROOT
#import uproot
#import awkward
import uproot4
import awkward1
import numpy as np
import scipy as scipy
from scipy import linalg
import matplotlib.pyplot as plt
import sys
import math
import concurrent.futures
import contextlib


#filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root"
#filename = "/data/bendavid/muoncaldata/test2/testlarge.root"
filename = "/data/bendavid/muoncaldata/test2/testlarge.root:tree"
#filename = "/data/bendavid/muoncaldata/test2/test.root:tree"
#filename = "/eos/cms/store/cmst3/user/bendavid/uproottest/test.root:tree"
treename = "tree"
branches = ["trackPt","globalidxv","gradv","hesspackedv","iidxhesspackedv","jidxhesspackedv"]
#branches = ["trackPt","trackParms","gradv"]


def dotest(e):
  #with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as e:

    it = uproot4.iterate(filename, branches, decompression_exector = e, interpretation_executor=e, step_size=100000)
    #it = uproot.iterate(filename, treename, branches=branches, executor=e, entrysteps=10000)


    for i,arrays in enumerate(it):
        print(i)
        if (i>5):
        break
        #print(arrays)
        #print(arrays.shape)
        #gradv = arrays[b"gradv"]
        #print(gradv.shape)
        
#dotest()

with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as e2:
    results = []
    for i in range(8):
        results.append(e2.submit(dotest, e2))
    for res in results:
        res.result()


#mtprof.run("dotest()",filename="tmpstats")

#p = pstats.Stats("tmpstats")

#p.sort_stats(pstats.SortKey.CUMULATIVE).print_callers()
#p.sort_stats("cumtime").print_callers()
