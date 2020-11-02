import os
import multiprocessing

ncpu = multiprocessing.cpu_count()

import cProfile
import mtprof
import pstats

import ROOT

import uproot4
import awkward1 as ak
import numpy as np
import scipy as scipy
from scipy import linalg
import matplotlib.pyplot as plt
import sys
import math
import concurrent.futures
import contextlib

import timeit
from numba import jit, prange


#filenameinfo = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGradsParmInfo.root:tree"
#with uproot4.open(filenameinfo) as infotree:
    #nglobal = infotree.num_entries
    
#print(type(nglobal))

filenameinfo = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGradsParmInfo.root"
finfo = ROOT.TFile.Open(filenameinfo)
runtree = finfo.tree
nglobal = np.int64(runtree.GetEntries())
print(nglobal)

grad = np.zeros((nglobal,), dtype=np.float64)
hess = np.zeros((nglobal,nglobal), dtype=np.float64)

gradflat = grad.ravel()
hessflat = hess.ravel()

@jit(nopython=True, nogil=True, parallel=True, cache=True)
def scatteraddjagged(arr, idxs, updates, counts):
    start = 0
    for count in counts:
        arr[idxs[start:start+count]] += updates[start:start+count]
        #for j in range(count):
            #arr[idxs[start+j]] += updates[start+j]
        start += count


@jit(nopython=True, nogil=True, parallel=False)
def add_in_place_nodup(arr, idxs, updates):
  for i in prange(idxs.shape[0]):
    arr[idxs[i]] += updates[i]


@jit(nopython=True, nogil=True, parallel=True)
def add_in_place_sorted(arr, idxs, updates):
  #print("add_in_place")
  #sortidxs = np.argsort(idxs, kind="mergesort")
  #idxs = idxs[sortidxs]
  #updates = updates[sortidxs]
  
  #idxs_split = np
  

  #print("done final sort")
  
  
  updates = updates.astype(np.float64)
  
  mask = np.empty(idxs.shape, dtype=np.bool_)
  mask[:1] = True
  mask[1:] = idxs[1:] != idxs[:-1]
  uniqueidxs = idxs[mask]
  binidxs = mask.astype(np.intp).cumsum()

  countidxs = np.concatenate( (np.nonzero(mask)[0], np.array((mask.size,),dtype=np.intp) ))
  counts = countidxs[1:] - countidxs[:-1]
  startidxs = counts.cumsum() - counts
  
  #print("uniqueidxs.shape", uniqueidxs.shape)
  
  #print("uniquesums")
  #uniquesums = np.empty(uniqueidxs.shape, dtype=np.float64)
  #for i in prange(uniqueidxs.shape[0]):
    #startidx = startidxs[i]
    #count = counts[i]
    #uniquesums[i] = np.sum(updates[startidx:startidx+count])    
  
  #print("final sum")
  for i in prange(uniqueidxs.shape[0]):
    startidx = startidxs[i]
    count = counts[i]
    uniquesum = np.sum(updates[startidx:startidx+count])
    #uniquesum = 0.
    #for j in prange(count):
      #uniquesum += updates[startidx + j]
    arr[uniqueidxs[i]] += uniquesum
    #arr[uniqueidxs[i]] += uniquesums[i]
  #print("done add in place")

@jit(nopython=True, nogil=True, parallel=False)
def add_in_place(arr, idxs, updates):
  Ntotal = idxs.shape[0]
  Nsections = ncpu
  #Nsections = 4
  Neach_section, extras = divmod(Ntotal, Nsections)
  #section_sizes = ([0] +
                    #extras * [Neach_section+1] +
                    #(Nsections-extras) * [Neach_section])
  section_sizes = ([0] +
                    [Neach_section + 1 for i in range(extras)] +
                    [Neach_section for i in range(Nsections-extras)])
  div_points = np.array(section_sizes, dtype=np.intp).cumsum()
  
  subidxs = []
  subupdates = []
  
  #subidxs = [idxs[div_points[i]:div_points[i+1]] for i in range(Nsections)]
  #subupdates = [updates[div_points[i]:div_points[i+1]] for i in range(Nsections)]
  
  #for i in range(Nsections):
    #subidx = idxs[div_points[i]:div_points[i+1]]
    #subupdate = updates[div_points[i]:div_points[i+1]]
    
    #subidxs.append(subidx)
    #subupdates.append(subupdates)
  
  #subsortidxs = [np.argsort(idxs[div_points[i]:div_points[i+1]]) for i in prange(Nsections)]
  
  #for i in prange(Nsections):
    #idxs[div_points[i]:div_points[i+1]] = idxs[div_points[i]:div_points[i+1]][subsortidxs[i]]
    #updates[div_points[i]:div_points[i+1]] = updates[div_points[i]:div_points[i+1]][subsortidxs[i]]
  
  #for i in prange(Nsections):
    #subidx = subidxs[i]
    #subsortidxs = np.argsort(subidx)

  
  #print("subsorts")
  print(div_points)
  print(idxs.shape)
  print(idxs)
  for i in prange(Nsections):
    subidxs = idxs[div_points[i]:div_points[i+1]]
    subupdates = updates[div_points[i]:div_points[i+1]]
    subsortidxs = np.argsort(subidxs, kind="mergesort")

    subidxs[...] = subidxs[subsortidxs]
    subupdates[...] = subupdates[subsortidxs]

    
  #print("final sort")
  #print(idxs)
  sortidxs = np.argsort(idxs, kind="mergesort")
  idxs = idxs[sortidxs]
  updates = updates[sortidxs]
  #print("done final sort")
  
  add_in_place_sorted(arr, idxs, updates)


#def add_in_place(arr, idxs, updates):
  #print("sort")
  #sortidxs = np.argsort(idxs)
  #idxs = idxs[sortidxs]
  #updates = updates[sortidxs]
  
  #print("add_in_place_sorted")
  #add_in_place_sorted(arr, idxs, updates)
  #print("done add_in_place_sorted")
  

@jit(nopython=True, nogil=True, parallel=True)
def handlenp(idxsflat, iidxsflat, jidxsflat, lgradflat, lhessflat, gradflat, hessflat):
  ##idxsflat = idxmap[idxsflat]
  #idxsflat = idxsflat.astype(np.int64)
  
  #iidxsflat = idxmap[iidxsflat]
  #iidxsflat = iidxsflat.astype(np.int64)
  
  #jidxsflat = idxmap[jidxsflat]
  #jidxsflat = jidxsflat.astype(np.int64)  
  
  #fill only upper triangular
  #switch = iidxsflat > jidxsflat
  #iidxsflat, jidxsflat = (np.where(switch, jidxsflat, iidxsflat), np.where(switch,iidxsflat,jidxsflat))

  hessidxsflat = nglobal*iidxsflat + jidxsflat
  
  #print("add")
  add_in_place(gradflat, idxsflat, lgradflat)
  add_in_place(hessflat, hessidxsflat, lhessflat)
  
  #add_in_place_nodup(gradflat, idxsflat, lgradflat)
  #add_in_place_nodup(hessflat, hessidxsflat, lhessflat)

#filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTreeGrads.root"
#filename = "/data/bendavid/muoncaldata/test2/testlarge.root"
filename = "/data/bendavid/muoncaldata/test3/testlz4large.root:tree"
#filename = "/data/bendavid/muoncaldata/test3/testlz4large.root"
#filename = "/data/bendavid/muoncaldata/test3/testzliblarge.root"
#filename = "/data/bendavid/muoncaldata/test3/testzliblarge.root:tree"
#filename = "/data/bendavid/muoncaldata/test3/testzstdlarge.root:tree"
#filename = "/data/bendavid/muoncaldata/test2/test.root:tree"
#filename = "/eos/cms/store/cmst3/user/bendavid/uproottest/test.root:tree"
treename = "tree"
branches = ["trackPt","globalidxv","gradv","hesspackedv","iidxhesspackedv","jidxhesspackedv"]
#branches = ["trackPt","globalidxv","gradv","hesspackedv"]
#branches = ["trackPt","trackParms","gradv"]

stepsize = 100000

#decompression_executor
def dotest(e=None, start=0, imax=None):
  #with concurrent.futures.ThreadPoolExecutor(max_workers=1) as esingle:

    #with concurrent.futures.ProcessPoolExecutor(max_workers=1) as eproc:
    #it = uproot.iterate(filename, treename, branches=branches, executor=e, entrysteps=10000)
    #tree = uproot4.open(filename)
    #tree = f["tree"]
    #print(tree)
    
    
    
    #assert(0)
    
    if e is None:
        e = uproot4.TrivialExecutor()
    
    #it = uproot4.iterate(filename, branches, decompression_executor = e, interpretation_executor=e, step_size=100000)
    
    #arrays=tree.arrays(branches, decompression_executor = e, interpretation_executor=e, entry_start=start, entry_stop=start+stepsize)
    #return arrays["gradv"][0][0]
    #return ak.to_numpy(ak.flatten(arrays["hesspackedv"]))
    
    
    #results = []
    #with concurrent.futures.ProcessPoolExecutor(max_workers=ncpu) as eblocks:
        #for i in range(2):
            #res = eblocks.submit(tree.arrays, branches, decompression_executor = e, interpretation_executor=e, entry_start=i*stepsize, entry_stop=(i+1)*stepsize)
            #results.append(res)
        
    #for res in results:
        #print(res.result())
    #arrays = tree.arrays(branches, decompression_executor = e, interpretation_executor=e, entry_start=0, entry_stop=100000)
    #print(arrays.shape)
    #print(len(tree))
    
    with uproot4.open(filename) as tree:
        nentries = tree.num_entries
        #print(nentries)
        #results = []
        
        #for i,arrays in enumerate(it):
            #results.append(True)
            ##results.append(arrays["gradv"][0][0])
        #return results
        stepsize=100000
        for istart in range(0,nentries,stepsize):
            arrays=tree.arrays(branches, decompression_executor = e, interpretation_executor=e, entry_start=istart, entry_stop=istart+stepsize)
            
            idxs = arrays["globalidxv"]
            lgrad = arrays["gradv"]
            
            counts = ak.count(idxs, axis=1)
            idxs = idxs.layout.content
            lgrad = lgrad.layout.content
            
            idxs = ak.to_numpy(idxs)
            lgrad = ak.to_numpy(lgrad)
            
            
            
            #idxs = ak.to_numpy(idxs).astype(np.int64)

            
            #print(offsets)
            #print(idxs)
            #print(lgrad)
            #assert(0)
            idxs = idxs.astype(np.int64)
            scatteraddjagged(gradflat, idxs, lgrad, counts)
            
            iidxs = arrays["iidxhesspackedv"]
            jidxs = arrays["iidxhesspackedv"]
            lhess = arrays["hesspackedv"]
            
            counts = ak.count(iidxs, axis=1)
            iidxs = iidxs.layout.content
            jidxs = jidxs.layout.content
            lhess = lhess.layout.content
            
            iidxs = ak.to_numpy(iidxs)
            jidxs = ak.to_numpy(jidxs)
            lhess = ak.to_numpy(lhess)
            
            
            iidxs = iidxs.astype(np.int64)
            jidxs = jidxs.astype(np.int64)
            hessidxs = nglobal*iidxs + jidxs

            
            scatteraddjagged(hessflat, hessidxs, lhess, counts)

            
            #handlenp(idxs, iidxs, jidxs, lgrad, lhess, gradflat, hessflat)
            
            #idxs = arrays["hesspackedv"].layout.content
            #idxsalt = ak.to_numpy(ak.flatten(arrays["hesspackedv"]))
            #print(idxs.shape, idxsalt.shape)
            #idxsnp = idxs.layout.content
            #print(idxsnp.shape)
            ##print(vars(idxs))
            #print(type(idxsnp))
            #for idx in idxs:
                #idxnp = ak.to_numpy(idx)
                #uniqueidx = np.unique(idxnp)
                ##print(uniqueidx.shape, idxnp.shape)
                #assert(uniqueidx.shape[0]==idxnp.shape[0])
            print(istart)
        return True
        
    

    #i=0
    #for arrays in it:
        #print(i)
        #i+=1

    #for i,arrays in enumerate(it):
        #print(i)
        #if (imax is not None and i>imax):
            #break
        ##print(arrays)
        ##print(arrays.shape)
        ##gradv = arrays[b"gradv"]
        ##print(gradv.shape)
    
#dotest()

def domultitest():
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as e2:
        results = []
        #results = e2
        #for j in range(10):
        for i in range(16):
            results.append(e2.submit(dotest, None,  0))
        #while True:
        for res in results:
            ##if res.done():
            print(res.result())

def dosingletest(imax=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as e2:
    #with uproot4.TrivialExecutor() as e2:
        dotest(e2, 0, imax)

#domultitest()
#assert(0)
#dosingletest(0)
#dosingletest()
#res = timeit.timeit(domultitest, number = 1)
#print(res)

dosingletest()
#for i in range(5):
    #dosingletest()
assert(0)
#mtprof.run("domultitest()",filename="tmpstats")
mtprof.run("dosingletest()",filename="tmpstats")


p = pstats.Stats("tmpstats")

#p.sort_stats(pstats.SortKey.CUMULATIVE).print_callers()
#p.sort_stats("cumtime").print_callers()
p.sort_stats("tottime").print_callers()
#p.sort_stats("tottime").print_stats()
