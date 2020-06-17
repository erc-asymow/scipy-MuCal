import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

#forcecpu = True
forcecpu = False

if forcecpu:
    ncpu = multiprocessing.cpu_count()
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ncpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#os.environ["OMP_NUM_THREADS"] = str(ncpu)
#os.environ["OPENBLAS_NUM_THREADS"] = str(ncpu)
#os.environ["MKL_NUM_THREADS"] = str(ncpu)
#os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncpu)
#os.environ["NUMEXPR_NUM_THREADS"] = str(ncpu)

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
#os.environ["XLA_FLAGS"]="--xla_hlo_profile"

#os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=32 inter_op_parallelism_threads=32 xla_force_host_platform_device_count=32"

#os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import jax
import jax.numpy as np
import jax.scipy as scipy
import numpy as onp
import scipy as oscipy
from jax import grad, hessian, jacobian, config
from jax.scipy.special import erf
config.update('jax_enable_x64', True)

import ROOT
import pickle
from termcolor import colored
#from scipy.optimize import minimize, SR1, LinearConstraint, check_grad, approx_fprime
#from scipy.optimize import Bounds
import itertools
from root_numpy import array2hist, fill_hist

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#from fittingFunctionsBinned import defineStatePars, nllPars, defineState, nll, defineStateParsSigma, nllParsSigma, plots, plotsPars, plotsParsBkg, scaleFromModelPars, splitTransformPars, nllBinsFromBinPars, chi2LBins, scaleSqSigmaSqFromBinsPars,scaleSqFromModelPars,sigmaSqFromModelPars,modelParsFromParVector,scaleSigmaFromModelParVector, plotsBkg, bkgModelFromBinPars
from fittingFunctionsBinned import computeTrackLength
from obsminimization import pmin,batch_vmap,jacvlowmemb, batch_accumulate, lbatch_accumulate, pbatch_accumulate,lpbatch_accumulate, pbatch_accumulate_simple,random_subset,lbatch
#from calInput import makeData
import argparse
import functools
import time
import sys
from header import CastToRNode


ROOT.gROOT.ProcessLine(".L src/module.cpp+")
ROOT.gROOT.ProcessLine(".L src/applyCalibration.cpp+")

ROOT.ROOT.EnableImplicitMT()



def makeData(inputFile,ptmin,ptmax):

    RDF = ROOT.ROOT.RDataFrame

    d = RDF('tree',inputFile)
    
    cut = "reco_charge>-2."
    d = d.Filter(cut)

    #d = d.Filter("gen_eta<-2.3")

    d = d.Define('kgen', "gen_curv")
    d = d.Define('eta', "gen_eta")
    d = d.Define('phi', "gen_phi")
    d = d.Define('q', "gen_charge")
    
    #d = d.Filter("eta<-2.3")
    #d = d.Filter("eta>-1.8 && eta<-1.7")
    d = d.Filter(f"kgen<1./{ptmin} && kgen>1./{ptmax} && fabs(eta)<2.4")
    
    d = d.Define("covrphi00","covrphi(0,0)")
    d = d.Define("covrphi01","covrphi(0,1)")
    d = d.Define("covrphi02","covrphi(0,2)")
    d = d.Define("covrphi11","covrphi(1,1)")
    d = d.Define("covrphi12","covrphi(1,2)")
    d = d.Define("covrphi22","covrphi(2,2)")
    
    cols=['eta','phi', 'q','kgen','covrphi00','covrphi01','covrphi02','covrphi11','covrphi12','covrphi22']
    
    data = d.AsNumpy(columns=cols)

    return data


def makeDataDiMuon(inputFile, idx, ptmin,ptmax):

    RDF = ROOT.ROOT.RDataFrame
    
    if idx==1:
        q = 1.0
    elif idx==2:
        q = -1.0

    d = RDF('tree',inputFile)
    
    #cut = f"mcpt{idx} > {ptmin} && fabs(mceta{idx})<2.4"
    #d = d.Filter(cut)
    
    #d = d.Filter(f"mceta{idx}<-2.3")
    #d = d.Filter(f"mceta{idx}<-2.3")

    d = d.Define('kgen', f'1./mcpt{idx}')
    d = d.Define('eta', f'mceta{idx}')
    d = d.Define('phi', f'mcphi{idx}')
    d = d.Define('q', f'{q}')
    
    #d = d.Filter("eta>-1.8 && eta<-1.7")
    #d = d.Filter("eta<-2.3")
    d = d.Filter(f"kgen<1./{ptmin} && kgen>1./{ptmax} && fabs(eta)<2.4")
    

    d = d.Define("covrphi00",f"covrphi{idx}(0,0)")
    d = d.Define("covrphi01",f"covrphi{idx}(0,1)")
    d = d.Define("covrphi02",f"covrphi{idx}(0,2)")
    d = d.Define("covrphi11",f"covrphi{idx}(1,1)")
    d = d.Define("covrphi12",f"covrphi{idx}(1,2)")
    d = d.Define("covrphi22",f"covrphi{idx}(2,2)")

    cols=['eta','phi', 'q','kgen','covrphi00','covrphi01','covrphi02','covrphi11','covrphi12','covrphi22']
    data = d.AsNumpy(columns=cols)

    return data


#def sigmasq(a,c,g1,d1,g2,d2,g3,d3,k,q):

    ##res = a*k**2 + c + g1/(1.+d1*k**2) + g2/(1.+d2*k**2)  + g3/(1.+d3*k**2)
    ##res = a*k**2 + c + g1/k**2 + g2/(1.+d2*k**2)  + g3/(1.+d3*k**2)
    #res = a*k**2 + c + g1/(1.+d1*k**2)  + g2/(1.+d2*k**2) + g3/(1.+d3*k**2)
    ##res = a*k**2 + c + g2/(1.+d2*k**2) + g3/(1.+d3*k**2)
    ##res = a*k**2 + c + g3/(1.+d3*k**2)
    
    #return res

def mvals(parms,ks,qs):
    xglobal = parms[...,:3]
    xlocal = np.reshape(parms[...,3:],(10,-1))
    #xlocal = np.reshape(parms[...,6:],(10,6))
    
    #d1p,d2p,d3p,d1m,d2m,d3m = xglobal
    d1,d2,d3 = xglobal
    ap,cp,g1p,g2p,g3p,am,cm,g1m,g2m,g3m = xlocal
    
    
    d1p = d1
    d2p = d2
    d3p = d3
    
    d1m = d1
    d2m = d2
    d3m = d3
    
    #k = ks
    #res = 0.5*(qs+1.)*(ap*k**2 + cp + g1p/(1.+d1p*k**2)  + g2p/(1.+d2p*k**2) + g3p/(1.+d3p*k**2))
    #res += 0.5*(qs-1.)*(am*k**2 + cm + g1m/(1.+d1m*k**2)  + g2m/(1.+d2m*k**2) + g3m/(1.+d3m*k**2))
    
    #resplus = 0.5*(qs+1.)*(ap*k**2 + cp + g2p/(1.+d2p*k**2) + g3p/(1.+d3p*k**2))
    #res += 0.5*(qs-1.)*(am*k**2 + cm + g2m/(1.+d2m*k**2) + g3m/(1.+d3m*k**2))    
    
    kminus = ks[0]
    kplus = ks[1]
    #resminus = (am*kminus**2 + cm + g2m/(1.+d2m*kminus**2) + g3m/(1.+d3m*kminus**2))    
    #resplus = (ap*kplus**2 + cp + g2p/(1.+d2p*kplus**2) + g3p/(1.+d3p*kplus**2))
    resminus = (am*kminus**2 + cm + g1m/(1.+d1m*kminus**2) + g2m/(1.+d2m*kminus**2) + g3m/(1.+d3m*kminus**2))    
    resplus = (ap*kplus**2 + cp + g1p/(1.+d1p*kplus**2) + g2p/(1.+d2p*kplus**2) + g3p/(1.+d3p*kplus**2))
    res = np.stack((resminus,resplus),axis=0)
    
    #res = (ap*k**2 + cp + g2p/(1.+d2p*k**2) + g3p/(1.+d3p*k**2))
    
    
    #ks = np.expand_dims(ks,axis=-1)
    #qs = np.expand_dims(qs,axis=-1)
    
    #mval = sigmasq(a,c,g1,d1,g2,d2,g3,d3,ks,qs) + 0.*qs
    return res

def nllbinnedmodel(parms, scales, covs, ks, qs):
    
    #a = parms[0]
    #c = parms[1]
    #g1 = parms[2]
    #d1 = parms[3]
    #g2 = parms[4]
    #d2 = parms[5]
    #g3 = parms[6]
    #d3 = parms[7]
    
    #xlocal = parms["local"]
    #xglobal = parms["global"]
    

    #xglobal = parms[:3]
    #xlocal = np.reshape(parms[3:],(5,-1))
    
    
    #xglobal = parms[...,:3]
    #xlocal = np.reshape(parms[...,3:],(2,5,-1))
    
    #d1,d2,d3 = xglobal
    #a,c,g1,g2,g3 = xlocal
    
    nelems = scales.shape[-1]
    xglobal = parms[...,:3]
    xlocal = np.reshape(parms[...,3:],(10,nelems))
    
    #d1p,d2p,d3p,d1m,d2m,d3m = xglobal
    d1,d2,d3 = xglobal
    ap,cp,g1p,g2p,g3p,am,cm,g1m,g2m,g3m = xlocal
    
    
    #d1,d2,d3 = xglobal
    #a,c,g1,g2,g3 = xlocal

    
    #mval = a + c/ks**2 + b/(1.+d*ks**2)
    
    #print("a, d1, scales, errs, ks, qs")
    #print(a.shape)
    #print(d1.shape)
    #print(scales.shape)
    #print(errs.shape)
    #print(ks.shape)
    #print(qs.shape)
    
    mval = mvals(parms,ks,qs)
    
    
    #nll = 0.5*(mval-scales)**2/errs**2
    #nll = 0.5*(mval-scales)**2/errs**2
    diff = mval-scales
    diff = np.expand_dims(diff,-1)
    nll = 0.5*np.swapaxes(diff,-1,-2)@covs@diff
    
    nll = np.sum(nll)
    
    #nll += a**2
    #nll += c**2
    
    sigmag = 5.
    
    nll += 0.5*np.sum(g1p**2)/sigmag**2
    nll += 0.5*np.sum(g2p**2)/sigmag**2
    nll += 0.5*np.sum(g3p**2)/sigmag**2
    nll += 0.5*np.sum(g1m**2)/sigmag**2
    nll += 0.5*np.sum(g2m**2)/sigmag**2
    nll += 0.5*np.sum(g3m**2)/sigmag**2
    
    #nll += 0.5*np.sum(g1p**2)
    #nll += 0.5*np.sum(g1m**2)
    #nll += 0.5*d1p**2
    #nll += 0.5*d1m**2
    
    #nll += 0.5*d1**2
    #nll += g2**2
    #nll += d2**2
    ##nll += g3**2
    #nll += 0.5*d2**2
    
    return nll

   
   
dataDir = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata"

#f = f"{dataDir}/muGunCov_eig.root"
f = f"{dataDir}/muGunCov_covrphi.root"
#fdj = f"{dataDir}/muonTree.root"
#fdj = f"{dataDir}/JpsiToMuMu_JpsiPt8_pythia8_eig.root"
fdj = f"{dataDir}/JpsiToMuMu_JpsiPt8_pythia8_covrphi.root"
#fdz = f"{dataDir}/muonTreeMCZ.root"
fdz = f"{dataDir}/ZJToMuMu_mWPilot.root"
ftrk = f"{dataDir}/trackTreeP.root"

dsets = []

#d = makeDataTrk(ftrk,33.)
#dset = onp.stack( (d["etagen"],d["qgen"], d["kgen"],d["kr"]), axis=-1)
#dsets.append(dset)
#dset = None

ptmax = 150.


d = makeData(f,5.5,ptmax)
dset = onp.stack( (d["eta"],d["q"], d["kgen"],d["covrphi00"],d["covrphi01"],d["covrphi02"],d["covrphi11"],d["covrphi12"],d["covrphi22"]), axis=-1)
d = None
dsets.append(dset)
dset = None

if True:
    #for fdi,minpt in zip([fdj,fdz], [5.5,12.]):
    for fdi,minpt in zip([fdj], [5.5]):
        for idx in [1,2]:
            #d = makeDataDiMuon(fdi,idx,5.5)
            d = makeDataDiMuon(fdi,idx,minpt,ptmax)
            #dset = onp.stack( (d["eta"],d["q"], d["kgen"],d["kr"]), axis=-1)
            dset = onp.stack( (d["eta"],d["q"], d["kgen"],d["covrphi00"],d["covrphi01"],d["covrphi02"],d["covrphi11"],d["covrphi12"],d["covrphi22"]), axis=-1)
            dsets.append(dset)
            dset = None
    
dataset = onp.concatenate(dsets,axis=0)
dsets = None
    
#dsets = None
#deta = dataset[:,0]
#dphi = dataset[:,1]
#print(dataset.shape)

#multibin fit



nEtaBins = 48
etas = onp.linspace(-2.4,2.4, nEtaBins+1, dtype=np.float64)

#nEtaBins = 24
#etas = onp.linspace(-1.2,1.2, nEtaBins+1, dtype=np.float64)

#nEtaBins = 28
#nEtaBins = 24
####nEtaBins = 480
#etas = onp.linspace(-1.4,1.4, nEtaBins+1, dtype=np.float64)

#nEtaBins = 10
#etas = onp.linspace(-2.4,-1.4, nEtaBins+1, dtype=np.float64)

#nEtaBins = 1
#etas = onp.linspace(-2.4,-2.3, nEtaBins+1)
#etas = onp.linspace(-1.8,-1.7, nEtaBins+1)

#etas = np.linspace(-0.1,0., nEtaBins+1)
#etas = np.linspace(-0.3,-0.2, nEtaBins+1)
#etas = np.linspace(-2.3,-2.2, nEtaBins+1)
#etas = np.linspace(-2.2,-2.1, nEtaBins+1)
#etas = np.linspace(-2.3,-2.1, nEtaBins+1)
#etas = np.linspace(1.5,1.6, nEtaBins+1)
#etas = np.linspace(-1.1,-0.9, nEtaBins+1)
#etas = onp.linspace(-1.1,-1., nEtaBins+1)
#etas = np.linspace(-1.02,-1., nEtaBins+1)
#etas = np.linspace(-2.4,-2.2, nEtaBins+1)
#etas = np.linspace(-2.4,-2.0, nEtaBins+1)

#ptmax = 75.
#ptmax = 100.

nPhiBins = 1
phis = onp.linspace(-np.pi,np.pi, nPhiBins+1, dtype=np.float64)

#nkbins = 25
#nkbins = 100
#ks = onp.linspace(1./20., 1./5.5, nkbins+1, dtype=np.float64)

#nptbins = 40
#ks = 1./onp.linspace(150., 5.5, nptbins+1, dtype=np.float64)

#nptbins = 100
#nptbins = 25
#pts = 1./onp.linspace(ptmax, 20., nptbins+1, dtype=np.float64)

#ks = onp.concatenate((pts,ks[1:]),axis=0)

nkbins = 200
#ks = 1./onp.linspace(150.,33.,nkbins+1, dtype=np.float64)
ks = onp.linspace(1./ptmax, 1./5.5, nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./12., nkbins+1, dtype=np.float64)

#ptmax = 150.
#nkbins = 200
#ks = onp.linspace(1./ptmax, 1./5.5, nkbins+1, dtype=np.float64)


nkbins = ks.shape[0]-1


nkbinsfine = 1000
#ksfine = 1./onp.linspace(100., 3.3, nkbinsfine+1, dtype=np.float64)
ksfine = onp.linspace(1./ptmax, 1./5.5, nkbinsfine+1, dtype=np.float64)

qs = onp.array([-1.5,0.,1.5], dtype=np.float64)
#qs = onp.array([0.,1.5], dtype=np.float64)

nBins = nEtaBins*nPhiBins

#dsetbinning = onp.stack((dataset[:,0], dataset[:,1], dataset[:,2], dataset[:,3]/dataset[:,2]), axis=-1)

etacond = onp.logical_and(dataset[:,0]>=etas[0], dataset[:,0]<etas[-1])
kcond = onp.logical_and(dataset[:,2]>=ks[0], dataset[:,2]<ks[-1])
bincond = onp.logical_and(etacond,kcond)
dataset = dataset[onp.where(bincond)]
etacond = None
kcond = None
bincond = None


hdsetsingle = onp.histogramdd(dataset[:,:3], (etas,qs,ks))[0]
hdsetsinglew = onp.histogramdd(dataset[:,:3], (etas,qs,ks), weights=dataset[:,2])[0]

hdsetks = hdsetsinglew/hdsetsingle

print("computing indices for median")
#compute bin indexes
etaidxs = onp.digitize(dataset[:,0],etas) - 1
qidxs = onp.digitize(dataset[:,1],qs) - 1
kidxs = onp.digitize(dataset[:,2],ks) - 1
dataset = dataset[:,3:]

#eta_unique_idxs, etacounts = onp.unique(etaidxs,axis=0,return_counts=True)
etacounts = onp.bincount(etaidxs)
eta_end_idxs = onp.cumsum(etacounts)
eta_start_idxs = eta_end_idxs - etacounts

#compute unique flattened index for bins
fullidxs = onp.ravel_multi_index((etaidxs,qidxs,kidxs),dims=(nEtaBins,2,nkbins))
#etaidxs = None
#qidxs = None
#kidxs = None

print("dataset presort", dataset.shape, fullidxs.shape)
sortidxs = onp.argsort(fullidxs,axis=0)
dataset = dataset[sortidxs]
print("dataset postsort", dataset.shape, fullidxs.shape)

#etaidxtest = etaidxs[sortidxs]
#print(etaidxtest)

#assert()

#count number of values in each bin to allow index computation
#unique_idxs, counts = onp.unique(fullidxs,axis=0,return_counts=True)
counts = onp.bincount(fullidxs)


nelems = 6
nbins = nEtaBins*2*nkbins

def computeMedians(dataset,counts):
    print("computing medians")
    
    end_idxs = onp.cumsum(counts)
    start_idxs = end_idxs-counts

    #for a sorted array, the median is just the middle element
    #varying the index by +-sqrt(nelements)/2 gives a good approximation for the bootstrap or jackknife uncertainty
    nmedian = counts/2
    sigman = onp.sqrt(counts/4)
    nmedianup = nmedian + sigman
    nmediandown = nmedian - sigman

    nmedian = onp.round(nmedian).astype(onp.int32)
    nmedianup = onp.round(nmedianup).astype(onp.int32)
    nmediandown = onp.round(nmediandown).astype(onp.int32)
    
    medians = []
    medianerrs = []
    mediancovs = []
    
    def computebin(ibin):
        bindata = dataset[start_idxs[ibin]:end_idxs[ibin]]
        binnmedian = nmedian[ibin]
        binnmedianup = nmedianup[ibin]
        binnmediandown = nmediandown[ibin]
        binpartitionidxs = onp.argpartition(bindata, (binnmedian,binnmedianup,binnmediandown), axis=0)
        #binpartitionidxs = onp.argsort(bindata, axis=0)
        
        medianidxs = binpartitionidxs[onp.newaxis,binnmedian]
        medianupidxs = binpartitionidxs[onp.newaxis,binnmedianup]
        mediandownidxs = binpartitionidxs[onp.newaxis,binnmediandown]
        
        median = onp.take_along_axis(bindata,medianidxs,axis=0)
        medianerr = 0.5*(onp.take_along_axis(bindata,medianupidxs,axis=0) - onp.take_along_axis(bindata,mediandownidxs,axis=0))

        #inverse partitioning permutation
        idxoffset = onp.empty_like(binpartitionidxs)
        onp.put_along_axis(idxoffset, binpartitionidxs, onp.arange(binpartitionidxs.shape[0])[:,onp.newaxis], axis=0)
        #compute index with respect to median
        idxoffset = idxoffset - binnmedian
        idxoffsetprod = idxoffset[...,onp.newaxis,:]*idxoffset[...,:,onp.newaxis]
        idxoffsetprod = onp.sign(idxoffsetprod)
        #count how many values are on the same vs opposite side of the median for the two quantities
        mediancor = onp.sum(idxoffsetprod,axis=0,keepdims=True)/counts[ibin]
        #print("mediancor:")
        #print(mediancor)
        #mediancor = onp.zeros((1,nelems,nelems),dtype=np.float64)
        #force diagonal correlation to be 1
        onp.fill_diagonal(mediancor[0],1.)
        mediancov = mediancor*medianerr[...,onp.newaxis,:]*medianerr[...,:,onp.newaxis]
                
        return (median, medianerr, mediancov)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        #results = executor.map(computebin,range(nbins),chunksize=100)
        results = executor.map(computebin,range(nbins))
        #results = map(computebin,range(nbins))
    
    #swap order of nested sequences
    medians,medianerrs,mediancovs = [list(result) for result in zip(*results)]

    medians = onp.concatenate(medians,axis=0)
    medianerrs = onp.concatenate(medianerrs,axis=0)
    mediancovs = onp.concatenate(mediancovs,axis=0)

    #reshape back to multidimensional binning
    medians = onp.reshape(medians, (nEtaBins,2,nkbins,nelems))
    medianerrs = onp.reshape(medianerrs, (nEtaBins,2,nkbins,nelems))
    mediancovs = onp.reshape(mediancovs, (nEtaBins,2,nkbins,nelems,nelems))
        
    return medians, medianerrs, mediancovs

etasL = etas[:-1]
etasH = etas[1:]
etasC = 0.5*(etas[1:]+etas[:-1])

phisL = phis[:-1]
phisH = phis[1:]
phisC = 0.5*(phis[1:]+phis[:-1])

fg = jax.value_and_grad(nllbinnedmodel)
h = jax.hessian(nllbinnedmodel)

fg = jax.jit(fg)
h = jax.jit(h)

jacmvals = jax.jit(jax.jacfwd(lambda *args: mvals(*args)))


@jax.jit
def diagnonalizebin(bindata, covval):
    n = bindata.shape[0]
    print("computing eigendecomposition")
    cov = np.empty((3,3),dtype=np.float64)
    #cov[onp.triu_indices(3)] = covvals[ieta]
    cov = jax.ops.index_update(cov,jax.ops.index[np.triu_indices(3)],covval)
    cov = jax.ops.index_update(cov,jax.ops.index[np.triu_indices(3)[1],np.triu_indices(3)[0]],covval)
    e,v = np.linalg.eigh(cov)
    
    print("constructing intermediate array")
    mdset = np.empty((n,3,3),dtype=np.float64)
    #mdset[:,onp.triu_indices(3)[0],onp.triu_indices(3)[1]] = dataset[idx:idx+count]
    #mdset[:,onp.triu_indices(3)[1],onp.triu_indices(3)[0]] = dataset[idx:idx+count]
    #bindata = dataset[idx:idx+count]
    mdset = jax.ops.index_update(mdset,jax.ops.index[:,np.triu_indices(3)[0],np.triu_indices(3)[1]],bindata)
    mdset = jax.ops.index_update(mdset,jax.ops.index[:,np.triu_indices(3)[1],np.triu_indices(3)[0]],bindata)
    
    #print("copying intermediate arrays to gpu")
    #mdset = jax.device_put(mdset)
    #v = jax.device_put(v)
    
    print(v)
    
    
    print("doing matrix multiplications")
    mdset = v.T @ mdset @ v
    print(mdset)
    print("filling output dataset")
    return mdset[:,np.triu_indices(3)[0],np.triu_indices(3)[1]]
    #mdset = None

def diagnonalize(dataset,eta_start_idxs,etacounts, covvals):
    print("doing diagonalization")
    
    #vs = []
    #for ieta,(idx,count) in enumerate(zip(eta_start_idxs,etacounts)):
        #print("computing eigendecomposition")
        #cov = np.empty((3,3),dtype=onp.float64)
        ##cov[onp.triu_indices(3)] = covvals[ieta]
        #cov = jax.ops.index_update(cov,jax.ops.index[np.triu_indices(3)],covvals[ieta])
        #e,v = np.linalg.eigh(cov,UPLO="U")
        
        #print("constructing intermediate array")
        #mdset = np.empty((count,3,3),dtype=np.float64)
        ##mdset[:,onp.triu_indices(3)[0],onp.triu_indices(3)[1]] = dataset[idx:idx+count]
        ##mdset[:,onp.triu_indices(3)[1],onp.triu_indices(3)[0]] = dataset[idx:idx+count]
        #bindata = dataset[idx:idx+count]
        #mdset = jax.ops.index_update(mdset,jax.ops.index[:,np.triu_indices(3)[0],onp.triu_indices(3)[1]],bindata)
        #mdset = jax.ops.index_update(mdset,jax.ops.index[:,np.triu_indices(3)[1],onp.triu_indices(3)[0]],bindata)
        
        ##print("copying intermediate arrays to gpu")
        ##mdset = jax.device_put(mdset)
        ##v = jax.device_put(v)
        
        #print(v)
        
        #print("doing matrix multiplications")
        #mdset = v.T @ mdset @ v
        #print(mdset)
        #print("filling output dataset")
        #dataset[idx:idx+count] = mdset[:,np.triu_indices(3)[0],np.triu_indices(3)[1]]
        #mdset = None
        
        ##vs.append(v)

    
    #diagnonalizebin = jax.jit(diagnonalizebin)
    for ieta,(idx,count) in enumerate(zip(eta_start_idxs,etacounts)):
        bindata = dataset[idx:idx+count]
        dataset[idx:idx+count] = diagnonalizebin(bindata,covvals[ieta])
    
    print("done diagonalization")

    #vs = onp.stack(vs,axis=0)
    #print(vs)


#diagnonalize = jax.jit(diagnonalize, static_argnums = (2,))

def dofit(xbinned,covsbinned):

    xs = []
    covs = []
    xerrs = []

    scalesigmamodels = []
    xerrsmodels = []

    scalesigmamodelsfine = []
    xerrsmodelsfine = []

    xbinned = jax.device_put(xbinned)
    covsbinned = jax.device_put(covsbinned)

    for ieta,(etaL, etaH, etaC) in enumerate(zip(etasL, etasH, etasC)):
        scales = xbinned[ieta,...]
        #errs = errsbinned[ieta,...]
        fitcovs = covsbinned[ieta,...]
        fitcovs = np.linalg.inv(fitcovs)
        ksfit = hdsetks[ieta,...,np.newaxis]
        
        
        
        #print("scales errs ks")
        #print(scales.shape)
        #print(errs.shape)
        #print(ksfit.shape)
        
        #assert(0)
        
        nelems = scales.shape[-1]
        
        #a = 0.
        #c = 0.
        #g1 = 0.
        #g2 = 0.
        #g3 = 0.
        
        d1 = 16.
        d2 = 100.
        d3 = 1000.
        
        xglobal = np.stack((d1,d2,d3)).astype(np.float64)
        
        xlocal = np.zeros((10,nelems),dtype=np.float64)
        
        #x = {}
        #x["local"] = xlocal
        #x["global"] = xglobal
        
        x = np.concatenate((xglobal, xlocal.flatten()))
        
        nparms = x.shape[0]
    
        qs = np.array([-1.,1.],dtype=np.float64)
        qs = qs[:,np.newaxis, np.newaxis]
        
        x = pmin(fg, x, (scales,fitcovs,ksfit,qs), doParallel=False, jac=True, h=None,xtol=1e-14,edmtol=1e-3)
        x = pmin(fg, x, (scales,fitcovs,ksfit,qs), doParallel=False, jac=True, h=h)
        
        #print("finished fit")
        #assert(0)
        
        print("computing hess")
        hess = h(x, scales, fitcovs,ksfit,qs)
        #hess = np.eye(x.shape[0])
        cov = np.linalg.inv(hess)
        #xerr = np.sqrt(np.diag(cov))
        

        #x = x*parmscale
        #cov = cov*parmscale[:,np.newaxis]*parmscale[np.newaxis,:]
        
        ksfinescale = ksfine[np.newaxis,:,np.newaxis]*np.ones((2,1,1),dtype=np.float64)
            
        iscalesigmamodelfine = mvals(x,ksfinescale,qs)
        jacmodelfine = jacmvals(x,ksfinescale,qs)
        jacmodelfine = np.reshape(jacmodelfine,(-1,nparms))
        print("iscalesigmamodelfine, jacmodelfine, jacmodelfine.T, cov")
        print(iscalesigmamodelfine.shape)
        print(jacmodelfine.shape)
        print(jacmodelfine.T.shape)
        print(cov.shape)
        #covmodelfine = np.matmul(jacmodelfine,np.matmul(cov,jacmodelfine.T))
        covmodelfine = jacmodelfine @ cov @ jacmodelfine.T
        #covmodelfine = cov @ jacmodelfine.T
        #covmodelfine  = jacmodelfine @ cov
        ierrsmodelfine = np.sqrt(np.diag(covmodelfine))
        ierrsmodelfine = np.reshape(ierrsmodelfine,iscalesigmamodelfine.shape)
            
        #iscalesigmamodelfine = scalesigma(x,etaC, ksfine[np.newaxis,:])
        #jacmodelfine = jacscalesigma(x,etaC,ksfine[np.newaxis,:])
        #covmodelfine = np.matmul(jacmodelfine,np.matmul(cov,jacmodelfine.T))
        #ierrsmodelfine = np.sqrt(np.diag(covmodelfine))
        #ierrsmodelfine = np.reshape(ierrsmodelfine,iscalesigmamodelfine.shape)
        
        scalesigmamodelsfine.append(iscalesigmamodelfine)
        xerrsmodelsfine.append(ierrsmodelfine)
            
        xerr = np.sqrt(np.diag(cov))
        
        cor = cov/xerr[:,np.newaxis]/xerr[np.newaxis,:]
        
        print(x)
        print(xerr)
        print(cor)
        
        #assert(0)
        
        xs.append(x)
        covs.append(cov)
        xerrs.append(xerr)
        
    xs = np.stack(xs, axis=0)
    xerrs = np.stack(xerrs, axis=0)
    covs = np.stack(covs,axis=0)
    scalesigmamodelfine = np.stack(scalesigmamodelsfine, axis=0)
    errsmodelfine = np.stack(xerrsmodelsfine, axis=0)
    
    return xs,xerrs,covs,scalesigmamodelfine,errsmodelfine

medians,medianerrs,mediancovs = computeMedians(dataset,counts)

for idiag in range(0):
    #average over charges
    covvals = onp.mean(medians[:,:,-1,:],axis=1)
    diagnonalize(dataset,eta_start_idxs,etacounts,covvals)
    medians,medianerrs,mediancovs = computeMedians(dataset,counts)

print(medians.shape)
print(medians)

for idiag in range(0):
    #scale for fits
    xbinned = 1e6*medians
    errsbinned = 1e6*medianerrs
    covsbinned = 1e6**2*mediancovs
        
    xs,xerrs,covs,scalesigmamodelfine,errsmodelfine = dofit(xbinned,covsbinned)
    
    nelems = xbinned.shape[-1]
    xglobal = xs[...,:3]
    xlocal = np.reshape(xs[...,3:],(-1,10,nelems))
    
    #d1,d2,d3 = xglobal
    #a,c,g1,g2,g3 = xlocal
    
    ap = xlocal[...,0,:]
    am = xlocal[...,5,:]
    #average over charges
    covvals = 0.5*(ap+am)
    
    diagnonalize(dataset,eta_start_idxs,etacounts,covvals)
    medians,medianerrs,mediancovs = computeMedians(dataset,counts)
                     
#scale for fits
xbinned = 1e6*medians
errsbinned = 1e6*medianerrs
covsbinned = 1e6**2*mediancovs

#good_idxs = (0,1,2,4)
###good_idxs = (1,2,4)
#####good_idxs = (0,1,2)
#####good_idxs = (3,4,5)
####good_idxs = (5,)
###good_idxs = (0,1,2,0)
#xbinned = xbinned[...,good_idxs]
#errsbinned = errsbinned[...,good_idxs]
#covsbinned = covsbinned[...,good_idxs,:]
#covsbinned = covsbinned[...,:,good_idxs]
    
xs,xerrs,covs,scalesigmamodelfine,errsmodelfine = dofit(xbinned,covsbinned)
                     
onp.savez_compressed("unbinnedfiterrsmed.npz",
                     xbinned = xbinned,
                     errsbinned = errsbinned,
                     hdsetks = hdsetks,
                     #scalesigmamodel = scalesigmamodel,
                     scalesigmamodelfine = scalesigmamodelfine,
                     #errsmodel = errsmodel,
                     errsmodelfine = errsmodelfine,
                     etas = etas,
                     ks = ks,
                     ksfine = ksfine,
                     xs = xs,
                     xerrs = xerrs,
                     covs = covs)
                     



