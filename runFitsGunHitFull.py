import os
import multiprocessing

#forcecpu = True
forcecpu = False

if forcecpu:
    ncpu = multiprocessing.cpu_count()
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ncpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda"

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
from jax import grad, hessian, jacobian, config
from jax.scipy.special import erf
config.update('jax_enable_x64', True)

import ROOT
import pickle
from termcolor import colored
#from scipy.optimize import minimize, SR1, LinearConstraint, check_grad, approx_fprime
#from scipy.optimize import Bounds
import itertools
from root_numpy import array2hist, fill_hist, hist2array

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#from fittingFunctionsBinned import defineStatePars, nllPars, defineState, nll, defineStateParsSigma, nllParsSigma, plots, plotsPars, plotsParsBkg, scaleFromModelPars, splitTransformPars, nllBinsFromBinPars, chi2LBins, scaleSqSigmaSqFromBinsPars,scaleSqFromModelPars,sigmaSqFromModelPars,modelParsFromParVector,scaleSigmaFromModelParVector, plotsBkg, bkgModelFromBinPars
from fittingFunctionsBinned import computeTrackLength
from obsminimization import pmin,batch_vmap,jacvlowmemb, batch_accumulate, lbatch_accumulate, pbatch_accumulate,lpbatch_accumulate, pbatch_accumulate_simple,random_subset
#from calInput import makeData
import argparse
import functools
import time
import sys
from header import CastToRNode

ROOT.ROOT.EnableImplicitMT()

#filename = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprec/globalcor_*.root"
#filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprec/globalcor_0.root"

#filename = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprecsimplecpe//globalcor_*.root"
#filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprecsimplecpe//globalcor_0.root"

filename = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantint4fullprecisegen//globalcor_*.root"
filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantint4fullprecisegen//globalcor_0.root"


finfo = ROOT.TFile.Open(filenameinfo)
runtree = finfo.runtree

def logsigpdfbinned(mu,sigma,krs):
    
    krs = krs[np.newaxis,np.newaxis,np.newaxis,:]
    width = krs[...,1:] - krs[...,:-1]
    #krl = krs[...,0]
    #krh = krs[...,-1]
    
    #krl = krl[...,np.newaxis]
    #krh = krh[...,np.newaxis]
    
    kr = 0.5*(krs[...,1:] + krs[...,:-1])

    #sigma = 2e-3
    
    alpha = 3.0
    alpha1 = alpha
    alpha2 = alpha
    
    A1 = np.exp(0.5*alpha1**2)
    A2 = np.exp(0.5*alpha2**2)
    
    t = (kr - mu)/sigma
    tleft = np.minimum(t,-alpha1)
    tright = np.maximum(t,alpha2)
    tcore = np.clip(t,-alpha1,alpha2)
    #tleft = np.where(t<-alpha1, t, -alpha1)
    #tright = np.where(t>=alpha2, t, alpha2)
    
    #pdfcore = np.exp(-0.5*tcore**2)
    #pdfleft = A1*np.exp(alpha1*tleft)
    #pdfright = A2*np.exp(-alpha2*tright)
    
    pdfcore = -0.5*tcore**2
    pdfleft = np.log(A1) + alpha1*tleft
    pdfright = np.log(A2) - alpha2*tright
    
    logpdf = np.where(t<-alpha1, pdfleft, np.where(t<alpha2, pdfcore, pdfright))
    
    I = np.sum(width*np.exp(logpdf),axis=-1,keepdims=True)
    

    
    #Icore = (scipy.special.ndtr(alpha2) - scipy.special.ndtr(-alpha1))*sigma*np.sqrt(2.*np.pi)
    #Ileft = (sigma/alpha1)*np.exp(-0.5*alpha1**2)
    #Iright = (sigma/alpha2)*np.exp(-0.5*alpha2**2)
    
    #I = Icore + Ileft + Iright
    
    #print("I")
    #print(I)
    
    return logpdf - np.log(I)

def nllbinned(parms, dataset, krs):
    mu = parms[...,0]
    sigma = parms[...,1]
    
    #sigma = np.sqrt(sigmasq)
    sigma = np.where(sigma>0.,sigma,np.nan)
    
    #mu = 1. + 0.1*np.tanh(mu)
    #sigma = 1e-3*(1. + np.exp(sigma))
    
    
    mu = mu[...,np.newaxis]
    sigma = sigma[...,np.newaxis]
    
    logpdf = logsigpdfbinned(mu,sigma,krs)
    #logpdf = loggauspdfbinned(mu,sigma,krs)
    
    nll = -np.sum(dataset*logpdf, axis=-1)
    #nll += np.squeeze(sigma,axis=-1)**2
    #nll += sigma**2
    
    return nll

def scale(A,e,M,R,k,q):
    return q*k*A + e*k**2 + M + q*R

def sigmasq(a, c, k):
    return c + a*k**2

def scalesigma(parms, qs, ks):
    A = parms[..., 0, np.newaxis, np.newaxis]
    e = parms[..., 1, np.newaxis, np.newaxis]
    M = parms[..., 2, np.newaxis, np.newaxis]
    R = parms[..., 3, np.newaxis, np.newaxis]
    a = parms[..., 4, np.newaxis, np.newaxis]
    c = parms[..., 5, np.newaxis, np.newaxis]
    
    qs = qs[np.newaxis, :, np.newaxis]
    ks = ks[np.newaxis, np.newaxis, :]
    
    scaleout = scale(A,e,M,R,ks,qs)
    sigmasqout = sigmasq(a,c,ks)
    sigmaout = np.sqrt(sigmasqout)
    sigmaout = np.ones_like(qs)*sigmaout
    
    return np.stack([scaleout, sigmaout], axis=-1)

jacscalesigma = jax.jit(jax.jacfwd(lambda *args: scalesigma(*args).flatten()))

def nllbinnedmodel(parms, dataset, qs, ks, krs):
    #A = parms[..., 0, np.newaxis, np.newaxis, np.newaxis]
    #e = parms[..., 1, np.newaxis, np.newaxis, np.newaxis]
    #M = parms[..., 2, np.newaxis, np.newaxis, np.newaxis]
    #R = parms[..., 3, np.newaxis, np.newaxis, np.newaxis]
    #a = parms[..., 4, np.newaxis, np.newaxis, np.newaxis]
    #c = parms[..., 5, np.newaxis, np.newaxis, np.newaxis]
    
    #qs = qs[np.newaxis, :, np.newaxis, np.newaxis]
    #ks = ks[np.newaxis, np.newaxis, :, np.newaxis]

    scalesigmaout = scalesigma(parms, qs, ks)
    mu = scalesigmaout[...,0]
    sigma = scalesigmaout[...,1]
    
    mu = mu[...,np.newaxis]
    sigma = sigma[...,np.newaxis]
    
    #mu = scale(A,e,M,R,ks,qs)
    #sigsq = sigmasq(a,c,ks)
    #sigma = np.sqrt(sigsq)
    
    logpdf = logsigpdfbinned(mu,sigma,krs)
    
    nll = -np.sum(dataset*logpdf, axis=(-1,-2,-3))
    
    return nll
    
    


parmset = set()
parmlistfull = []
for parm in runtree:
    iidx = parm.iidx
    parmtype = runtree.parmtype
    #ieta = math.floor(runtree.eta/0.1)
    #ieta = runtree.stereo
    #iphi = math.floor(runtree.phi/(math.pi/8.))
    #iphi = math.floor(runtree.phi/(math.pi/1024.))
    #iphi = 0
    #ieta = math.floor(runtree.eta/1.0)
    subdet = runtree.subdet
    layer = abs(runtree.layer)
    stereo = runtree.stereo
    
    #if (subdet==0 and layer==1):
        #print("pxb1 module:")
        #print(iidx)
        #print(parm.rawdetid)
        
    #if (parmtype==2 and subdet==0 and layer==1) :
    #if (parmtype!=2):
    #if (False):
    #if (parmtype==2):
    #if (parmtype!=2 or (subdet==0 and layer==1)) :
    #if (parmtype>2):
    #if (abs(gradfull[ieta,0])<1e-9):
        #parmtype = -1
        #subdet = -1
        #layer = -1
        #ieta = 0
        #iphi = 0
    ##elif (parmtype==3):
    #else:
        #ieta = iidx
        #iphi = 0
      
  #if parmtype>1:
    #if (subdet==3 and layer==7) or (subdet==5 and layer==9):
      #subdet = -1
      #layer = -1
      #ieta = 0
      #parmtype = -1
    key = (subdet, layer, stereo)
    #key = (parmtype, subdet, layer, (ieta,iphi))
    parmset.add(key)
    parmlistfull.append(key)
  
parmlist = list(parmset)
parmlist.sort()

parmmap = {}
for iparm,key in enumerate(parmlist):
  parmmap[key] = iparm
  
idxmap = []
for iidx, key in enumerate(parmlistfull):
  idxmap.append(parmmap[key])

#print(len(parmlist))
nglobal = len(parmlist)
idxmap = onp.array(idxmap)


print(idxmap)
print(nglobal)

@ROOT.Numba.Declare(["RVec<unsigned int>"], "RVec<unsigned int>")
def layer(idx):
    return idxmap[idx].astype(onp.uint32)

treename = "tree"
d = ROOT.ROOT.RDataFrame(treename,filename)

d = d.Define("dx", "dxrecgen")
#d = d.Define("dx", "dyrecgen")
#d = d.Define("dx", "dxsimgen")
#d = d.Define("dx", "dysimgen")
#d = d.Define("dx", "dxrecsim")
#d = d.Define("dx", "dyrecsim")

cut = "genPt > 5.5 && genPt < 150."

d = d.Filter(cut)

d = d.Define("hitidxr", "Numba::layer(hitidxv)")
d = d.Define("kgen", "(1./genPt)*dxsimgen/dxsimgen")

#nEtaBins = nglobal
#nkbins = 50
nkbins = 25

#nkbins = 25
#ks = onp.linspace(1./20., 1./5.5, nkbins+1, dtype=np.float64)

##nptbins = 40
##ks = 1./onp.linspace(150., 5.5, nptbins+1, dtype=np.float64)

#nptbins = 25
#pts = 1./onp.linspace(150., 20., nptbins+1, dtype=np.float64)

#ks = onp.concatenate((pts,ks[1:]),axis=0)
#nkbins = ks.shape[0]-1

#print(ks)

#assert(0)

ks = onp.linspace(1./150., 1./5.5, nkbins+1)

nkbinsfine = 1000
ksfine = onp.linspace(1./150., 1./5.5, nkbinsfine+1, dtype=np.float64)

qcs = onp.array([-1.,1.], dtype=np.float64)
kcs = 0.5*(ks[1:] + ks[:-1])

kcsfine = 0.5*(ksfine[1:] + ksfine[:-1])

nqrbins = 40000
qrlim = 0.05
#qrlim = 0.025
#qrlim = 0.005
qrs = onp.linspace(-qrlim,qrlim,nqrbins+1,dtype=np.float64)

dminus = d.Filter("genCharge<0")
dplus = d.Filter("genCharge>0")

hdxsimgenminus = dminus.Histo3D(("hdxsimgenminus", "", nglobal, -0.5, float(nglobal)+0.5, nkbins, 1./150., 1./5.5, nqrbins, -qrlim, qrlim),"hitidxr","kgen", "dx")
hdxsimgenplus = dplus.Histo3D(("hdxsimgenplus", "", nglobal, -0.5, float(nglobal)+0.5, nkbins, 1./150., 1./5.5, nqrbins, -qrlim, qrlim),"hitidxr","kgen", "dx")

#print(hdxsimgen)

dxsimgenminus = hist2array(hdxsimgenminus.GetValue())
dxsimgenplus = hist2array(hdxsimgenplus.GetValue())

dxsimgen = onp.stack([dxsimgenminus, dxsimgenplus], axis=1)

#dxsimgen = onp.reshape(dxsimgen, (nglobal, 2, 50, 10000))

print(dxsimgen.shape)

lsum = onp.sum(dxsimgen, axis=(1,2,3))

goodidxs = []
for idx in range(nglobal):
    if lsum[idx] > 10000.:
    #if lsum[idx] > 10000. and parmlist[idx][0]<2:
    #if lsum[idx] > 10000. and parmlist[idx][0]>=2:
        goodidxs.append(idx)
        
goodidxs = onp.array(goodidxs)
#goodidxs = onp.array([0])
dxsimgen = dxsimgen[goodidxs]

nEtaBins = dxsimgen.shape[0]
print(nEtaBins)
#assert(0)

nllbinnedsum = lambda *args: np.sum(nllbinned(*args),axis=(0,1,2))
gbinned = jax.grad(nllbinnedsum)

def fgbinned(*args):
    return nllbinned(*args), gbinned(*args)

gbinnedsum = lambda *args: np.sum(gbinned(*args),axis=(0,1,2))
jacbinned = jax.jacrev(gbinnedsum)
hbinned = lambda *args: np.moveaxis(jacbinned(*args),0,-1)

fgbinned = jax.jit(fgbinned)
hbinned = jax.jit(hbinned)

#fgbinned = lbatch_accumulate(fgbinned, batch_size=int(1), in_axes=(0,0,None))
#hbinned = lbatch_accumulate(hbinned, batch_size=int(1), in_axes=(0,0,None))


#xmu = np.zeros((nEtaBins,2,nkbins),dtype=np.float64)
xmu = np.zeros((nEtaBins,2,nkbins),dtype=np.float64)
xsigma = (5e-3)*np.ones((nEtaBins,2,nkbins),dtype=np.float64)
xbinned = np.stack((xmu,xsigma),axis=-1)

#val = nllbinned(xbinned, hdset, qrs)
#assert(0)

#htest = hbinned(xbinned,hdset,qrs)
#print(htest.shape)


hdset = dxsimgen


#xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=None, edmtol = 1e-3, reqposdef = False)
xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=hbinned)

hessbinned = hbinned(xbinned, hdset, qrs)
covbinned = np.linalg.inv(hessbinned)

errsbinned =  np.sqrt(np.diagonal(covbinned, offset=0, axis1=-1, axis2=-2))


nllbinnedmodelsum = lambda *args: np.sum(nllbinnedmodel(*args),axis=(0,))
gbinnedmodel = jax.grad(nllbinnedmodelsum)

def fgbinnedmodel(*args):
    return nllbinnedmodel(*args), gbinnedmodel(*args)

gbinnedmodelsum = lambda *args: np.sum(gbinnedmodel(*args),axis=(0,))
jacbinnedmodel = jax.jacrev(gbinnedmodelsum)
hbinnedmodel = lambda *args: np.moveaxis(jacbinnedmodel(*args),0,-1)

fgbinnedmodel = jax.jit(fgbinnedmodel)
hbinnedmodel = jax.jit(hbinnedmodel)

parmscale = np.zeros((nEtaBins, 4), dtype=np.float64)
parmsigma = 1e-6*np.ones((nEtaBins, 2), dtype=np.float64)

parmsmodel = np.concatenate([parmscale, parmsigma], axis=-1)

parmsmodel = pmin(fgbinnedmodel, parmsmodel, (hdset,qcs, kcs, qrs), jac=True, h=hbinnedmodel)

x = parmsmodel

hess = hbinnedmodel(parmsmodel, hdset, qcs, kcs, qrs)
cov = np.linalg.inv(hess)
xerr = np.sqrt(np.diagonal(cov, axis1=-2,axis2=-1))

scalesigmamodelfine = scalesigma(x, qcs, kcsfine)

errsmodelfines = []

for i in range(nEtaBins):
    jacmodelfine = jacscalesigma(x[i:i+1], qcs, kcsfine)
    jacmodelfine = np.swapaxes(jacmodelfine, 0, 1)
    jacmodelfineT = np.swapaxes(jacmodelfine, -1,-2)
    print(jacmodelfine.shape)
    print(jacmodelfineT.shape)
    print(cov.shape)
    covmodelfine = np.matmul(jacmodelfine,np.matmul(cov[i:i+1],jacmodelfineT))
    ierrsmodelfine = np.sqrt(np.diagonal(covmodelfine, axis1=-2, axis2=-1))
    ierrsmodelfine = np.reshape(ierrsmodelfine,scalesigmamodelfine[i:i+1].shape)
    
    errsmodelfines.append(ierrsmodelfine)

errsmodelfine = np.concatenate(errsmodelfines, axis=0)

parmlistarr = onp.array(parmlist)

subdets = parmlistarr[goodidxs][:,0]
layers = parmlistarr[goodidxs][:,1]
stereos = parmlistarr[goodidxs][:,2]

onp.savez_compressed("unbinnedfitglobalitercor.npz",
                     xbinned = xbinned,
                     errsbinned = errsbinned,
                     #hdsetks = hdsetks,
                     #etas = etas,
                     subdets = subdets,
                     layers = layers,
                     stereos = stereos,
                     ks = ks,
                     xs = x,
                     ksfine = ksfine,
                     xerrs = xerr,
                     covs = cov,
                     scalesigmamodelfine = scalesigmamodelfine,
                     errsmodelfine = errsmodelfine,
)
