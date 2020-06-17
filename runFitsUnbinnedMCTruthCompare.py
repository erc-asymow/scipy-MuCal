import os
import multiprocessing

forcecpu = True
#forcecpu = False

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
from obsminimization import pmin,batch_vmap,jacvlowmemb, batch_accumulate, lbatch_accumulate, pbatch_accumulate,lpbatch_accumulate, pbatch_accumulate_simple,random_subset
#from calInput import makeData
import argparse
import functools
import time
import sys
from header import CastToRNode


ROOT.gROOT.ProcessLine(".L src/module.cpp+")
ROOT.gROOT.ProcessLine(".L src/applyCalibration.cpp+")

ROOT.ROOT.EnableImplicitMT()


def makeData(inputFile, idx, ptmin):

    RDF = ROOT.ROOT.RDataFrame
    
    if idx==1:
        q = 1.0
    elif idx==2:
        q = -1.0
        

    d = RDF('tree',inputFile)
    
    cut = f"mcpt{idx} > {ptmin} && fabs(eta{idx})<2.4"
    #cut += f" && phi{idx}>=0. && phi{idx}<0.4"
    
    #cut += f" && eta{idx}>2.3"
    
    #print(cut)

    d = d.Filter(cut)

    #d = d.Define('krec', f'1./pt{idx}*(1.-cErr{idx}*cErr{idx})')
    #d = d.Define('kgen', f'1./mcpt{idx}')

    #d = d.Define('krec', f'1./pt{idx}')
    d = d.Define('kr', f'mcpt{idx}/pt{idx}')
    d = d.Define('kgen', f'1./mcpt{idx}')
    
    #d = d.Define('krec', f'pt{idx}')
    #d = d.Define('kgen', f'mcpt{idx}')
    
    d = d.Define('eta', f'eta{idx}')
    d = d.Define('phi', f'phi{idx}')
    d = d.Define('q', f'{q}')
    #d = d.Filter("krec/kgen>0.5 && krec/kgen<2.0")

    
    cols=['eta','phi', 'q','kgen','kr']
    
    data = d.AsNumpy(columns=cols)

    return data

def scale(A,e,M,W,Y,Z,V,k,q,eta):
    sintheta = np.sin(2*np.arctan(np.exp(-eta)))
    l = computeTrackLength(eta)
    
    #delta = A - e*sintheta*k + q*M/k + W*l**4/k**2
    #delta = A - e*sintheta*k + q*M/k + W/k**2
    #g = b/c + d
    
    #delta = A - e*sintheta*k + q*M/k - W*l**4/k**2*(1.+g*k**2/l**2)/(1.+d*k**2/l**2)
    #delta = A - e*sintheta*k + q*M/k + W*l**4/k**2
    delta = A - e*sintheta*k + q*M/k + W/k**2 + Y/k + Z*k**2 + q*V
    return 1.-delta
    #return 1.+delta

def sigmasq(a,b,c,d,k,eta):
    l = computeTrackLength(eta)
    #res1 = a1*np.power(l1,2) + c1*np.power(l1,4)*np.power(p1,2) + b1*np.power(l1,2)/(1+d1/(np.power(p1,2)*np.power(l1,2)))
    #return a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    #return a*l**2 + c*l**4/k**2*(1.+b*k**2/l**2)/(1.+d*k**2/l**2)
    
    ##res = a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    #res = a*l**2 + c*l**4/k**2*(1.+b*k**2*l**2)/(1.+d*k**2*l**2)
    
    res = a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    
    return res

def logsigpdf(krec, kgen, scale, res):

    mu = scale*kgen
    sigma = res*kgen
    
    alpha = 3.
    alpha1 = alpha
    alpha2 = alpha
    
    A1 = np.exp(0.5*alpha1**2)
    A2 = np.exp(0.5*alpha2**2)
    
    t = (krec - mu)/sigma
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
    
    Icore = (scipy.special.ndtr(alpha2) - scipy.special.ndtr(-alpha1))*sigma*np.sqrt(2.*np.pi)
    Ileft = (sigma/alpha1)*np.exp(-0.5*alpha1**2)
    Iright = (sigma/alpha2)*np.exp(-0.5*alpha2**2)
    
    I = Icore + Ileft + Iright
    
    return logpdf - np.log(I)


def logsigpdfbinned(mu,sigma,krs):
    
    krs = krs[np.newaxis,np.newaxis,np.newaxis,:]
    width = krs[...,1:] - krs[...,:-1]
    #krl = krs[...,0]
    #krh = krs[...,-1]
    
    #krl = krl[...,np.newaxis]
    #krh = krh[...,np.newaxis]
    
    kr = 0.5*(krs[...,1:] + krs[...,:-1])

    
    alpha = 3.
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

def loggauspdf(krec, kgen, scale, res):

    mu = scale*kgen
    sigma = res*kgen
    
    t = (krec - mu)/sigma
    
    pdf = -0.5*t**2
    logI = np.log(sigma) + 0.5*np.log(2.*np.pi)
    
    return pdf - logI

#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1e2])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1.])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1.])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1e2])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-3, 1e-3, 1e-7,1.])


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
    
    return -np.sum(dataset*logpdf, axis=-1)

#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1.,1e-4,1e-2])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1.])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-3, 1e-3, 1e-7,1.])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-3, 1e-3, 1e-7,1.,1e-4,1e-2])
parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-3, 1e-3, 1e-7,1.,1e-4,1e-2,1e-4])


def nll(parms,dataset,eta):
    parms = parms*parmscale
    
    A = parms[0]
    e = parms[1]
    M = parms[2]
    #W = parms[3]
    a = parms[3]
    b = parms[4]
    c = parms[5]
    d = parms[6]
    Y = parms[7]
    Z = parms[8]
    V = parms[9]
    
    W = np.zeros_like(A)
    #Y = np.zeros_like(A)
    #Z = np.zeros_like(A)
    
    #a = parms[3]
    #b = parms[4]
    #c = parms[5]
    #d = parms[6]
    
    #A = A*1e-4
    #e = e*1e-3
    #M = M*1e-5
    #W = W*1e-6
    #a = a*1e-3
    #b = b*1e-3
    #c = c*1e-7
    #d = 100.*d
    
    #W = np.zeros_like(A)
    
    #d = 370.*np.ones_like(a)
    #b = np.exp(b)
    #d = np.exp(d)
    #b = b**2
    #d = d**2
    
    #eta = dataset[...,0]
    q = dataset[...,1]
    kgen = dataset[...,2]
    kr = dataset[...,3]
    krec = kr*kgen
    
    #scalem = scale(A,e,M,W, a,b,c,d,kgen, q, eta)
    scalem = scale(A,e,M,W,Y,Z,V,kgen, q, eta)
    resm = np.sqrt(sigmasq(a,b,c,d,kgen,eta))
        
    pdf = logsigpdf(krec,kgen, scalem, resm)
    #pdf = loggauspdf(krec, kgen, scalem, resm)
    nll = -np.sum(pdf)
    
    #nll = np.where(np.any(sigma<=0.,), np.nan, nll)
    
    return nll

def nllfull(parms,dataset,ieta,etas):
    parms = parms*parmscale[np.newaxis,:parms.shape[-1]]
    
    etasc = 0.5*(etas[1:] + etas[:-1])
    
    #eta = dataset[...,0]
    kgen = dataset[...,2]
    q = dataset[...,3]
    krec = dataset[...,4]
    
    parms = parms[ieta]
    eta = etasc[ieta]
    
    A = parms[...,0]
    e = parms[...,1]
    M = parms[...,2]
    W = parms[...,3]
    a = parms[...,4]
    b = parms[...,5]
    c = parms[...,6]
    d = parms[...,7]
    #Y = parms[:,8]
    #Z = parms[:,9]
    
    Y = np.zeros_like(A)
    Z = np.zeros_like(A)
    
    scalem = scale(A,e,M,W, Y,Z, kgen, q, eta)
    resm = np.sqrt(sigmasq(a,b,c,d,kgen,eta))
        
    pdf = logsigpdf(krec,kgen, scalem, resm)
    nll = -np.sum(pdf)
    
    return nll

def nllpll(parms,dataset,ieta,etas):
    #parms = parms*parmscale[np.newaxis,:]
    parms = parms*parmscale[np.newaxis,:parms.shape[-1]]
    
    etasc = 0.5*(etas[1:] + etas[:-1])
    
    #eta = dataset[...,0]
    q = dataset[...,1]
    kgen = dataset[...,2]
    kr = dataset[...,3]
    krec = kr*kgen
    
    parms = parms[ieta]
    eta = etasc[ieta]
    
    A = parms[...,0]
    e = parms[...,1]
    M = parms[...,2]
    W = parms[...,3]
    a = parms[...,4]
    b = parms[...,5]
    c = parms[...,6]
    d = parms[...,7]
    Y = parms[...,8]
    Z = parms[...,9]
    
    #Y = np.zeros_like(A)
    #Z = np.zeros_like(A)
    
    scalem = scale(A,e,M,W, Y,Z, kgen, q, eta)
    resm = np.sqrt(sigmasq(a,b,c,d,kgen,eta))
        
    pdf = logsigpdf(krec,kgen, scalem, resm)
    
    nll = -jax.ops.segment_sum(pdf,ieta,etas.shape[0]-1)
    
    nllsum = -np.sum(pdf)
    #return -pdf
    
    return nll,nllsum
    #return nll

#def fgpll(parms,dataset,ieta,etas):
#fgi = jax.vmap(jax.value_and_grad(nllpll), in_axes=(None,0,0,None))

#def fgpll(parms,dataset,ieta,etas):
    #val,grad = fgi(parms,dataset,ieta,etas)
    #val = jax.ops.segment_sum(val,ieta,etas.shape[0]-1)
    #grad = np.sum(grad,axis=0)
    #return val, grad

   
def parmsg(parms):    
    A = parms[...,0]
    e = parms[...,1]
    M = parms[...,2]
    #W = parms[...,3]
    a = parms[...,3]
    b = parms[...,4]
    c = parms[...,5]
    d = parms[...,6]
    Y = parms[...,7]
    Z = parms[...,8]
    V = parms[...,9]
    
    g = b/c + d
    
    parms = np.stack((A,e,M,a,g,c,d,Y,Z,V),axis=-1)
    #parms = np.stack((A,e,M,a,g,c,d),axis=-1)
    #parms = np.stack((A,e,M,a,g,c,d,Y,Z),axis=-1)
    #parms = np.stack((A,e,M,W,a,g,c,d),axis=-1)
    #parms = np.stack((A,e,M,W,a,g,c,d,Y,Z),axis=-1)
    return parms   

#jacg = jax.jit(jax.vmap(jax.jacfwd(parmsg)))
jacg = jax.jit(jax.jacfwd(parmsg))

   
dataDir = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata"

files = [f"{dataDir}/muonTree.root", f"{dataDir}/muonTreeMCZ.root"]
#minpts = [3.3, 12.]
minpts = [4., 12.]
#cuts at 4 or 5, 12 here to avoid edge effects

dsets = []
for f, minpt in zip(files,minpts):
    for idx in [1,2]:
        d = makeData(f,idx,minpt)
        #dset = onp.stack( (d["eta"], d["phi"], d["kgen"],d["q"],d["krec"]), axis=-1)
        dset = onp.stack( (d["eta"],d["q"], d["kgen"],d["kr"]), axis=-1)
        print(dset.dtype)
        #dset = dset[:int(10e3)]
        dsets.append(dset)

dataset = onp.concatenate(dsets,axis=0)
dsets = None
#deta = dataset[:,0]
#dphi = dataset[:,1]
#print(dataset.shape)

#multibin fit



#nEtaBins = 48
#nEtaBins = 24
##nEtaBins = 480
#etas = onp.linspace(-2.4,2.4, nEtaBins+1, dtype=np.float64)

nEtaBins = 1
etas = np.linspace(-1.1,-1., nEtaBins+1)
#etas = np.linspace(-2.4,-2.2, nEtaBins+1)
#etas = np.linspace(-2.4,-2.0, nEtaBins+1)

nPhiBins = 1
phis = onp.linspace(-np.pi,np.pi, nPhiBins+1, dtype=np.float64)

nkbins = 40
ks = onp.linspace(1./200., 1./4.0, nkbins+1, dtype=np.float64)
#ks = 1./onp.linspace(100., 3.3, nkbins+1, dtype=np.float64)

nkbinsfine = 1000
#ksfine = 1./onp.linspace(100., 3.3, nkbinsfine+1, dtype=np.float64)
ksfine = onp.linspace(1./200., 1./4.0, nkbinsfine+1, dtype=np.float64)

qs = onp.array([-1.5,0.,1.5], dtype=np.float64)

nqrbins = 200
qrs = onp.linspace(0.7,1.3,nqrbins+1,dtype=np.float64)

qrsingle = onp.array([0.7,1.3],dtype=np.float64)

nBins = nEtaBins*nPhiBins

#dsetbinning = onp.stack((dataset[:,0], dataset[:,1], dataset[:,2], dataset[:,3]/dataset[:,2]), axis=-1)


hdset = onp.histogramdd(dataset, (etas,qs,ks,qrs))[0]

hdsetsingle = onp.histogramdd(dataset, (etas,qs,ks,qrsingle))[0]
hdsetsinglew = onp.histogramdd(dataset, (etas,qs,ks,qrsingle), weights=dataset[:,2])[0]

hdsetks = hdsetsinglew/hdsetsingle
hdsetks = onp.squeeze(hdsetks,axis=-1)


nllbinnedsum = lambda *args: np.sum(nllbinned(*args),axis=(0,1,2))
gbinned = jax.grad(nllbinnedsum)

def fgbinned(*args):
    return nllbinned(*args), gbinned(*args)

gbinnedsum = lambda *args: np.sum(gbinned(*args),axis=(0,1,2))
jacbinned = jax.jacrev(gbinnedsum)
hbinned = lambda *args: np.moveaxis(jacbinned(*args),0,-1)

fgbinned = jax.jit(fgbinned)
hbinned = jax.jit(hbinned)

xmu = np.ones((nEtaBins,2,nkbins),dtype=np.float64)
xsigma = (5e-3)*np.ones((nEtaBins,2,nkbins),dtype=np.float64)
xbinned = np.stack((xmu,xsigma),axis=-1)

#val = nllbinned(xbinned, hdset, qrs)
#assert(0)

htest = hbinned(xbinned,hdset,qrs)
print(htest.shape)



#xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=None, edmtol = 1e-3, reqposdef = False)
xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=hbinned)

hessbinned = hbinned(xbinned, hdset, qrs)
covbinned = np.linalg.inv(hessbinned)

errsbinned =  np.sqrt(np.diagonal(covbinned, offset=0, axis1=-1, axis2=-2))

#assert(0)

#dks = []
#dkerrs = []
#sigmaks = []
#for q in [1.,-1.]:
    #dsetbinning = dataset[np.where(np.equal(dataset[:,2],q))]
    #w = q*(dsetbinning[:,3]/dsetbinning[:,1]-1.)
    
    #hist =  onp.histogramdd(dsetbinning[:,:2], (etas,ks))[0]
    #histw = onp.histogramdd(dsetbinning[:,:2], (etas,ks), weights=w)[0]
    #histw2 = onp.histogramdd(dsetbinning[:,:2], (etas,ks), weights=w**2)[0]
    
    #dk = histw/hist
    #sigmak = np.sqrt(histw2/hist - dk**2)
    #dkerr = sigmak/np.sqrt(hist)
    
    #dks.append(dk)
    #dkerrs.append(dkerr)
    #sigmaks.append(sigmak)
    
    
#onp.savez_compressed("unbinnedfit.npz", (*dks, *sigmaks), ["dkplus", "dkminus","sigmakplus, sigmakminus"])

#assert(0)
    #dks[q] = dk
    #sigmaks[q] = sigmak
    



#etas = np.linspace(0.,0.1, nEtaBins+1)
etasL = etas[:-1]
etasH = etas[1:]
etasC = 0.5*(etas[1:]+etas[:-1])

phisL = phis[:-1]
phisH = phis[1:]
phisC = 0.5*(phis[1:]+phis[:-1])



etacond = onp.logical_and(dataset[:,0]>=etas[0], dataset[:,0]<etas[-1])
#phicond = onp.logical_and(dataset[:,1]>=phis[0], dataset[:,1]<phis[-1])

#phicond = onp.logical_and(dataset

#dseteta = dataset[np.where(dataset[:,0]>2.3)]
#dsetbin = dataset[onp.where(onp.logical_and(etacond,phicond))]
#dataset = dataset[onp.where(onp.logical_and(etacond,phicond))]
dataset = dataset[onp.where(etacond)]

#ieta = onp.digitize(dataset[:,0],etas)-1






fg = jax.value_and_grad(nll)
#h = jax.hessian(nll)
h = jax.jacrev(jax.grad(nll))

fg = jax.jit(fg)
h = jax.jit(h)

if forcecpu:
    #fg = pbatch_accumulate(fg, batch_size=int(1e9),ncpu=32, in_axes=(None,0,None))
    #h = pbatch_accumulate(h, batch_size=int(1e9),ncpu=32, in_axes=(None,0,None))

    fg = lpbatch_accumulate(fg, batch_size=int(1e5),ncpu=32, in_axes=(None,0,None))
    h = lpbatch_accumulate(h, batch_size=int(1e4),ncpu=32, in_axes=(None,0,None))
    
    #fg = jax.jit(fg)
    #h = jax.jit(h)
    pass
else:
    fg = lbatch_accumulate(fg, batch_size=int(1e6), in_axes=(None,0,None))
    h = lbatch_accumulate(h, batch_size=int(1e5), in_axes=(None,0,None))


#fg = lpbatch_accumulate(fg, batch_size=16384, ncpu=32, in_axes=(None,0,None))

#fg = batch_fun(fg, batch_size=10e3)
#h = batch_fun(h, batch_size=10e3)

#h = jax.hessian(nll)
#h = jax.jit(jax.hessian(nll))
#h = jax.jit(jax.jacrev(jax.grad(nll)))



def scalesigma(x, etasc, ks):
    
    A = x[...,0,np.newaxis,np.newaxis]
    e = x[...,1,np.newaxis,np.newaxis]
    M = x[...,2,np.newaxis,np.newaxis]
    #W = x[...,3,np.newaxis,np.newaxis]
    a = x[...,3,np.newaxis,np.newaxis]
    b = x[...,4,np.newaxis,np.newaxis]
    c = x[...,5,np.newaxis,np.newaxis]
    d = x[...,6,np.newaxis,np.newaxis]
    Y = x[...,7,np.newaxis,np.newaxis]
    Z = x[...,8,np.newaxis,np.newaxis]
    V = x[...,9,np.newaxis,np.newaxis]

    W = np.zeros_like(A)
    #Y = np.zeros_like(A)
    #Z = np.zeros_like(A)

    qs = np.array([-1.,1.],dtype=np.float64)
    #TODO make this more elegant and dynamic
    if len(x.shape)>1:
        qs = qs[np.newaxis,:,np.newaxis]
    else:
        qs = qs[:,np.newaxis]

    scaleout = scale(A,e,M,W,Y,Z,V,ks, qs, etasc[...,np.newaxis,np.newaxis])
    sigmaout = np.sqrt(sigmasq(a,b,c,d, ks, etasc[...,np.newaxis,np.newaxis]))
    sigmaout = sigmaout*np.ones_like(qs)

    return np.stack((scaleout,sigmaout),axis=-1)

jacscalesigma = jax.jit(jax.jacfwd(lambda *args: scalesigma(*args).flatten()))

xs = []
covs = []
xerrs = []

scalesigmamodels = []
xerrsmodels = []

scalesigmamodelsfine = []
xerrsmodelsfine = []

deta = dataset[:,0]
for ieta,(etaL, etaH, etaC) in enumerate(zip(etasL, etasH, etasC)):

    etacond = onp.logical_and(deta>=etaL, deta<etaH)
    #phicond = onp.logical_and(dphi>=phiL, dphi<phiH)
    
    #dseteta = dataset[np.where(dataset[:,0]>2.3)]
    #dsetbin = dataset[onp.where(onp.logical_and(etacond,phicond))]
    #dsetbin = dataset[onp.where(onp.logical_and(etacond,phicond))]
    dsetbin = dataset[onp.where(etacond)]
    #print("dsetbin type", type(dsetbin))
    #
    #dsetbin = dsetbin[:3605504]
    
    #dsetbin = jax.device_put(dsetbin)
    
    
    #dsetorig = onp.array(dsetbin)
    #dsetjax = np.array(dsetorig)
    
    
    #for i in range(10000):
        #print(i)
        ##res = np.reshape(dsetjax,[-1])
        ##res = dsetjax.reshape((-1,))
        ##res = dsetjax.view()
        ##res = np.expand_dims(dsetjax,-1)
        ##res = dsetjax.ravel()
        ##res = dsetjax[:,:-2]
        ##dsetjax.shape=(-1,)
        #res = dsetjax.view()
        #res = np.reshape(res,(-1,))
        #res.block_until_ready()
        ##dsetjax.shape = (-1,)
    
    #assert(0)
    
    
    print("dsetbin type", type(dsetbin))
    #etaC = jax.device_put(etaC)
    #dseteta = dseteta.astype(np.float64)
    #ieta = onp.digitize(dseteta[:,0],etas)-1
    
    print("dsetbin shape", dsetbin.shape)
    
    print(etaC)
    
    A = 0.
    e = 0.
    M = 0.
    W = 0.
    #a = 1e-3
    #b = 1e-3
    #c = 1e-5
    #d = 370.
    a = 1.
    b = 1.
    c = 1.
    d = 100.
    Y = 0.
    Z = 0.
    V = 0.
    

    x = np.stack((A,e,M,a,b,c,d,Y,Z,V)).astype(np.float64)
    #x = np.stack((A,e,M,a,b,c,d)).astype(np.float64)
    #x = np.stack((A,e,M,a,b,c,d,Y,Z)).astype(np.float64)
    #x = np.stack((A,e,M,W,a,b,c,d)).astype(np.float64)
    #x = np.stack((A,e,M,W,a,b,c,d,Y,Z)).astype(np.float64)
    
    #x = np.stack((A,e,M,a,b,c,d)).astype(np.float64)
    #lb = np.array([-1e-1, -1e-1, -1., -1., 1e-9, 0., 0., 0.])
    #ub = np.array([1e-1, 1e-1, 1., 1., 1., np.inf, np.inf, np.inf])

    #lb = np.array([-100., -100., -100., -100., 1e-2, 0., 0., 10.])
    #ub = np.array([100., 100., 100., 100., np.inf, np.inf, np.inf, 1000.])
    
    lb = np.array( 7*[-np.inf] + [0.01] ,dtype=np.float64)
    ub = np.array( 7*[np.inf] + [100.] ,dtype=np.float64)

    #lb = np.array( 8*[-np.inf] ,dtype=np.float64)
    #ub = np.array( 8*[np.inf] ,dtype=np.float64)    
    
    
    #with jax.disable_jit():
        #val,grad = fg(x,dsetbin,etaC)
    #assert(0)
    
    #x = pmin(fg, x, (dseteta,etaC), doParallel=False, jac=True, h=None, lb=lb,ub=ub)
    x = pmin(fg, x, (dsetbin,etaC), doParallel=False, jac=True, h=None,xtol=1e-4, edmtol=1.,reqposdef=False)
    x = pmin(fg, x, (dsetbin,etaC), doParallel=False, jac=True, h=h,edmtol=0.01)
    
    print("computing hess")
    hess = h(x, dsetbin, etaC)
    #hess = np.eye(x.shape[0])
    cov = np.linalg.inv(hess)
    #xerr = np.sqrt(np.diag(cov))
    

    x = x*parmscale
    cov = cov*parmscale[:,np.newaxis]*parmscale[np.newaxis,:]
    
    
    iscalesigmamodel = scalesigma(x,etaC, hdsetks[ieta])
    jacmodel = jacscalesigma(x,etaC,hdsetks[ieta])
    covmodel = np.matmul(jacmodel,np.matmul(cov,jacmodel.T))
    ierrsmodel = np.sqrt(np.diag(covmodel))
    ierrsmodel = np.reshape(ierrsmodel,iscalesigmamodel.shape)
    
    scalesigmamodels.append(iscalesigmamodel)
    xerrsmodels.append(ierrsmodel)
    
    iscalesigmamodelfine = scalesigma(x,etaC, ksfine[np.newaxis,:])
    jacmodelfine = jacscalesigma(x,etaC,ksfine[np.newaxis,:])
    covmodelfine = np.matmul(jacmodelfine,np.matmul(cov,jacmodelfine.T))
    ierrsmodelfine = np.sqrt(np.diag(covmodelfine))
    ierrsmodelfine = np.reshape(ierrsmodelfine,iscalesigmamodelfine.shape)
    
    scalesigmamodelsfine.append(iscalesigmamodelfine)
    xerrsmodelsfine.append(ierrsmodelfine)
    
    #convert from abcd to agcd
    jac = jacg(x)
    x = parmsg(x)
    cov = np.matmul(jac,np.matmul(cov,jac.T))
    
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
scalesigmamodel = np.stack(scalesigmamodels, axis=0)
errsmodel = np.stack(xerrsmodels, axis=0)
scalesigmamodelfine = np.stack(scalesigmamodelsfine, axis=0)
errsmodelfine = np.stack(xerrsmodelsfine, axis=0)
    
nparms = xs.shape[-1]
cov = onp.zeros(shape=(nBins*nparms,nBins*nparms), dtype=xs.dtype)
print("nparms", nparms)
print("xs.shape", xs.shape)
print("cov.shape", cov.shape)
for i, icov in enumerate(covs):
    cov[i*nparms:(i+1)*nparms, i*nparms:(i+1)*nparms] = icov
    
print(cov)

covdiag = np.diag(cov)
cor = cov/np.sqrt(covdiag[:,np.newaxis]*covdiag[np.newaxis,:])
print(cor)


#dkfines = []
#sigmakfines = []
#for q in [1.,-1.]:
    


#jacscalesigmapre = jax.jacfwd(lambda *args: np.sum(scalesigma(*args), axis=0).flatten())
#jacscalesigma = lambda *args: np.moveaxis(jacscalesigmapre(*args),0,-2)
#jacscalesigma = jax.jit(jacscalesigma)


#onp.savez_compressed("unbinnedfit.npz", (*dks, *dkerrs, *sigmaks, *dkfines, *sigmakfines,etas,ks,ksfine, xs, xerrs, covs), ["dkplus", "dkminus","dkerrplus", "dkerrminus","sigmakplus, sigmakminus", "dkfineplus", "dkfineminus", "sigmakfineplus", "sigmakfineminus","etas","ks","ksfine", "xs","xerrs","covs"])

#onp.savez_compressed("unbinnedfit.npz",
                     #dkplus = dks[0],
                     #dkminus = dks[1],
                     #dkerrplus = dkerrs[0],
                     #dkerrminus = dkerrs[1],
                     #sigmakplus = sigmaks[0],
                     #sigmakminus = sigmaks[1],
                     #dkfineplus = dkfines[0],
                     #dkfineminus = dkfines[1],
                     #sigmakfineplus = sigmakfines[0],
                     #sigmakfineminus = sigmakfines[1],
                     #etas = etas,
                     #ks = ks,
                     #ksfine = ksfine,
                     #xs = xs,
                     #xerrs = xerrs,
                     #covs = covs)
                     
onp.savez_compressed("unbinnedfit.npz",
                     xbinned = xbinned,
                     errsbinned = errsbinned,
                     hdsetks = hdsetks,
                     scalesigmamodel = scalesigmamodel,
                     scalesigmamodelfine = scalesigmamodelfine,
                     errsmodel = errsmodel,
                     errsmodelfine = errsmodelfine,
                     etas = etas,
                     ks = ks,
                     ksfine = ksfine,
                     xs = xs,
                     xerrs = xerrs,
                     covs = covs)
                     

nparms = xs.shape[-1]
cov = onp.zeros(shape=(nBins*nparms,nBins*nparms), dtype=xs.dtype)
print("nparms", nparms)
print("xs.shape", xs.shape)
print("cov.shape", cov.shape)
for i, icov in enumerate(covs):
    cov[i*nparms:(i+1)*nparms, i*nparms:(i+1)*nparms] = icov
    
print(cov)




covdiag = np.diag(cov)
cor = cov/np.sqrt(covdiag[:,np.newaxis]*covdiag[np.newaxis,:])
print(cor)

A = xs[:,0]
e = xs[:,1]
M = xs[:,2]
#W = xs[:,3]
a = xs[:,3]
b = xs[:,4]
c = xs[:,5]
d = xs[:,6]
Y = xs[:,7]
Z = xs[:,8]

W = np.zeros_like(A)
#Y = np.zeros_like(A)
#Z = np.zeros_like(A)

#a = xs[:,3]
#b = xs[:,4]
#c = xs[:,5]
#d = xs[:,6]

Aerr = xerrs[:,0]
eerr = xerrs[:,1]
Merr = xerrs[:,2]
#Werr = xerrs[:,3]
aerr = xerrs[:,3]
berr = xerrs[:,4]
cerr = xerrs[:,5]
derr = xerrs[:,6]
Yerr = xerrs[:,7]
Zerr = xerrs[:,8]
Werr = np.zeros_like(A)
#Yerr = np.zeros_like(A)
#Zerr = np.zeros_like(A)

#aerr = xerrs[:,3]
#berr = xerrs[:,4]
#cerr = xerrs[:,5]
#derr = xerrs[:,6]


if nPhiBins>1:
    binarr = onp.linspace(-0.5,nBins-0.5,nBins+1)
else:
    binarr = onp.array(etas.tolist())

hA = ROOT.TH1D("A", "A", nBins, binarr)
he = ROOT.TH1D("e", "e", nBins, binarr)
hM = ROOT.TH1D("M", "M", nBins, binarr)
hW = ROOT.TH1D("W", "W", nBins, binarr)
ha = ROOT.TH1D("a", "a", nBins, binarr)
hc = ROOT.TH1D("c", "c", nBins, binarr)
hb = ROOT.TH1D("g", "g", nBins, binarr)
hd = ROOT.TH1D("d", "d", nBins, binarr)
hY = ROOT.TH1D("Y", "Y", nBins, binarr)
hZ = ROOT.TH1D("Z", "Z", nBins, binarr)

hA = array2hist(A, hA, Aerr)
he = array2hist(e, he, eerr)
hM = array2hist(M, hM, Merr)
hW = array2hist(W, hW, Werr)
ha = array2hist(a, ha, aerr)
hc = array2hist(c, hc, cerr)
hb = array2hist(b, hb, berr)
hd = array2hist(d, hd, derr)
hY = array2hist(Y, hY, Yerr)
hZ = array2hist(Z, hZ, Zerr)

hA.GetYaxis().SetTitle('b field correction')
he.GetYaxis().SetTitle('material correction')
hM.GetYaxis().SetTitle('alignment correction')
hW.GetYaxis().SetTitle('charge-independent "alignment" correction')
ha.GetYaxis().SetTitle('material correction (resolution) a^2')
hc.GetYaxis().SetTitle('hit position (resolution) c^2')

hA.GetXaxis().SetTitle('#eta')
he.GetXaxis().SetTitle('#eta')
hM.GetXaxis().SetTitle('#eta')
hW.GetXaxis().SetTitle('#eta')
ha.GetXaxis().SetTitle('#eta')
hc.GetXaxis().SetTitle('#eta')

hCov = ROOT.TH2D("cov","cov", nBins*nparms, -0.5, nBins*nparms-0.5, nBins*nparms, -0.5, nBins*nparms-0.5)
hCov = array2hist(cov, hCov)

hCor = ROOT.TH2D("cor","cor", nBins*nparms, -0.5, nBins*nparms-0.5, nBins*nparms, -0.5, nBins*nparms-0.5)
hCor = array2hist(cor, hCor)


f = ROOT.TFile("calibrationMC.root", 'recreate')
f.cd()

hA.Write()
he.Write()
hM.Write()
hW.Write()
ha.Write()
hc.Write()
hb.Write()
hd.Write()
hY.Write()
hZ.Write()
hCov.Write()
hCor.Write()
