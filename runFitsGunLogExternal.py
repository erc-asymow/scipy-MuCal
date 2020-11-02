import os
import multiprocessing

#forcecpu = True
forcecpu = False

if forcecpu:
    ncpu = multiprocessing.cpu_count()
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ncpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

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


def makeData(inputFile,ptmin):

    RDF = ROOT.ROOT.RDataFrame

    #ptmin = 5.5
    #ptmax = 
    
    #if idx==1:
        #q = 1.0
    #elif idx==2:
        #q = -1.0
        

    d = RDF('tree',inputFile)
    
    cut = f"1./gen_curv>{ptmin} && fabs(gen_eta)<2.4 && reco_charge>-2."
    
    #cut = f"mcpt{idx} > {ptmin} && fabs(eta{idx})<2.4"
    #cut += f" && gen_phi>=0. && gen_phi<0.4"
    #cut += f" && gen_phi>={-np.pi} && gen_phi<{np.pi/4.}"
    
    #cut += f" && eta{idx}>2.3"
    
    #print(cut)

    d = d.Filter(cut)

    #d = d.Define('krec', f'1./pt{idx}*(1.-cErr{idx}*cErr{idx})')
    #d = d.Define('kgen', f'1./mcpt{idx}')

    #d = d.Define('krec', f'1./pt{idx}')
    #d = d.Define('kr', "log(reco_curv/gen_curv)")
    d = d.Define('kr', "reco_charge*reco_curv/gen_curv/gen_charge")
    #d = d.Define('kr', "gen_curv/reco_curv")
    d = d.Define('kgen', "gen_curv")
    
    d = d.Filter("kr>=0. && kr<2.")
    #d = d.Filter("kr>=(-log(3.)) && kr<log(3.)")
    
    #d = d.Define('krec', f'pt{idx}')
    #d = d.Define('kgen', f'mcpt{idx}')
    
    d = d.Define('eta', "gen_eta")
    d = d.Define('phi', "gen_phi")
    d = d.Define('q', "gen_charge")
    #d = d.Filter("krec/kgen>0.5 && krec/kgen<2.0")

    
    cols=['eta','phi', 'q','kgen','kr']
    
    data = d.AsNumpy(columns=cols)

    return data


def makeDataDiMuon(inputFile, idx, ptmin):

    RDF = ROOT.ROOT.RDataFrame
    
    if idx==1:
        q = 1.0
    elif idx==2:
        q = -1.0
        

    d = RDF('tree',inputFile)
    
    cut = f"mcpt{idx} > {ptmin} && fabs(mceta{idx})<2.4"
    #cut += f" && phi{idx}>=0. && phi{idx}<0.4"
    #cut += f" && phi{idx}>={-np.pi} && phi{idx}<{np.pi/4.}"
    
    #cut += f" && eta{idx}>2.3"
    
    #print(cut)

    d = d.Filter(cut)

    #d = d.Define('krec', f'1./pt{idx}*(1.-cErr{idx}*cErr{idx})')
    #d = d.Define('kgen', f'1./mcpt{idx}')

    #d = d.Define('krec', f'1./pt{idx}')
    d = d.Define('kr', f'mcpt{idx}/pt{idx}')
    #d = d.Define('kr', f'pt{idx}/mcpt{idx}')
    
    #d = d.Define('kr', f'log(mcpt{idx}/pt{idx})')
    d = d.Define('kgen', f'1./mcpt{idx}')
    
    d = d.Filter("kr>=0. && kr<2.")
    
    #d = d.Filter("kr>=-log(3.) && kr<log(3.)")
    
    #d = d.Define('krec', f'pt{idx}')
    #d = d.Define('kgen', f'mcpt{idx}')
    
    d = d.Define('eta', f'mceta{idx}')
    d = d.Define('phi', f'mcphi{idx}')
    d = d.Define('q', f'{q}')
    #d = d.Filter("krec/kgen>0.5 && krec/kgen<2.0")

    
    cols=['eta','phi', 'q','kgen','kr']
    
    data = d.AsNumpy(columns=cols)

    return data




def logsigpdf(kr, scale, res):

    mu = scale
    sigma = res
    
    #sigma = 1e-4
    
    alpha = 3.0
    alpha1 = alpha
    alpha2 = alpha
    
    #A1 = np.exp(0.5*alpha1**2)
    #A2 = np.exp(0.5*alpha2**2)
    
    logA1 = 0.5*alpha1**2
    logA2 = 0.5*alpha2**2
    
    A1 = np.exp(logA1)
    A2 = np.exp(logA2)
    
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
    
    #thigh = (np.log(3.)-mu)/sigma
    #tlow = (-np.log(3.)-mu)/sigma
    
    thigh = (2.-mu)/sigma
    tlow = (0.-mu)/sigma
    
    
    #Icore = (scipy.special.ndtr(alpha2) - scipy.special.ndtr(-alpha1))*sigma*np.sqrt(2.*np.pi)
    #Icore = 0.5*(scipy.special.erf(alpha2/np.sqrt(2.)) - scipy.special.erf(-alpha1/np.sqrt(2.)))*sigma*np.sqrt(2.*np.pi)
    #Ileft = (sigma/alpha1)*A1*(np.exp(-alpha1**2) - np.exp(alpha1*tlow))
    #Iright = (sigma/alpha2)*A2*(np.exp(-alpha2*thigh) - np.exp(-alpha2**2))
    
    #Ileft = (sigma/alpha1)*np.exp(-0.5*alpha1**2)
    #Iright = (sigma/alpha2)*np.exp(-0.5*alpha2**2)
    
    
    t1 = np.clip(-alpha1, tlow, thigh)
    t2 = np.clip(alpha2, tlow, thigh)
    Icore = 0.5*(scipy.special.erf(t2/np.sqrt(2.)) - scipy.special.erf(t1/np.sqrt(2.)))*sigma*np.sqrt(2.*np.pi)
    Ileft = (sigma/alpha1)*A1*(np.exp(alpha1*t1) - np.exp(alpha1*tlow))
    Iright = (sigma/alpha2)*A2*(np.exp(-alpha2*thigh) - np.exp(-alpha2*t2))

    
    I = Icore + Ileft + Iright
    
    #I = np.where(np.logical_and(tlow<-alpha1,thigh>alpha2),I,np.nan)
    
    return logpdf - np.log(I)

def loggauspdfbinned(mu,sigma,krs):
    krs = krs[np.newaxis,np.newaxis,np.newaxis,:]
    width = krs[...,1:] - krs[...,:-1]
    #krl = krs[...,0]
    #krh = krs[...,-1]
    
    kr = 0.5*(krs[...,1:] + krs[...,:-1])

    t = (kr - mu)/sigma
    
    logpdf = -0.5*t**2
    
    I = np.sum(width*np.exp(logpdf),axis=-1,keepdims=True)
    
    return logpdf - np.log(I)
    
    

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

def loggauspdf(kr, scale, res):

    mu = scale
    sigma = res
    
    t = (kr - mu)/sigma
    
    pdf = -0.5*t**2 - np.log(sigma) - 0.5*np.log(2.*np.pi)
    
    #thigh = (np.log(3.)-mu)/sigma
    #tlow = (-np.log(3.)-mu)/sigma
    
    thigh = (2.-mu)/sigma
    tlow = (0.-mu)/sigma
    
    #I = scipy.special.ndtr(thigh) - scipy.special.ndtr(tlow)
    I = 0.5*(scipy.special.erf(thigh/np.sqrt(2.)) - scipy.special.erf(tlow/np.sqrt(2.)))
    logI = np.log(I)
    
    #logI = np.log(sigma) + 0.5*np.log(2.*np.pi)
    
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
    #logpdf = loggauspdfbinned(mu,sigma,krs)
    
    nll = -np.sum(dataset*logpdf, axis=-1)
    #nll += np.squeeze(sigma,axis=-1)**2
    #nll += sigma**2
    
    return nll

#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1.,1e-4,1e-2])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1.])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-3, 1e-3, 1e-7,1.])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-3, 1e-3, 1e-7,1.,1e-4,1e-2])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-3, 1e-3, 1e-7,1.,1e-4,1e-2,1e-4])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-3, 1e-3, 1e-7,1.,1e-4,1e-2,1e-4,1e-6,1e-3,1e-4,1e-2])
#parmscale = np.array([1e-4, 1e-3, 1e-5,1e-6,1e-4,1e-2,1e-3, 1e-3, 1e-7,1.])
#parmscale = np.array([1e-4, 1e-3, 1e-5,1e-6,1e-4,1e-2,1e-3, 1e-3, 1e-7,1.,1e-6,1e-3,1e-4,1e-2])
#parmscale = np.array([1e-4, 1e-3, 1e-5,1e-6,1e-4,1e-2,1e-3, 1e-3, 1e-7,1.,1e-3,1e-2])
#parmscale = np.array([1e-4, 1e-3, 1e-5,1e-6,1e-4,1e-2,1e-3, 1e-3, 1e-7,1.])

#parmscale = np.array([1e-4, 1e-3, 1e-5,1e-3, 1e-3, 1e-7,1.,1e-6,1e-4,1e-5])


#parmscale = np.array([1e-4, 1e-3, 1e-5,1e-3, 1e-3, 1e-7,1.,1e-6,1e-4,1e-5,1e-4,1e-4,1e-4])
#parmscale = np.array([1e-4, 1e-3, 1e-5,1e-3, 1e-3, 1e-7,1.,1e-2,1.,1.,1.,1.,1.])

parmscale = np.array([1e-4, 1e-3, 1e-5,1e-3, 1e-3, 1e-7,1.,1.,1.,1.,1.,1.,1.])
#parmscale = np.array([1e-4,1., 1e-5,1e-3, 1e-3, 1e-7,1.,1.,1.,1.,1.,1.,1.])

#parmscale = np.array([1.,1.,1.,1e-3, 1e-3, 1e-7,1.,1.,1.,1.,1.,1.,1.])

#parmscale = 1.



def mvalsprior(parms,ds,ks,qs):
    #scaledparms = parms*parmscale
    
    #A0,e0,M0,A3,e3,M3,a,b2,b3,c = parms
    A0,e0,M0,A1,e1,M1,A2,e2,M2,A3,e3,M3,a,b1,b2,b3,c = parms
    d1,d2,d3 = ds

    k = ks
    q = qs

    cor1 = 1./(1.+d1*k**2)
    cor2 = 1./(1.+d2*k**2)
    cor3 = 1./(1.+d3*k**2)

    #scale = 1. + A0 + q*M0/k - e0*k + A1*cor1 + q*M1*k*cor1 - e1*k*cor1 + A2*cor2 + q*M2*k*cor2 - e2*k*cor2 + A3*cor3 + q*M3*k*cor3 -
    #scale = 1. + A0 + q*M0/k - e0*k + A1*cor1 + q*M1*k*cor1 + A2*cor2 + q*M2*k*cor2 + A3*cor3 + q*M3*k*cor3
    #scale = 1. + A0 + q*M0/k - e0*k + A3*cor3 + q*M3*k*cor3
    #sigmasq = a**2 + c**2/k**2 + b1*cor1 + b2*cor2 + b3*cor3 + 0.*q
    #sigmasq = a**2 + c**2/k**2 + b2*cor2 + 0.*q
    
    scale = 1.
    scale += A0 + q*M0/k -e0*k
    scale += (A3 + q*M3/k - e3*k)*cor3
    scale += (A2 + q*M2/k - e2*k)*cor2
    scale += (A1 + q*M1/k - e1*k)*cor1
    
    sigmasq = a**2 + c**2/k**2 + 0.*q
    sigmasq += b2*cor2
    sigmasq += b3*cor3
    sigmasq += b1*cor1

    res = np.stack((scale,sigmasq),axis=-1)
    
    prior = 0.
    prior += 0.5*A0**2
    prior += 0.5*A1**2
    prior += 0.5*A2**2
    prior += 0.5*A3**2
    prior += 0.5*M0**2
    prior += 0.5*M1**2
    prior += 0.5*M2**2
    prior += 0.5*M3**2
    prior += 0.5*e0**2
    prior += 0.5*e1**2
    prior += 0.5*e2**2
    prior += 0.5*e3**2
    prior += 0.5*b1**2
    prior += 0.5*b2**2
    prior += 0.5*b3**2
    
    return res,prior
    
def mvals(parms,ds,ks,qs):
    return mvalsprior(parms,ds,ks,qs)[0]
    
    

def nll(parms,ds,dataset,eta,ntotal):
      
    #eta = dataset[...,0]
    q = dataset[...,1]
    kgen = dataset[...,2]
    kr = dataset[...,3]
    #krec = kr*kgen
    
    res,prior = mvalsprior(parms,ds,kgen,q)
    
    scalem = res[...,0]
    resmsq = res[...,1]
    
    resm = np.sqrt(resmsq)
        
    #pdf = logsigpdf(krec,kgen, scalem, resm)
    #pdf = loggauspdf(krec, kgen, scalem, resm)
    pdf = logsigpdf(kr, scalem, resm)
    #pdf = loggauspdf(kr, scalem, resm)
    nll = -np.sum(pdf)
    #nll += Z**2
    #nll += b**2
    #nll += 1000.*(a**2 + b**2 + c**2 + d**2)
    #nll += 1e6*(M**2 + Y**2 + V**2 + e2**2 + Z2**2)
    ##nll += 1e6*(Y**2 + V**2 + e2**2 + Z2**2)
    cweight = dataset.shape[0]/ntotal
    
    print("dataset.shape[0], ntotal, cweight",dataset.shape[0], ntotal, cweight)
    
    nll += cweight*prior
    
    return nll


   
dataDir = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata"

#files = [f"{dataDir}/muonTree.root", f"{dataDir}/muonTreeMCZ.root"]
#minpts = [3.3, 12.]
#minpts = [4., 12.]
#cuts at 4 or 5, 12 here to avoid edge effects

#dsets = []
#for f, minpt in zip(files,minpts):
    #for idx in [1,2]:
        #d = makeData(f,idx,minpt)
        ##dset = onp.stack( (d["eta"], d["phi"], d["kgen"],d["q"],d["krec"]), axis=-1)
        #dset = onp.stack( (d["eta"],d["q"], d["kgen"],d["kr"]), axis=-1)
        #print(dset.dtype)
        ##dset = dset[:int(10e3)]
        #dsets.append(dset)

#dataset = onp.concatenate(dsets,axis=0)

f = f"{dataDir}/muonGuntree.root"
#fdj = f"{dataDir}/muonTree.root"
fdj = f"{dataDir}/JpsiToMuMu_JpsiPt8_pythia8.root"
#fdz = f"{dataDir}/muonTreeMCZ.root"
fdz = f"{dataDir}/ZJToMuMu_mWPilot.root"
ftrk = f"{dataDir}/trackTreeP.root"

dsets = []

#d = makeDataTrk(ftrk,33.)
#dset = onp.stack( (d["etagen"],d["qgen"], d["kgen"],d["kr"]), axis=-1)
#dsets.append(dset)
#dset = None

d = makeData(f,5.5)
dset = onp.stack( (d["eta"],d["q"], d["kgen"],d["kr"]), axis=-1)
d = None
dsets.append(dset)
dset = None

#for fdi,minpt in zip([fdj,fdz], [5.5,12.]):
for fdi,minpt in zip([fdj], [5.5]):
    for idx in [1,2]:
        #d = makeDataDiMuon(fdi,idx,5.5)
        d = makeDataDiMuon(fdi,idx,minpt)
        dset = onp.stack( (d["eta"],d["q"], d["kgen"],d["kr"]), axis=-1)
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
#nEtaBins = 24
####nEtaBins = 480
etas = onp.linspace(-2.4,2.4, nEtaBins+1, dtype=np.float64)

#nEtaBins = 1
#etas = onp.linspace(-2.4,-2.3, nEtaBins+1)
#etas = np.linspace(-2.3,-2.2, nEtaBins+1)
#etas = np.linspace(-2.2,-2.1, nEtaBins+1)
#etas = np.linspace(-2.3,-2.1, nEtaBins+1)
#etas = np.linspace(1.5,1.6, nEtaBins+1)
#etas = np.linspace(-1.1,-0.9, nEtaBins+1)
#etas = onp.linspace(-1.1,-1., nEtaBins+1)
#etas = np.linspace(-1.02,-1., nEtaBins+1)
#etas = np.linspace(-2.4,-2.2, nEtaBins+1)
#etas = np.linspace(-2.4,-2.0, nEtaBins+1)

nPhiBins = 1
phis = onp.linspace(-np.pi,np.pi, nPhiBins+1, dtype=np.float64)

nkbins = 25
ks = onp.linspace(1./20., 1./5.5, nkbins+1, dtype=np.float64)

#nptbins = 40
#ks = 1./onp.linspace(150., 5.5, nptbins+1, dtype=np.float64)

nptbins = 25
pts = 1./onp.linspace(150., 20., nptbins+1, dtype=np.float64)

ks = onp.concatenate((pts,ks[1:]),axis=0)

#nkbins = 40
#ks = 1./onp.linspace(150.,33.,nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./5.5, nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./12., nkbins+1, dtype=np.float64)



nkbins = ks.shape[0]-1


nkbinsfine = 1000
#ksfine = 1./onp.linspace(100., 3.3, nkbinsfine+1, dtype=np.float64)
ksfine = onp.linspace(1./150., 1./5.5, nkbinsfine+1, dtype=np.float64)

qs = onp.array([-1.5,0.,1.5], dtype=np.float64)
#qs = onp.array([0.,1.5], dtype=np.float64)

nqrbins = 10000
qrs = onp.linspace(0.,2.,nqrbins+1,dtype=np.float64)
qrsingle = onp.array([0.,2.],dtype=np.float64)
#qrs = onp.linspace(-np.log(3.),np.log(3.),nqrbins+1,dtype=np.float64)
#qrsingle = onp.array([-np.log(3.),np.log(3.)],dtype=np.float64)

nBins = nEtaBins*nPhiBins

#dsetbinning = onp.stack((dataset[:,0], dataset[:,1], dataset[:,2], dataset[:,3]/dataset[:,2]), axis=-1)

etacond = onp.logical_and(dataset[:,0]>=etas[0], dataset[:,0]<etas[-1])
dataset = dataset[onp.where(etacond)]



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

#fgbinned = lbatch_accumulate(fgbinned, batch_size=int(1), in_axes=(0,0,None))
#hbinned = lbatch_accumulate(hbinned, batch_size=int(1), in_axes=(0,0,None))


#xmu = np.zeros((nEtaBins,2,nkbins),dtype=np.float64)
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



fname = "compresmixedtriplecheckpoint/unbinnedfiterrsmedfulldiag.npz"
with np.load(fname) as f:
    xs = f["xs"]
    xerrs = f["xerrs"]

didxs = slice(0,3)

ds = xs[:,didxs]
derrs = xerrs[:,didxs]

#derrs = 0.5/np.sqrt(ds)*derrs
#ds = np.sqrt(ds)

sortidxs = onp.argsort(ds, axis=-1)

ds = onp.take_along_axis(ds,sortidxs,axis=-1)
derrs = onp.take_along_axis(derrs,sortidxs,axis=-1)

ds = ds[:nEtaBins]
derrs = derrs[:nEtaBins]

ds = jax.device_put(ds)
derrs = jax.device_put(derrs)

#phicond = onp.logical_and(dataset[:,1]>=phis[0], dataset[:,1]<phis[-1])

#phicond = onp.logical_and(dataset

#dseteta = dataset[np.where(dataset[:,0]>2.3)]
#dsetbin = dataset[onp.where(onp.logical_and(etacond,phicond))]
#dataset = dataset[onp.where(onp.logical_and(etacond,phicond))]


#ieta = onp.digitize(dataset[:,0],etas)-1






fg = jax.value_and_grad(nll)
h = jax.hessian(nll)
#h = jax.jacrev(jax.grad(nll))

fg = jax.jit(fg)
h = jax.jit(h)

if forcecpu:
    #fg = pbatch_accumulate(fg, batch_size=int(1e9),ncpu=32, in_axes=(None,0,None))
    #h = pbatch_accumulate(h, batch_size=int(1e9),ncpu=32, in_axes=(None,0,None))

    fg = lpbatch_accumulate(fg, batch_size=int(1e6),ncpu=32, in_axes=(None,None,0,None,None))
    h = lpbatch_accumulate(h, batch_size=int(1e5),ncpu=32, in_axes=(None,None,0,None,None))
    
    #fg = jax.jit(fg)
    #h = jax.jit(h)
    pass
else:
    fg = lbatch_accumulate(fg, batch_size=int(1e6), in_axes=(None,None,0,None,None))
    h = lbatch_accumulate(h, batch_size=int(1e5), in_axes=(None,None,0,None,None))


#fg = lpbatch_accumulate(fg, batch_size=16384, ncpu=32, in_axes=(None,0,None))

#fg = batch_fun(fg, batch_size=10e3)
#h = batch_fun(h, batch_size=10e3)

#h = jax.hessian(nll)
#h = jax.jit(jax.hessian(nll))
#h = jax.jit(jax.jacrev(jax.grad(nll)))




jacmvals = jax.jit(jax.jacfwd(lambda *args: mvals(*args).flatten()))

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
    ntotal = dsetbin.shape[0]
    
    dsfit = ds[ieta]
     
    print("dsetbin type", type(dsetbin))
    #etaC = jax.device_put(etaC)
    #dseteta = dseteta.astype(np.float64)
    #ieta = onp.digitize(dseteta[:,0],etas)-1
    
    print("dsetbin shape", dsetbin.shape)
    
    print(etaC)
    
    #A0,e0,M0,A1,e1,M1,a,b1,c = parms

    
    A0 = 0.
    e0 = 0.
    M0 = 0.
    A1 = 0.
    e1 = 0.
    M1 = 0.
    A2 = 0.
    e2 = 0.
    M2 = 0.
    A3 = 0.
    e3 = 0.
    M3 = 0.
    a = 1e-4
    b1 = 0.
    b2 = 0.
    b3 = 0.
    c = 1e-3

    #x = np.stack((A0,e0,M0,A3,e3,M3,a,b2,b3,c)).astype(np.float64)
    x = np.stack((A0,e0,M0,A1,e1,M1,A2,e2,M2,A3,e3,M3,a,b1,b2,b3,c)).astype(np.float64)

    
    x = pmin(fg, x, (dsfit,dsetbin,etaC,ntotal), doParallel=False, jac=True, h=None,xtol=1e-14,edmtol=1e-3)
    x = pmin(fg, x, (dsfit,dsetbin,etaC,ntotal), doParallel=False, jac=True, h=h)
    
    print("computing hess")
    hess = h(x, dsfit, dsetbin, etaC, ntotal)
    #hess = np.eye(x.shape[0])
    cov = np.linalg.inv(hess)
    #xerr = np.sqrt(np.diag(cov))
    

    #x = x*parmscale
    #cov = cov*parmscale[:,np.newaxis]*parmscale[np.newaxis,:]
    
    
    #iscalesigmamodel = scalesigma(x,etaC, hdsetks[ieta])
    #jacmodel = jacscalesigma(x,etaC,hdsetks[ieta])
    #covmodel = np.matmul(jacmodel,np.matmul(cov,jacmodel.T))
    #ierrsmodel = np.sqrt(np.diag(covmodel))
    #ierrsmodel = np.reshape(ierrsmodel,iscalesigmamodel.shape)
    
    #scalesigmamodels.append(iscalesigmamodel)
    #xerrsmodels.append(ierrsmodel)
    
    #iscalesigmamodelfine = scalesigma(x,etaC, ksfine[np.newaxis,:])
    #jacmodelfine = jacscalesigma(x,etaC,ksfine[np.newaxis,:])
    #covmodelfine = np.matmul(jacmodelfine,np.matmul(cov,jacmodelfine.T))
    #ierrsmodelfine = np.sqrt(np.diag(covmodelfine))
    #ierrsmodelfine = np.reshape(ierrsmodelfine,iscalesigmamodelfine.shape)
    

    ksfinescale = ksfine[np.newaxis,:]*np.ones((2,1),dtype=np.float64)
    
    qs = np.array([-1.,1.],dtype=np.float64)
    qs = qs[:,np.newaxis]

    iscalesigmamodelfine = mvals(x,dsfit,ksfinescale,qs)
    jacmodelfine = jacmvals(x,dsfit,ksfinescale,qs)
    #jacmodelfine = np.reshape(jacmodelfine,(-1,nparms))
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
    
    scalesigmamodelsfine.append(iscalesigmamodelfine)
    xerrsmodelsfine.append(ierrsmodelfine)
    
    #convert from abcd to agcd
    #jac = jacg(x)
    #x = parmsg(x)
    #cov = np.matmul(jac,np.matmul(cov,jac.T))
    
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
#scalesigmamodel = np.stack(scalesigmamodels, axis=0)
#errsmodel = np.stack(xerrsmodels, axis=0)
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

