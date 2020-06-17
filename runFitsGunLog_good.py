import os
import multiprocessing

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



def makeDataTrk(inputFile,ptmin):

    RDF = ROOT.ROOT.RDataFrame

    #ptmin = 5.5
    #ptmax = 
    
    #if idx==1:
        #q = 1.0
    #elif idx==2:
        #q = -1.0
        

    d = RDF('tree',inputFile)
    
    cut = f"genPt>{ptmin} && fabs(genEta)<2.4"
    
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
    d = d.Define('kr', "genPt/trackPt")
    #d = d.Define('kr', "gen_curv/reco_curv")
    d = d.Define('kgen', "1./genPt")
    
    d = d.Filter("kr>=0. && kr<2.")
    #d = d.Filter("kr>=(-log(3.)) && kr<log(3.)")
    
    #d = d.Define('krec', f'pt{idx}')
    #d = d.Define('kgen', f'mcpt{idx}')
    
    d = d.Define('etagen', "genEta")
    d = d.Define('phigen', "genPhi")
    d = d.Define('qgen', "genCharge")
    #d = d.Filter("krec/kgen>0.5 && krec/kgen<2.0")

    
    cols=['etagen','phigen', 'qgen','kgen','kr']
    
    data = d.AsNumpy(columns=cols)

    return data


def scale(A,e,M,W,Y,Z,V,e2,Z2,k,q,eta):
    sintheta = np.sin(2*np.arctan(np.exp(-eta)))
    l = computeTrackLength(eta)
    
    #delta = A - e*sintheta*k + q*M/k + W*l**4/k**2
    #delta = A - e*sintheta*k + q*M/k + W/k**2
    #g = b/c + d
    
    #delta = A - e*sintheta*k + q*M/k - W*l**4/k**2*(1.+g*k**2/l**2)/(1.+d*k**2/l**2)
    #delta = A - e*sintheta*k + q*M/k + W*l**4/k**2
    #delta = A - e*sintheta*k + q*M/k + W/k**2 + Y/k + Z*k**2 + q*V + q*e2*k + q*W2/k**2 + q*Z2*k**2
    #delta = A - e*k + q*M/k + W/k**2 + Y/k + Z*k**2 + q*V + q*e2*k + q*W2/k**2 + q*Z2*k**2
    #delta = A - e*k + q*M/k +Z*k**2 + Y/k + q*W*k
    #delta = A - e*k + q*M/k 
    #return 1.-delta
    #return np.log1p(delta) - 0.5*sigmasq
    
    #res = np.log1p(A) - np.log1p(e*k) + np.log1p(q*M/k) -0.5*W/k**2 + Y*k**2 -0.5*Z/k
    
    #res = np.log1p(A) - np.log1p(e*k) + np.log1p((1.+e*k)*q*M/k) -0.5*W/k**2 + Y*k**2 -0.5*Z/k
    #res = np.log1p(A) - np.log1p(e*k) + np.log1p((1.+e*k)*q*M/k) -0.5*W/k**2 + Y*k**2 -0.5*Z/k + V*k
    #res = np.log1p(A) + np.log1p((1.+e*k)*q*M/k) -0.5*W/k**2 + Y*k**2 -0.5*Z/k + V*k
    #res = A - np.log1p(e*k) + np.log1p(q*M/k + q*e*M) -0.5*W/k**2 + Y*k**2 -0.5*Z/k + V*k
    #res = A + np.log1p(q*M/k + q*e) -0.5*W/k**2 + Y*k**2 -0.5*Z/k + V*k
    
    #res =  A + q*M/k + e*k + W/k**2 + Y*k**2 + Z/k + q*V + q*e2*k + q*Z2*k**2
    
    #res =  1. -A - q*M/k - e*k - W**2/(1.+Y**2/k**2)
    #res =  -A - q*M/k - e*k + (1.+W**2*Y**2/k**2)/(1.+Y**2/k**2)
    #res = 1 - A - q*M/k - e*k - W/(1.+ Y*k**2)
    #res = 1. - A - q*M/k - e*k - W/(1.+ Y**2*k**2)
    #res = (1.+A)*(1. + W*k**2)/(1. + Y*k**2) - q*M/k - e*k
    #res = 1.  - A - q*M/k - e*k + W**2/(1.+Y**2*k**2)
    #res = (1. + A + W**2*Y**2/k**2)/(1. + Y**2/k**2) - q*M/k
    #res = (1. + A + W**2*Y**2*k**2)/(1. + Y**2*k**2) - q*M/k
    #res = (1. + A + np.exp(W+Y)/k**2)/(1. + np.exp(Y)/k**2) - q*M/k
    #res = 1. - A - q*M/k - W/(1.+Y**2*k**2)
    #res = 1. - A - q*M/k - W/(1.+Y**2*k**2)
    #res = (1. + A + (1.+W)*Y**2/k**2)/(1.+Y**2/k**2) - q*M/k
    #res = (1.+A)*(1.+np.exp(W+Y)/k**2)/(1.+np.exp(Y)/k**2)*(1.+q*M/k)/(1.+e*k)
    #res = (1.+A)*(1.+np.exp(W+Y)/k**2)/(1.+np.exp(Y)/k**2)*(1.+q*M/k)/(1.+e*k)
    
    
    #good
    #res = (1.+A)*(1.+np.exp(W+Y)/k**2)/(1.+np.exp(Y)/k**2)*(1.+q*M/k)
    
    #good2
    #res = (1.+A)*(1.+np.exp(W+Y)*k**2)/(1.+np.exp(Y)*k**2)*(1.+q*M/k)
    
    #good3
    #res = (1.+A)*(1.+np.exp(W+Y)/k**2)/(1.+np.exp(Y)/k**2)*(1.+q*M/k)/(1.+e*k)
    
    #bad
    #res = (1.+A)*(1.+np.exp(W+Y)*k**2)/(1.+np.exp(Y)*k**2)*(1.+q*M/k)/(1.+e*k)
    
    #res = (1.+A)*(1.+np.exp(W+Y)*k**2)/(1.+np.exp(Y)*k**2)*(1. +q*M/k + q*V + q*e2*k +  q*Z2*k**2)/(1.+e*k)

    
    #emod = 5e-2*np.sin(e)
    #emod = 1e-3*e
    #emod = 0.1*np.tanh(e)
    
    Y = np.sqrt(1.+Y**2) - 1.
    
    #res = (1.+A)*(1.+np.exp(W+Y)*k**2)/(1.+np.exp(Y)*k**2)*(1. +q*M/k)/(1. + e*k)
    #res = (1. + A + np.exp(W+Y)*k**2)/(1.+np.exp(Y)*k**2)*(1. +q*M/k)/(1. + e*k)
    res = (1. + A + (1.+W)*Y*k**2)/(1.+Y*k**2)*(1. +q*M/k)/(1. + e*k) 
    
    #res = np.where(np.abs(e)>0.1, np.nan, res)
    
    #res = np.where(Y>=0., res, np.nan)
    return res
    #res =  A + q*M/k + e*k + W/k**3 + Y*k**2 + Z/k + q*V + q*e2*k + q*Z2*k**2
    #res =  A + q*M/k + e*k + Z/k
    #res =  A + q*M/k + e*k + Z/k + W/k**3
    
    #res =  A + q*M/k + e*k  + W/k**2 + Z/k
    
    
    #res =  A + q*M/k + e*np.sin(k) + W/k**2 + Y*np.cos(k) + Z/k + q*V + q*e2*k + q*Z2*k**2
    #res =  A + q*M/k + e*k + W*np.sin(k) + Y*k**2 + Z*np.cos(k) + q*V + q*e2*k + q*Z2*k**2
    
    
    
    #res =  A + q*M/k + e*k + W/k**2 + Y*k**2 + Z/k + q*e2*k - 4.*e2*q*k**2
    
    #res = np.log1p(A) - np.log1p(e*k) + np.log1p(q*M/k) -0.5*W/k**2 + Y*k**2
    #res = np.log1p(A) - np.log1p(e*k) + np.log1p(q*M/k) + Y*k**2
    #res = A - e*k +  q*M/k -0.5*W/k**2 + Y*k**2 -0.5*Z/k
    #res = A - e*k +  q*M/k -0.5*W/k**2 + Y*k**2
    
    #res = - np.log1p(e*k) + np.log1p(q*M/k) -0.5*W/k**2 + Y*k**2 -0.5*Z/k + q*M*W/k**3/3. + q*M*Z/k**2/3. - 0.5*A + q*M*A/k/3.
    
    #return 1.-res
    #return 1.+res
    #return 1.+delta

def sigmasq(a,b,c,d,k,eta):
    l = computeTrackLength(eta)
    #res1 = a1*np.power(l1,2) + c1*np.power(l1,4)*np.power(p1,2) + b1*np.power(l1,2)/(1+d1/(np.power(p1,2)*np.power(l1,2)))
    #return a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    #return a*l**2 + c*l**4/k**2*(1.+b*k**2/l**2)/(1.+d*k**2/l**2)
    
    ##res = a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    #res = a*l**2 + c*l**4/k**2*(1.+b*k**2*l**2)/(1.+d*k**2*l**2)
    
    res = a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    #res = a**2 + c**2/k**2 + b/k + d**2*k**2
    #res = a**2 + c**2/k**2 + b**2/k + d**2*k**2
    
    return res

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



def nll(parms,dataset,eta,ntotal):
    parms = parms*parmscale
    
    A = parms[0]
    e = parms[1]
    M = parms[2]
    a = parms[3]
    b = parms[4]
    c = parms[5]
    d = parms[6]
    W = parms[7]
    Y = parms[8]
    Z = parms[9]
    V = parms[10]
    e2 = parms[11]
    Z2 = parms[12]
    
    #e2 = parms[10]
    #Z2 = parms[11]
    
    #V = parms[10]
    #e2 = parms[11]
    ##W2 = parms[12]
    #Z2 = parms[13]

    #V = np.zeros_like(A)
    #W2 = np.zeros_like(A)

    a = a**2
    b = b**2
    c = c**2
    d = d**2
    
    
    #a = np.where(a>0.,a,np.nan)
    #b = np.where(b>0.,b,np.nan)
    #c = np.where(c>0.,c,np.nan)
    #d = np.where(d>0.,d,np.nan)
    
    #W = np.zeros_like(A)
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
    #krec = kr*kgen
    
    #scalem = scale(A,e,M,W,Y,Z,kgen, q, eta)
    resmsq = sigmasq(a,b,c,d,kgen,eta)
    resm = np.sqrt(resmsq)
    scalem = scale(A,e,M,W,Y,Z,V,e2,Z2,kgen, q, eta)
        
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
    
    #nll += 0.5*cweight*(Z**2 +  e2**2 + Z2**2 + V**2 + e**2)
    #nll += 0.5*cweight*(Z**2 +  e2**2 + Z2**2 + V**2 + e**2)
    nll += 0.5*cweight*(Z**2 +  e2**2 + Z2**2 + V**2)
    
    nll += 0.5*cweight/10.**2*W**2
    nll += 0.5*cweight/1e6**2*Y**2
    nll += 0.5*cweight/0.01**2*e**2
    
    #nll += 0.5*cweight/1e-3**2*e**2
    #nll += 0.5*cweight*(Z**2)
    
    #nll += 0.5*cweight/1000.**2*(A**2 + W**2)
    #nll += 0.5*cweight/100.**2*W**2
    #nll += 0.5*cweight/100.**2*Y**2
    
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
    #Y = parms[...,4]
    #Z = parms[...,5]
    a = parms[...,3]
    b = parms[...,4]
    c = parms[...,5]
    d = parms[...,6]
    W = parms[...,7]
    Y = parms[...,8]
    Z = parms[...,9]
    #e2 = parms[...,10]
    #Z2 = parms[...,11]
    
    
    a = a**2
    b = b**2
    c = c**2
    d = d**2
    
    g = b/c + d
    
    parms = np.stack((A,e,M,a,g,c,d,W,Y,Z),axis=-1)
    
    #parms = np.stack((A,e,M,W,Y,Z,a,g,c,d,e2,Z2),axis=-1)
    #parms = np.stack((A,e,M,W,Y,Z,a,g,c,d),axis=-1)
    #parms = np.stack((A,e,M,W,Y,Z,a,g,c,d),axis=-1)
    #parms = np.stack((A,e,M,a,g,c,d,Y,Z,V),axis=-1)
    #parms = np.stack((A,e,M,a,g,c,d,Y,Z,V,W,e2,W2,Z2),axis=-1)
    #parms = np.stack((A,e,M,a,g,c,d),axis=-1)
    #parms = np.stack((A,e,M,a,g,c,d,Y,Z),axis=-1)
    #parms = np.stack((A,e,M,W,a,g,c,d),axis=-1)
    #parms = np.stack((A,e,M,W,a,g,c,d,Y,Z),axis=-1)
    return parms   

#jacg = jax.jit(jax.vmap(jax.jacfwd(parmsg)))
jacg = jax.jit(jax.jacfwd(parmsg))

   
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
fdj = f"{dataDir}/JpsiToMuMu_JpsiPt8_pythia8_2.root"
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
#etas = np.linspace(-2.4,-2.3, nEtaBins+1)
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

    fg = lpbatch_accumulate(fg, batch_size=int(1e6),ncpu=32, in_axes=(None,0,None,None))
    h = lpbatch_accumulate(h, batch_size=int(1e5),ncpu=32, in_axes=(None,0,None,None))
    
    #fg = jax.jit(fg)
    #h = jax.jit(h)
    pass
else:
    fg = lbatch_accumulate(fg, batch_size=int(1e6), in_axes=(None,0,None,None))
    h = lbatch_accumulate(h, batch_size=int(1e5), in_axes=(None,0,None,None))


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
    #Y = x[...,4,np.newaxis,np.newaxis]
    #Z = x[...,5,np.newaxis,np.newaxis]
    a = x[...,3,np.newaxis,np.newaxis]
    b = x[...,4,np.newaxis,np.newaxis]
    c = x[...,5,np.newaxis,np.newaxis]
    d = x[...,6,np.newaxis,np.newaxis]
    W = x[...,7,np.newaxis,np.newaxis]
    Y = x[...,8,np.newaxis,np.newaxis]
    Z = x[...,9,np.newaxis,np.newaxis]
    V = x[...,10,np.newaxis,np.newaxis]
    e2 = x[...,11,np.newaxis,np.newaxis]
    Z2 = x[...,12,np.newaxis,np.newaxis]
    #e2 = x[...,10,np.newaxis,np.newaxis]
    #Z2 = x[...,11,np.newaxis,np.newaxis]

    #V = np.zeros_like(A)
    #W2 = np.zeros_like(A)
    #e2 = np.zeros_like(A)
    #Z2 = np.zeros_like(A)

    a = a**2
    b = b**2
    c = c**2
    d = d**2

    qs = np.array([-1.,1.],dtype=np.float64)
    #TODO make this more elegant and dynamic
    if len(x.shape)>1:
        qs = qs[np.newaxis,:,np.newaxis]
    else:
        qs = qs[:,np.newaxis]

    #scaleout = scale(A,e,M,W,Y,Z,ks, qs, etasc[...,np.newaxis,np.newaxis])
    sigmasqout = sigmasq(a,b,c,d, ks, etasc[...,np.newaxis,np.newaxis])
    sigmaout = np.sqrt(sigmasqout)
    scaleout = scale(A,e,M,W,Y,Z,V,e2,Z2,ks, qs, etasc[...,np.newaxis,np.newaxis])
    
    
    sigmaout = sigmaout*np.ones_like(qs)
    
    print("scaleout.shape, sigmaout.shape", scaleout.shape, sigmaout.shape)


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
    ntotal = dsetbin.shape[0]
    
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
    d = 10.
    Y = 0.
    Z = 0.
    V = 0.
    e2 = 0.
    W2 = 0.
    Z2 = 0.
    
    #W = 1.
    Y = 1000.

    x = np.stack((A,e,M,a,b,c,d,W,Y,Z,V,e2,Z2)).astype(np.float64)
    #x = np.stack((A,e,M,W,Y,Z,a,b,c,d)).astype(np.float64)
    #x = np.stack((A,e,M,W,Y,Z,a,b,c,d,e2,Z2)).astype(np.float64)
    #x = np.stack((A,e,M,W,Y,Z,a,b,c,d)).astype(np.float64)
    #x = np.stack((A,e,M,a,b,c,d,Y,Z,V,W,e2,W2,Z2)).astype(np.float64)
    #x = np.stack((A,e,M,a,b,c,d,Y,Z,V)).astype(np.float64)
    #x = np.stack((A,e,M,a,b,c,d,Y,Z,V,W)).astype(np.float64)
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
    #x = pmin(fg, x, (dsetbin,etaC), doParallel=False, jac=True, h=None,xtol=1e-4, edmtol=1.,reqposdef=False)
    #x = pmin(fg, x, (dsetbin,etaC), doParallel=False, jac=True, h=h,edmtol=0.001)
    
    
    
    x = pmin(fg, x, (dsetbin,etaC,ntotal), doParallel=False, jac=True, h=None,xtol=1e-14)
    x = pmin(fg, x, (dsetbin,etaC,ntotal), doParallel=False, jac=True, h=h, edmtol = 1e-4)
    
    print("computing hess")
    hess = h(x, dsetbin, etaC, ntotal)
    #hess = np.eye(x.shape[0])
    cov = np.linalg.inv(hess)
    #xerr = np.sqrt(np.diag(cov))
    

    x = x*parmscale
    cov = cov*parmscale[:,np.newaxis]*parmscale[np.newaxis,:]
    
    
    #iscalesigmamodel = scalesigma(x,etaC, hdsetks[ieta])
    #jacmodel = jacscalesigma(x,etaC,hdsetks[ieta])
    #covmodel = np.matmul(jacmodel,np.matmul(cov,jacmodel.T))
    #ierrsmodel = np.sqrt(np.diag(covmodel))
    #ierrsmodel = np.reshape(ierrsmodel,iscalesigmamodel.shape)
    
    #scalesigmamodels.append(iscalesigmamodel)
    #xerrsmodels.append(ierrsmodel)
    
    iscalesigmamodelfine = scalesigma(x,etaC, ksfine[np.newaxis,:])
    jacmodelfine = jacscalesigma(x,etaC,ksfine[np.newaxis,:])
    covmodelfine = np.matmul(jacmodelfine,np.matmul(cov,jacmodelfine.T))
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

