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
#matplotlib.use('agg')
#matplotlib.use('TkAgg')
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
#ROOT.TTreeProcessorMT.SetMaxTasksPerFilePerWorker(1);


#filename = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprec/globalcor_*.root"
#filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprec/globalcor_0.root"

#filename = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprecsimplecpe//globalcor_*.root"
#filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprecsimplecpe//globalcor_0.root"

#filename = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantint4fullprecisegen//globalcor_*.root"
#filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantint4fullprecisegen//globalcor_0.root"

#filename = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v1_Rec/201116_001722/0000/globalcor_*.root"

#filename = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v27_Rec/201205_145455/0000/globalcor_*.root"

#filename = "correctedTracks2016.root"
#filename = "correctedTracks.root"
#filename = "results_v27_aligdigi_01p67/correctedTracks.root"
#filename = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v28_Rec/201214_151021/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v8_Rec/201124_020803/0000/globalcor_*.root"

#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recideal/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealnopxe/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealnopxb/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealrkprec/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealpixelsimx/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealpixelsimxy/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealsimxy/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealsimxygenstart/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealsimxygenstartnopixely/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealnopixelyconstraint/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealpixelsimy/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealgenstartpixelgeny/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealgeantprec/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealidealfield/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealidealfieldanalyticprop/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealidealfieldanalyticpropnopxb/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealpxbsimxy/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealpxbgenxygenstart/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recidealdebugnotemplate/globalcor_*.root"
#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/recnonidealdebugnotemplate/globalcor_*.root"

#filename = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v3_Rec_idealquality/210212_165035/0000/globalcor_*.root"
#filename = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v3_Rec_idealnoquality/210212_164913/0000/globalcor_*.root"
#filename = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v3_Rec_noquality/210212_165222/0000/globalcor_*.root"
#filename = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v3_Rec_quality/210212_165132/0000/globalcor_*.root"

#filename = "/data/shared/muoncal/resultsqualitycompare/MuonGUNUL2016Fwd_v4_Rec_quality/210212_220352/0000/globalcor_*.root"

#filename = "/data/shared/muoncal/MuonGUNUL2016Fwd_v33_Rec_idealquality/210228_002451/0000/globalcor_*.root"

#filename = "/data/shared/muoncal/MuonGunUL2016_v33_Rec_quality/210228_002026/0000/globalcor_*.root"

#filename = "/data/shared/muoncal/MuonGunUL2016_v41_Rec_noquality/210405_115619/0000/globalcor_*.root"

#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v36plus2_Rec_noquality/210405_185722/0000/globalcor_*.root"


#filename = "/data/shared/muoncal/MuonGunUL2016_v62_RecJpsiPhotos_quality_constraint/210506_144841/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/JPsiToMuMuGun_Pt5To30-pythia8-photos/MuonGunUL2016_v62_RecJpsiPhotos_quality_noconstraint/210506_225452/0000/globalcor_*.root"
#filename = "/data/shared/muoncal/MuonGunUL2016_v96_RecJpsiPhotosTail0_quality_noconstraint/210714_101530/0000/globalcor_*.root"

chain = ROOT.TChain("tree");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v109_RecJpsiPhotos_quality_constraint/210720_225832/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v109_RecJpsiPhotos_quality_constraint/210720_225832/0001/globalcor_*.root");


chain.Add("/data/shared/muoncal/MuonGunUL2016_v155_RecJpsiPhotos_idealquality_constraint/210822_142043/0000/globalcor_*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v155_RecJpsiPhotos_idealquality_constraint/210822_142043/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v117_RecJpsiPhotos_quality_constraint/210723_144009/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v117_RecJpsiPhotos_quality_constraint/210723_144009/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v122_RecJpsiPhotos_quality_constraint/210726_013107/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v122_RecJpsiPhotos_quality_constraint/210726_013107/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_RecJpsiPhotos_idealquality_zeromaterial_noconstraint/210728_081911/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_RecJpsiPhotos_idealquality_zeromaterial/210728_080349/0000/globalcor_*.root")

#filename = "/data/shared/muoncal/MuonGunUL2016_v45_RecJpsiPhotos_quality_constraint_v2/210411_195122/0000/globalcor_*.root"
#filename = "/data/shared/muoncal/MuonGunUL2016_v48_RecJpsiPhotos_quality_constraint/210414_072114/0000/globalcor_*.root"

#filename = "results_V33_01p567idealquality/correctedTracks.root"
#filename = "correctedTracks.root"







#filename = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v1_Gen/201116_000843/0000/globalcor_*.root"
#filenameinfo = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v1_Gen/201116_000843/0000/globalcor_0_1.root"

#finfo = ROOT.TFile.Open(filenameinfo)
#runtree = finfo.runtree

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

def scale(A,e,M,k,q):
    return q*A*k + M - q*e*k**2
    #return q*k*A + e*k**2 + M + q*R

def sigmasq(a, c, k):
    #return a + c/k**2
    return a*k**2 + c
    #return c + a*k**2

def scalesigma(parms, qs, ks):
    A = parms[..., 0, np.newaxis, np.newaxis]
    e = parms[..., 1, np.newaxis, np.newaxis]
    M = parms[..., 2, np.newaxis, np.newaxis]
    a = parms[..., 3, np.newaxis, np.newaxis]
    c = parms[..., 4, np.newaxis, np.newaxis]
    
    qs = qs[np.newaxis, :, np.newaxis]
    ks = ks[np.newaxis, np.newaxis, :]
    
    scaleout = scale(A,e,M,ks,qs)
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
    
    
ROOT.gInterpreter.Declare("""
float deltaphi(float phi0, float phi1) {
    float dphi = phi1 - phi0;
    if (dphi > M_PI) {
        dphi -= 2.0*M_PI;
    }
    else if (dphi <= - M_PI) {
      dphi += 2.0*M_PI;
    }
    return dphi;

}
    
""")
    

treename = "tree"
#d = ROOT.ROOT.RDataFrame(treename,filename)
d = ROOT.ROOT.RDataFrame(chain)

#d = d.Define("dx", "dxrecgen")
#d = d.Define("dx", "dyrecgen")
#d = d.Define("dx", "dxsimgen")
#d = d.Define("dx", "dysimgen")
#d = d.Define("dx", "dxrecsim")
#d = d.Define("dx", "dyrecsim")

ptmin = 3.5

#cut = "genPt > 5.5 && genPt < 150. && nValidHits > 0"
#d = d.Filter("Muplusgen_pt > 5.5 && Muminusgen_pt > 5.5 && Jpsigen_mass > 3.0968")
d = d.Filter(f"Muplusgen_pt > {ptmin} && Muminusgen_pt > {ptmin}")

#d = d.Filter("Jpsikin_mass > 2.8 && Jpsikin_mass < 3.4")

d = d.Filter("Jpsigen_mass > 3.0968")
#d = d.Filter("Muplus_nvalid > 2 && Muplus_nvalidpixel<2 && Muminus_nvalid > 2 && Muminus_nvalidpixel < 2");

#d = d.Define("recParms", "trackParms");
#d = d.Define("recParms", "refParms");
#d = d.Define("recParms", "corParms");

#d = d.Define("recPt", "std::abs(1./recParms[0])*std::sin(M_PI_2 - recParms[1])");
#d = d.Define("recCharge", "std::copysign(1.0,recParms[0])");

#d = d.Define("kr", "genPt*genCharge/recPt/recCharge");
#d = d.Define("krplus", "Muplusgen_pt/Muplus_pt");
#d = d.Define("krminus", "Muminusgen_pt/Muminus_pt");

#d = d.Define("krplus", "(M_PI_2 - 2.0*atan(exp(-Muplus_eta))) - (M_PI_2 - 2.0*atan(exp(-Muplusgen_eta)))");
#d = d.Define("krminus", "(M_PI_2 - 2.0*atan(exp(-Muminus_eta))) - (M_PI_2 - 2.0*atan(exp(-Muminusgen_eta)))");

#d = d.Define("krplus", "deltaphi(Muplusgen_phi, Muplus_phi)");
#d = d.Define("krminus", "deltaphi(Muminusgen_phi, Muminus_phi)");

d = d.Define("krplus", "deltaphi(Muplusgen_phi, Mupluscons_phi)");
d = d.Define("krminus", "deltaphi(Muminusgen_phi, Muminuscons_phi)");


d = d.Define('kgenplus', "1./Muplusgen_pt")
d = d.Define('kgenminus', "1./Muminusgen_pt")

qrmin = -0.01
qrmax = 0.01

#qrmin = 0.8
#qrmax = 1.2

#d = d.Filter(f"kr>={qrmin} && kr<{qrmax}")

#d = d.Filter("hitidxv[0]==9")



#d = d.Filter("kr>=(-log(3.)) && kr<log(3.)")

#d = d.Define('krec', f'pt{idx}')
#d = d.Define('kgen', f'mcpt{idx}')

d = d.Define('etaplus', "Muplusgen_eta")
d = d.Define('etaminus', "Muminusgen_eta")
#d = d.Define('phi', "genPhi")
#d = d.Define('q', "genCharge")

#d = d.Define("hitidxr", "Numba::layer(hitidxv)")
#d = d.Define("kgen", "(1./genPt)*dxsimgen/dxsimgen")

#nEtaBins = nglobal
#nkbins = 50
#nkbins = 25

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

#ks = onp.linspace(1./150., 1./5.5, nkbins+1)

nkbins = 25
ks = onp.linspace(1./20., 1./ptmin, nkbins+1, dtype=np.float64)

#nptbins = 40
#ks = 1./onp.linspace(150., 5.5, nptbins+1, dtype=np.float64)

ptmax = 30.

nptbins = 25
pts = 1./onp.linspace(ptmax, 20., nptbins+1, dtype=np.float64)

#ks = onp.concatenate((pts,ks[1:]),axis=0)
ks = ks

#nkbins = 40
#ks = 1./onp.linspace(150.,33.,nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./5.5, nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./12., nkbins+1, dtype=np.float64)



nkbins = ks.shape[0]-1

nkbinsfine = 1000
ksfine = onp.linspace(1./ptmax, 1./ptmin, nkbinsfine+1, dtype=np.float64)

qcs = onp.array([-1.,1.], dtype=np.float64)
kcs = 0.5*(ks[1:] + ks[:-1])

kcsfine = 0.5*(ksfine[1:] + ksfine[:-1])

#nqrbins = 20000
nqrbins = 2000
#nqrbins = 60000

#qrlim = 0.05
#qrlim = 0.025
#qrlim = 0.005
qrs = onp.linspace(qrmin,qrmax,nqrbins+1,dtype=np.float64)



#nEtaBins = 1
#etamin = -2.4
#etamax = -2.3

nEtaBins = 48
etamin = -2.4
etamax = 2.4
#nEtaBins = 24
####nEtaBins = 480
etas = onp.linspace(etamin, etamax, nEtaBins+1, dtype=np.float64)

#dminus = d.Filter("genCharge<0")
#dplus = d.Filter("genCharge>0")

dplus = d.Filter(f"krplus>={qrmin} && krplus<{qrmax}")
dminus = d.Filter(f"krminus>={qrmin} && krminus<{qrmax}")

hdxsimgenminus = dminus.Histo3D(("hdxsimgenminus", "", nEtaBins, etas, nkbins, ks, nqrbins, qrs),"etaminus","kgenminus", "krminus")
hdxsimgenplus = dplus.Histo3D(("hdxsimgenplus", "", nEtaBins, etas, nkbins, ks, nqrbins, qrs),"etaplus","kgenplus", "krplus")

#print(hdxsimgenminus.GetSumOfWeights())
#print(hdxsimgenplus.GetSumOfWeights())


#cminus = ROOT.TCanvas()
#hdxsimgenminus.Project3D("xy").Draw("COLZ")

#cplus = ROOT.TCanvas()
#hdxsimgenplus.Project3D("xy").Draw("COLZ")

#input("wait")

#hdxsimgenminus = dminus.Histo3D(("hdxsimgenminus", "", nEtaBins, etamin, etamax, nkbins, 1./150., 1./5.5, nqrbins, qrmin, qrmax),"eta","kgen", "kr")
#hdxsimgenplus = dplus.Histo3D(("hdxsimgenplus", "", nEtaBins, etamin, etamax, nkbins, 1./150., 1./5.5, nqrbins, qrmin, qrmax),"eta","kgen", "kr")

#print(hdxsimgen)

print("starting rdf loop")

dxsimgenminus = hist2array(hdxsimgenminus.GetValue())
dxsimgenplus = hist2array(hdxsimgenplus.GetValue())

print("done converting hists")

dxsimgen = onp.stack([dxsimgenminus, dxsimgenplus], axis=1)

#dxsimgen = onp.reshape(dxsimgen, (nglobal, 2, 50, 10000))

#print(dxsimgen.shape)

#lsum = onp.sum(dxsimgen, axis=(1,2,3))

#goodidxs = []
#for idx in range(nglobal):
    #if lsum[idx] > 10000.:
    ##if lsum[idx] > 10000. and parmlist[idx][0]<2:
    ##if lsum[idx] > 10000. and parmlist[idx][0]>=2:
        #goodidxs.append(idx)
        
#goodidxs = onp.array(goodidxs)
##goodidxs = onp.array([0])
#dxsimgen = dxsimgen[goodidxs]

#nEtaBins = dxsimgen.shape[0]
#print(nEtaBins)
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

#print(hdset.shape)
#assert(0)


#xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=None, edmtol = 1e-3, reqposdef = False)
xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=hbinned, edmtol = 1e-5)

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

#parmscale = np.zeros((nEtaBins, 3), dtype=np.float64)
#parmsigma = 1e-5*np.ones((nEtaBins, 2), dtype=np.float64)

#parmsmodel = np.concatenate([parmscale, parmsigma], axis=-1)

parmscale = np.zeros((nEtaBins, 3), dtype=np.float64)
parmsigma0 = 1e-4*np.ones((nEtaBins, 1), dtype=np.float64)
parmsigma1 = 1e-8*np.ones((nEtaBins, 1), dtype=np.float64)

parmsmodel = np.concatenate([parmscale, parmsigma0, parmsigma1], axis=-1)

#parmsmodel = pmin(fgbinnedmodel, parmsmodel, (hdset,qcs, kcs, qrs), jac=True, h=None, edmtol = 1e-3, reqposdef = False)
parmsmodel = pmin(fgbinnedmodel, parmsmodel, (hdset,qcs, kcs, qrs), jac=True, h=hbinnedmodel, edmtol = 1e-5)

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

#parmlistarr = onp.array(parmlist)

#subdets = parmlistarr[goodidxs][:,0]
#layers = parmlistarr[goodidxs][:,1]
#stereos = parmlistarr[goodidxs][:,2]





onp.savez_compressed("unbinnedfitglobalitercorscale.npz",
                     xbinned = xbinned,
                     errsbinned = errsbinned,
                     #hdsetks = hdsetks,
                     etas = etas,
                     #subdets = subdets,
                     #layers = layers,
                     #stereos = stereos,
                     ks = ks,
                     xs = x,
                     ksfine = ksfine,
                     xerrs = xerr,
                     covs = cov,
                     scalesigmamodelfine = scalesigmamodelfine,
                     errsmodelfine = errsmodelfine,
)


if (False):
    rebin = 200

    qrcs = 0.5*(qrs[:-1] + qrs[1:])
    qrcscoarse = np.sum(qrcs.reshape(-1,rebin), axis=-1)/float(rebin)

    xerr = 0.5*(qrs[1:]-qrs[:-1])*float(rebin)
    xerr = xerr[0]

    matplotlib.use('TkAgg')

    #shapetgt = hdset.shape[:3] + (-1,200)

    hdsetcoarse = np.sum(hdset.reshape(hdset.shape[:-1] + (-1,rebin)),axis=-1)

    pdfvals = np.exp(logsigpdfbinned(xbinned[...,0,np.newaxis], xbinned[...,1,np.newaxis], qrs))


    print("debug")
    print(hdset.shape)
    print(hdsetcoarse.shape)
    print(qrs.shape)
    print(qrcs.shape)
    #print(shapetgt)
    histdata = hdsetcoarse[0,0,0]

    pdfnorm = np.sum(histdata)
    histpdf = pdfnorm*pdfvals[0,0,0]*2.*xerr

    plt.figure()
    plt.errorbar(qrcscoarse,histdata, xerr=xerr, yerr=np.sqrt(histdata), fmt='none' )
    plt.plot(qrcs, histpdf)
    plt.show()
