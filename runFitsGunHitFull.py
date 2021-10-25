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
from utils import lumitools

from header import CastToRNode


ROOT.gInterpreter.ProcessLine(".O3")
ROOT.ROOT.EnableImplicitMT()
#ROOT.TTreeProcessorMT.SetMaxTasksPerFilePerWorker(1);


lumitools.init_lumitools()
jsonhelper = lumitools.make_jsonhelper("data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")


#filename = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprec/globalcor_*.root"
#filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprec/globalcor_0.root"

#filename = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprecsimplecpe//globalcor_*.root"
#filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantintgenhelixprecsimplecpe//globalcor_0.root"

#filename = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantint4fullprecisegen//globalcor_*.root"
#filenameinfo = "/data/bendavid/cmsswdev/muonscale/CMSSW_10_6_17_patch1/work/resultsgeantint4fullprecisegen//globalcor_0.root"

#filename = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v1_Gen/201116_000843/0000/globalcor_*.root"
#filenameinfo = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v1_Gen/201116_000843/0000/globalcor_0_1.root"

#filename = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v29_Gen/201214_201633/0000/globalcor_*.root"
#filenameinfo = "root://eoscms.cern.ch//store/group/phys_smp/bendavid/DoubleMuonGun_Pt3To150/MuonGunUL2016_v29_Gen/201214_201633/0000/globalcor_0_1.root"


#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgrads/globalcor_*.root"
#filenameinfo = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgrads/globalcor_0.root"

#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/gennonidealgradsdebugqbins/globalcor_*.root"
#filenameinfo = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/gennonidealgradsdebugqbins/globalcor_0.root"

#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgradsdebugnotemplate/globalcor_*.root"
#filenameinfo = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/genidealgradsdebugnotemplate/globalcor_0.root"

#filename = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/gennonidealgradsdebugnotemplate/globalcor_*.root"
#filenameinfo = "/data/home/bendavid/muonscale/CMSSW_10_6_17_patch1/work/gennonidealgradsdebugnotemplate/globalcor_0.root"

#filename = "/data/shared/muoncal/MuonGunUL2016_v41_RecDataMuIsoH_noquality/210405_115340/0000/globalcor_*.root"
#filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v41_RecDataMuIsoH_noquality/210405_115340/0000/globalcor_0_1.root"

#filename = "/data/shared/muoncal/MuonGunUL2016_v41_Rec_noquality/210405_115619/0000/globalcor_*.root"
#filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v41_Rec_noquality/210405_115619/0000/globalcor_0_1.root"

#filename = "/data/shared/muoncal/MuonGunUL2016_v42_RecDataMuIsoH_noquality/210405_185116/0000/globalcor_*.root"
#filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v42_RecDataMuIsoH_noquality/210405_185116/0000/globalcor_0_1.root"

#filenameinfo = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v36plus2_Rec_noquality/210405_185722/0000/globalcor_0_1.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v36plus2_Rec_noquality/210405_185722/0000/globalcor_*.root"

#filenameinfo = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v33_Rec_idealquality/210307_161142/0000/globalcor_0_1.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v33_Rec_idealquality/210307_161142/0000/globalcor_*.root"

#filenameinfo = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v54_Gen_idealquality/210429_092032/0000/globalcor_0_1.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v54_Gen_idealquality/210429_092032/0000/globalcor_*.root"

#filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v113_RecJpsiPhotos_quality_constraint/210721_170757/0000/globalcor_0_1.root"
#filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v121_Gen_idealquality/210725_154147/0000/globalcor_0_10.root"
#filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v171_RecJpsiPhotos_idealquality_constraint/210911_050035/0000/globalcor_0_1.root"
filenameinfo = "infofullbz.root"
#filename = "/data/shared/muoncal/MuonGunUL2016_v66_Rec_quality_nobs/210509_195947/0000/globalcor_*.root"

chain = ROOT.TChain("tree")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v66_Gen_quality/210509_200135/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v76_Gen_quality/210629_144601/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v84_GenJpsiPhotosSingle_quality/210707_164413/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v121_Gen_idealquality/210725_154147/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v121_Gen_idealquality_zeromaterial/210726_002805/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_GenJpsiPhotosSingle_quality/210719_142518/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_GenJpsiPhotosSingle_quality/210719_142518/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_Gen_quality/210728_174037/0000/globalcor_0_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v123_Gen_quality/210728_174037/0001/globalcor_0_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v85_Gen_quality/210708_095424/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v85_GenJpsiPhotosSingle_quality/210708_094852/0000/globalcor_*.root")


#chain.Add("/data/shared/muoncal/MuonGunUL2016_v82c_GenJpsiPhotosSingle_quality/210706_102419/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v81_GenJpsiPhotosSingle_quality/210705_124339/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v66_Rec_quality_nobs/210509_195947/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v81_RecJpsiPhotosSingle_quality/210704_225624/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v161a_Gen_idealquality_nograds/210901_180544/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v161a_Gen_idealquality_nograds/210901_180544/0001/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v161_GenJpsiPhotosSingle_idealquality_nograds/210901_160653/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v161_GenJpsiPhotosSingle_idealquality_nograds/210901_160653/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_Gen_idealquality/210913_032945/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_Gen_idealquality/210913_032945/0001/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_GenJpsiPhotosSingle_idealquality/210913_032642/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_GenJpsiPhotosSingle_idealquality/210913_032642/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_Gen_idealquality_zeromaterial_nograds/210913_155341/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_Gen_idealquality_zeromaterial_nograds/210913_155341/0001/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_GenJpsiPhotosSingle_idealquality_zeromaterial_nograds/210913_155453/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_GenJpsiPhotosSingle_idealquality_zeromaterial_nograds/210913_155453/0001/globalcor_*.root");


#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_Gen_idealquality/210930_200938/0000/*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_Gen_idealquality/210930_200938/0001/*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_GenJpsiPhotosSingle_idealquality/210930_201130/0000/*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_GenJpsiPhotosSingle_idealquality/210930_201130/0001/*.root");

chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_Gen_idealquality/211002_182020/0000/*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_Gen_idealquality/211002_182020/0001/*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_GenJpsiPhotosSingle_idealquality/211002_181835/0000/*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_GenJpsiPhotosSingle_idealquality/211002_181835/0001/*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_Gen_idealquality_zeromaterial/210930_201608/0000/*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_Gen_idealquality_zeromaterial/210930_201608/0001/*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_GenJpsiPhotosSingle_idealquality_zeromaterial/210930_201436/0000/*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v202_GenJpsiPhotosSingle_idealquality_zeromaterial/210930_201436/0001/*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v204_Gen_idealquality/211002_132131/0000/*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v204_Gen_idealquality/211002_132131/0001/*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v204_GenJpsiPhotosSingle_idealquality/211002_132415/0000/*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v204_GenJpsiPhotosSingle_idealquality/211002_132415/0001/*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v177_Gen_idealquality/210913_173054/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v177_Gen_idealquality/210913_173054/0001/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v177_GenJpsiPhotosSingle_idealquality/210913_172953/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v177_GenJpsiPhotosSingle_idealquality/210913_172953/0001/globalcor_*.root");

#filename = "/data/shared/muoncal/MuonGunUL2016_v30_Gen210206_025446/0000/globalcor_*.root"
#filenameinfo = "/data/shared/muoncal/MuonGunUL2016_v30_Gen210206_025446/0000/globalcor_0_1.root"




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
    #alpha = 1.0
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
    #return q*k*A + q*e*k**2 + M + q*R

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

@ROOT.Numba.Declare(["RVec<float>", "RVec<float>", "RVec<float>", "RVec<float>"], "RVec<float>")
def fitcorR(dxraws, dxfit, dyfit, rx):
    #rx = rx.reshape((-1, 2))
    #rx = onp.array(rx).reshape((-1, 2))
    rx = rx.copy().reshape(-1, 2)
    #rx = rx.copy().reshape(2, -1).transpose()
    #print(rx)
    #return dxraws - rx[:,0]*dxfit - rx[:,1]*dyfit
    return dxraws - rx[:,0]*dxfit - rx[:,1]*dyfit

treename = "tree"
#d = ROOT.ROOT.RDataFrame(treename,filename)
d = ROOT.ROOT.RDataFrame(chain)

d = d.Filter(jsonhelper, ["run", "lumi"], "jsonfilter")


#d = d.Define("dx", "Numba::fitcorR(dxrecgen,dlocalx,dlocaly,rx)")

#d = d.Define("dx", "dxrecgen - dlocalx")

#d = d.Define("dx", "dxrecgen - deigx")




#d = d.Define("dx", "E/Epred - 1.")
#d = d.Define("dx", "dE/dEpred - 1.")
#d = d.Define("dx", "dxrecgen")
#d = d.Define("dx", "dyrecgen")
#d = d.Define("dx", "dxsimgen")
#d = d.Define("dx", "dysimgen")
#d = d.Define("dx", "dxsimgenconv")
#d = d.Define("dx", "dysimgenconv")
#d = d.Define("dx", "dxsimgenlocal")
d = d.Define("dx", "dysimgenlocal")
#d = d.Define("dx", "dxrecsim")
#d = d.Define("dx", "dyrecsim")

#cut = "genPt > 5.5 && genPt < 150."
#d = d.Filter(cut)

#ptmin = 3.5
ptmin = 1.1
#ptmin = 0.9

d = d.Define("refPt", "std::abs(1./refParms[0])*std::sin(M_PI_2 - refParms[1])")
#d = d.Filter("refPt > 5.5")
d = d.Filter(f"genPt > {ptmin}")
#d = d.Filter("refPt > 26.")

d = d.Define("refCharge","std::copysign(1.0f, refParms[0])")

d = d.Filter("genEta>0.")
#d = d.Filter("genEta<0.")

#d = d.Filter("dE!= -99.")

#d = d.Filter("genEta>1.6 && genEta<2.2")
#d = d.Filter("genEta>1.3 && genEta<1.5")
#d = d.Filter("abs(genEta) < 0.8")
#d = d.Filter("genEta>2.3 && genEta<2.4")

#d = d.Filter("genEta>0.1 && genEta<0.3")
#d = d.Filter("genEta>-1.7 && genEta<-1.4")
#d = d.Filter("genEta>-2.4 && genEta<-2.3")

#d = d.Filter("hitidxv[0]==9")
#d = d.Filter("hitidxv[0]==24")

d = d.Define("hitidxr", "Numba::layer(hitidxv)")
d = d.Define("kgen", "(1./genPt)*dxsimgen/dxsimgen")
#d = d.Define("kgen", "(1./refPt)*dxrecgen/dxrecgen")

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

nptbins = 25
pts = 1./onp.linspace(150., 20., nptbins+1, dtype=np.float64)

ks = onp.concatenate((pts,ks[1:]),axis=0)



#nkbins = 40
#ks = 1./onp.linspace(150.,33.,nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./5.5, nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./12., nkbins+1, dtype=np.float64)







nkbinsfine = 1000
ksfine = onp.linspace(1./150., 1./ptmin, nkbinsfine+1, dtype=np.float64)

if (False):
    #override binning for isomu data
    nptbins = 25
    pts = 1./onp.linspace(150., 26., nptbins+1, dtype=np.float64)
    ks = pts
    ksfine = onp.linspace(1./150., 1./26., nkbinsfine+1, dtype=np.float64)


nkbins = ks.shape[0]-1


qcs = onp.array([-1.,1.], dtype=np.float64)
kcs = 0.5*(ks[1:] + ks[:-1])

kcsfine = 0.5*(ksfine[1:] + ksfine[:-1])

#nqrbins = 40000
#qrlim = 0.5

#nqrbins = 4000
qrlim = 5.0
#qrlim = 0.1

nqrbins = 4000
#qrlim = 1.0
#qrlim = 0.05
#qrlim = 0.01
#qrlim = 0.2

#qrlim = 0.05

#qrlim = 0.05
#qrlim = 0.02
#qrlim = 0.1
#qrlim = 0.025
#qrlim = 0.005
qrs = onp.linspace(-qrlim,qrlim,nqrbins+1,dtype=np.float64)

dminus = d.Filter("genCharge<0")
dplus = d.Filter("genCharge>0")

#dminus = d.Filter("refCharge<0")
#dplus = d.Filter("refCharge>0")

globs = onp.arange(nglobal+1)-0.5

hdxsimgenminus = dminus.Histo3D(("hdxsimgenminus", "", nglobal, globs, nkbins, ks, nqrbins, qrs),"hitidxr","kgen", "dx")
hdxsimgenplus = dplus.Histo3D(("hdxsimgenplus", "", nglobal, globs, nkbins, ks, nqrbins, qrs),"hitidxr","kgen", "dx")

#print(hdxsimgen)

print("starting rdf loop")

dxsimgenminus = hist2array(hdxsimgenminus.GetValue())
dxsimgenplus = hist2array(hdxsimgenplus.GetValue())

print("done converting hists")

dxsimgen = onp.stack([dxsimgenminus, dxsimgenplus], axis=1)

#dxsimgen = onp.reshape(dxsimgen, (nglobal, 2, 50, 10000))

print(dxsimgen.shape)

lsum = onp.sum(dxsimgen, axis=(1,2,3))

goodidxs = []
for idx in range(nglobal):
    if lsum[idx] > 10000.:
    #if lsum[idx] > 10000. and ( (parmlist[idx][0]==0 and parmlist[idx][1]==1) or parmlist[idx][0]==1 ):
    #if lsum[idx] > 10000. and parmlist[idx][0] < 2:
    #if lsum[idx] > 10000. and parmlist[idx][0]<4:
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
#xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=hbinned, edmtol = 1e-5)
xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=hbinned, edmtol = 1e-2)

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
