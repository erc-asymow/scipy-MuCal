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


#filename = "results_V33_01p567idealquality/correctedTracks.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v33_Rec_idealquality/210307_161142/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v54_Rec_idealquality/210429_092212/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v55_Rec_idealquality/210429_133749/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v56_Rec_idealquality/210429_145128/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v57_Rec_idealquality/210429_150243/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v58_Rec_idealquality/210430_002404/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v59_Rec_idealquality/210430_012633/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v60_Rec_quality/210430_223749/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/MuonGunDesign_GeantInt/MuonGunUL2016_v61_Rec_idealquality/210505_005651/0000/globalcor_0_*.root"
#filename = "/data/shared/muoncal/MuonGunUL2016_v61_Rec_quality_nobs/210505_074303/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v63_RecFromMuons_quality_nobs/210506_232413/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/DoubleMuonGun_Pt3To150/MuonGunUL2016_v63_Rec_quality_nobs/210506_234558/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/JPsiToMuMuGun_Pt5To30-pythia8-photos/MuonGunUL2016_v63_RecJpsiPhotosSingle_quality/210506_232140/0000/globalcor_*.root"
#filename = "root://eoscms.cern.ch//store/cmst3/group/wmass/bendavid/muoncal/JPsiToMuMuGun_Pt5To30-pythia8-photos/MuonGunUL2016_v64_RecJpsiPhotosSingle_quality/210508_125153/0000/globalcor_*.root"
#filename = "/data/shared/muoncal/MuonGunUL2016_v66_Rec_quality_nobs/210509_195947/0000/globalcor_*.root"
#filename = "/data/shared/muoncal/MuonGunUL2016_v66_Rec_quality_nobs/210509_195947/0000/globalcor_*.root"
#filename = "/data/shared/muoncal/MuonGunUL2016_v68_Rec_quality_nobs/210621_095523/0000/globalcor_*.root"
#filename = "/data/shared/muoncal/MuonGunUL2016_v74_Rec_quality_nobs/210628_222143/0000/globalcor_*.root"
#filename = "/data/shared/muoncal/MuonGunUL2016_v75_Rec_quality_nobs/210629_084405/0000/globalcor_*.root"
#filename = "correctedTracks.root"

chain = ROOT.TChain("tree")

#chain.Add("correctedTracks.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v121_Rec_idealquality_zeromaterial/210726_002951/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v123a_RecJpsiPhotosSingle_idealquality_zeromaterial/210728_090339/0000/globalcor_*.root")


#chain.Add("/data/shared/muoncal/MuonGunUL2016_v66_Rec_quality_nobs/210509_195947/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v84_RecJpsiPhotosSingle_quality/210707_164015/0000/globalcor_*.root")


#chain.Add("/data/shared/muoncal/MuonGunUL2016_v85_Rec_quality_nobs/210708_095233/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v85_RecJpsiPhotosSingle_quality/210708_095030/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v87_Rec_quality_nobs/210709_013206/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v87_RecJpsiPhotosSingle_quality/210709_013609/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v88_Rec_quality_nobs/210709_112232/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v88_RecJpsiPhotosSingle_quality/210709_112548/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v92_Rec_quality_nobs/210712_163945/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v92_RecJpsiPhotosSingle_quality/210712_163622/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v93_Rec_quality_nobs/210712_211420/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v93_RecJpsiPhotosSingle_quality/210712_211742/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v97_Rec_quality_nobs/210714_153858/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v97_RecJpsiPhotosSingle_quality/210714_153322/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v98_Rec_quality_nobs/210715_164101/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v98_RecJpsiPhotosSingle_quality/210715_163814/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v99_Rec_idealquality_zeromaterial/210717_161218/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v100_Rec_idealquality_zeromaterial/210717_200424/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v102_Rec_quality_nobs/210718_163642/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v102_RecJpsiPhotosSingle_quality/210718_163829/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v104_Rec_quality_nobs/210718_210633/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v104_RecJpsiPhotosSingle_quality/210718_211001/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v105_Rec_quality_nobs/210718_212330/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v105_RecJpsiPhotosSingle_quality/210718_212511/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v106_Rec_quality_nobs/210719_042053/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v106_RecJpsiPhotosSingle_quality/210719_042246/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v121_Rec_idealquality/210725_154031/0000/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_Rec_quality_nobs/210719_142714/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_Rec_quality_nobs/210719_142714/0001/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_RecJpsiPhotosSingle_quality/210719_142322/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v107_RecJpsiPhotosSingle_quality/210719_142322/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v108_Rec_quality_nobs/210720_152125/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v108_Rec_quality_nobs/210720_152125/0001/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v108_RecJpsiPhotosSingle_quality/210720_152437/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v108_RecJpsiPhotosSingle_quality/210720_152437/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v110a_Rec_quality_nobs/210721_023323/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v110a_Rec_quality_nobs/210721_023323/0001/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v110_RecJpsiPhotosSingle_quality/210721_023415/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v110_RecJpsiPhotosSingle_quality/210721_023415/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v111_Rec_quality_nobs/210721_024445/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v111_Rec_quality_nobs/210721_024445/0001/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v111_RecJpsiPhotosSingle_quality/210721_024543/0000/globalcor_*.root")
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v111_RecJpsiPhotosSingle_quality/210721_024543/0001/globalcor_*.root")

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v66_RecJpsiPhotosSingle_quality/210509_200944/0000/globalcor_*.root")


#chain.Add("/data/shared/muoncal/MuonGunUL2016_v161_Rec_idealquality/210901_104621/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v161_Rec_idealquality/210901_104621/0001/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v161_RecJpsiPhotosSingle_idealquality/210901_104513/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v161_RecJpsiPhotosSingle_idealquality/210901_104513/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v198a_Rec_quality_nobs/210927_160634/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v198a_Rec_quality_nobs/210927_160634/0001/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v198_RecJpsiPhotosSingle_quality/210927_154536/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v198_RecJpsiPhotosSingle_quality/210927_154536/0001/globalcor_*.root");

#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_Gen_idealquality_zeromaterial_nograds/210913_155341/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_Gen_idealquality_zeromaterial_nograds/210913_155341/0001/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_GenJpsiPhotosSingle_idealquality_zeromaterial_nograds/210913_155453/0000/globalcor_*.root");
#chain.Add("/data/shared/muoncal/MuonGunUL2016_v171_GenJpsiPhotosSingle_idealquality_zeromaterial_nograds/210913_155453/0001/globalcor_*.root");

chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_Gen_idealquality/211002_182020/0000/*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_Gen_idealquality/211002_182020/0001/*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_GenJpsiPhotosSingle_idealquality/211002_181835/0000/*.root");
chain.Add("/data/shared/muoncal/MuonGunUL2016_v205_GenJpsiPhotosSingle_idealquality/211002_181835/0001/*.root");


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
    #return q*A*k + M - q*e*k**2
    return q*A*k + M - e*k**2
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
    
    #e = parms[..., 1]
    #sige = 1e-6
    #nll += np.sum(np.square(e), axis=-1)/sige/sige
    
    
    return nll
    

#@ROOT.Numba.Declare(["RVec<float>", "RVec<float>"], "float")
#def sigmaweight(parms, cov):
    #cov = onp.reshape(cov.copy(), (5,5))
    
    #p = onp.abs(1./parms[0])
    #p2 = p*p
    #pt = p*onp.sin(onp.pi/2. - parms[1])
    #pt2 = pt*pt
    #pz = p*onp.cos(onp.pi/2. - parms[1])
    #q = onp.sign(parms[0])
    
    #sigmapt2 = pt2*p2*cov[0,0] + 2.*p*pt*q*pz*cov[0,1] + pz*pz*cov[1,1]
    
    
    #return pt2/sigmapt2
    
#ROOT.gInterpreter.Declare("""
#float sigmaweight(const ROOT::VecOps::RVec<float> &parms, const ROOT::VecOps::RVec<float> &cov) {
    #const float p = std::abs(1./parms[0]);
    #const float p2 = p*p;
    #const float pt = p*std::sin(M_PI_2 - parms[1]);
    #const float pt2 = pt*pt;
    #const float pz = p*std::cos(M_PI_2 - parms[1]);
    #const float q = parms[0] > 0 ? 1. : -1.;
    
    #float sigmapt2 = pt2*p2*cov[0] + 2.*p*pt*q*pz*cov[1] + pz*pz*cov[6];
    #return pt2/sigmapt2;

#}
    
#""")

ROOT.gInterpreter.Declare("""
float sigmaweight(const ROOT::VecOps::RVec<float> &parms, const ROOT::VecOps::RVec<float> &cov) {
    return parms[0]*parms[0]/cov[0];

}
    
""")


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



if True:

    corfilename = "calibrationMCJul18-2021.root"

    corfile = ROOT.TFile.Open(corfilename)
    hA = corfile.Get("A")
    he = corfile.Get("e")
    hM = corfile.Get("M")

    #print(hA)
    #print(type(hA))

    corA = onp.zeros((48,), dtype=onp.float32)
    core = onp.zeros((48,), dtype=onp.float32)
    corM = onp.zeros((48,), dtype=onp.float32)

    for ibin in range(48):
        corA[ibin] = hA.GetBinContent(ibin+1)
        core[ibin] = he.GetBinContent(ibin+1)
        corM[ibin] = hM.GetBinContent(ibin+1)
elif True:
    corfilename = "plotscale_v84_ref/unbinnedfitglobalitercorscale.npz"
    corfile = onp.load(corfilename)
    xs = corfile["xs"]
    
    print(xs.shape)
    
    corA = -xs[:,0]
    core = -xs[:,1]
    corM = -xs[:,2]
    

print(corA)
print(core)
print(corM)

#assert(0)

@ROOT.Numba.Declare(["RVec<float>"], "float")
def curvcor(parms):
    p = onp.abs(1./parms[0])
    pt = p*onp.sin(onp.pi/2. - parms[1])
    q = onp.sign(parms[0])
    eta = -onp.log(onp.tan(0.5*(onp.pi/2. - parms[1])))
    ieta = int( (eta+2.4)/0.1 )
    #ieta = onp.clip(ieta, 0, 47)
    ieta = onp.maximum(0, ieta)
    ieta = onp.minimum(47, ieta)
    
    #print(parms[0])
    #print(parms[1])
    #print(p)
    #print(pt)
    #print(eta)
    #print(ieta)
    
    
    A = corA[ieta]
    e = core[ieta]
    M = corM[ieta]
    
    cor = 1. + A - e/pt + q*M*pt
    
    return cor
    

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
#ptmin = 5.5

#cut = "genPt > 5.5 && genPt < 150. && nValidHits > 0"
cut = f"genPt > {ptmin} && genPt < 150. && nValidHits > 0"
#cut = f"genPt > {ptmin} && genPt < 150. && nValidHits > 9 && nValidPixelHits > 0"

d = d.Filter(cut)

#d = d.Filter("edmval < 0.2 && !std::isinf(edmval) && !std::isnan(edmval)")

#d = d.Define("recParms", "trackParms");
d = d.Define("recParms", "refParms");
#d = d.Define("recParms", "corParms");

d = d.Define("recPt", "std::abs(1./recParms[0])*std::sin(M_PI_2 - recParms[1])");
d = d.Define("recCharge", "std::copysign(1.0,recParms[0])");

#d = d.Define("kr", "genPt*genCharge/recPt/recCharge");
#d = d.Define("kr", "deltaphi(genPhi, recParms[2])");
#d = d.Define("kr", "dxrecgen[0]");
d = d.Define("kr", "dysimgen[nValidHits-1]");
#d = d.Define("kr", "genPt*genCharge/recPt/recCharge*Numba::curvcor(refParms)");
#d = d.Define("kr", "genPt*genCharge/recPt/recCharge*Numba::curvcor(genParms)");
d = d.Define('kgen', "1./genPt")

#d = d.Define("sigw", "sigmaweight(recParms, refCov)")
#d = d.Define("sigw", "sigmaweight(genParms, refCov)")

#d = d.Filter("!std::isinf(kr) && !std::isnan(kr)")

#qrmin = -0.01
#qrmax = 0.01

#qrlim = 0.05
qrlim = 0.5
qrmin = -qrlim
qrmax = qrlim

#qrmin = 0.8
#qrmax = 1.2

d = d.Filter(f"kr>={qrmin} && kr<{qrmax}")

#d = d.Filter("hitidxv[0]==9")



#d = d.Filter("kr>=(-log(3.)) && kr<log(3.)")

#d = d.Define('krec', f'pt{idx}')
#d = d.Define('kgen', f'mcpt{idx}')

d = d.Define('eta', "genEta")
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

ptmax = 150.
#ptmax = 20.

#nkbins = 25
nkbins = 20
#ks = onp.linspace(1./20., 1./5.5, nkbins+1, dtype=np.float64)
ks = onp.linspace(1./20., 1./ptmin, nkbins+1, dtype=np.float64)

#nptbins = 40
#ks = 1./onp.linspace(150., 5.5, nptbins+1, dtype=np.float64)

nptbins = 25
pts = 1./onp.linspace(150., 20., nptbins+1, dtype=np.float64)

ks = onp.concatenate((pts,ks[1:]),axis=0)


#pts = 1./onp.linspace(ptmax, 20., nptbins+1, dtype=np.float64)


#nkbins = 40
#ks = 1./onp.linspace(150.,33.,nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./5.5, nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./12., nkbins+1, dtype=np.float64)



nkbins = ks.shape[0]-1

nkbinsfine = 1000
#ksfine = onp.linspace(1./ptmax, 1./5.5, nkbinsfine+1, dtype=np.float64)
ksfine = onp.linspace(1./ptmax, 1./ptmin, nkbinsfine+1, dtype=np.float64)

qcs = onp.array([-1.,1.], dtype=np.float64)
kcs = 0.5*(ks[1:] + ks[:-1])

kcsfine = 0.5*(ksfine[1:] + ksfine[:-1])

#nqrbins = 20000
nqrbins = 4000
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


#nEtaBins = 1
#etamin = 1.3
#etamax = 1.5

#nEtaBins = 24
####nEtaBins = 480
etas = onp.linspace(etamin, etamax, nEtaBins+1, dtype=np.float64)

dminus = d.Filter("genCharge<0")
dplus = d.Filter("genCharge>0")



#weight = "Numba::sigmaweight(recParms, refCov)"
#weight = "sigmaweight(recParms, refCov)"
#hdxsimgenminus = dminus.Histo3D(("hdxsimgenminus", "", nEtaBins, etas, nkbins, ks, nqrbins, qrs),"eta","kgen", "kr", "sigw")
#hdxsimgenplus = dplus.Histo3D(("hdxsimgenplus", "", nEtaBins, etas, nkbins, ks, nqrbins, qrs),"eta","kgen", "kr", "sigw")

hdxsimgenminus = dminus.Histo3D(("hdxsimgenminus", "", nEtaBins, etas, nkbins, ks, nqrbins, qrs),"eta","kgen", "kr")
hdxsimgenplus = dplus.Histo3D(("hdxsimgenplus", "", nEtaBins, etas, nkbins, ks, nqrbins, qrs),"eta","kgen", "kr")


#testhist = hdxsimgenplus.ProjectionZ("testhist", 14, 14, 45, 45)
#testhist.Rebin(20)
#testhist.Draw()

#input("wait")

#hdxsimgenminus = dminus.Histo3D(("hdxsimgenminus", "", nEtaBins, etamin, etamax, nkbins, 1./150., 1./5.5, nqrbins, qrmin, qrmax),"eta","kgen", "kr")
#hdxsimgenplus = dplus.Histo3D(("hdxsimgenplus", "", nEtaBins, etamin, etamax, nkbins, 1./150., 1./5.5, nqrbins, qrmin, qrmax),"eta","kgen", "kr")

#print(hdxsimgen)

print("starting rdf loop")

hdxsimgenminus.Scale(hdxsimgenminus.GetEntries()/hdxsimgenminus.GetSumOfWeights())
hdxsimgenplus.Scale(hdxsimgenplus.GetEntries()/hdxsimgenplus.GetSumOfWeights())

dxsimgenminus = hist2array(hdxsimgenminus.GetValue())
dxsimgenplus = hist2array(hdxsimgenplus.GetValue())

print("done converting hists")

dxsimgen = onp.stack([dxsimgenminus, dxsimgenplus], axis=1)


#testhist = dxsimgen[13, 1, -1, :]
#plot = plt.figure()
#plt.plot(1./kcfine, scalemodelfineplus-1.)

#plt.show()


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
