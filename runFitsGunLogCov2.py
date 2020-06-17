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


ROOT.gInterpreter.AddIncludePath("/usr/include/eigen3/")

#ROOT.gInterpreter.Declare("""
##include <Eigen/Dense>
    
#double symdeteigen(double c00, double c01, double c02, double c03, double c04, double c11, double c12, double c13, double c14, double c22, double c23, double c24, double c33, double c34, double c44) {
    
    #typedef Eigen::Matrix<double,5,5> Matrix5d;
    
    #Matrix5d M;
    #M << c00,c01,c02,c03,c04,  c01,c11,c12,c13,c14,  c02,c12,c22,c23,c24,  c03,c13,c23,c33,c34,  c04,c14,c24,c34,c44;
    
    #Eigen::SelfAdjointEigenSolver<Matrix5d> s;
    
    #//return log(M.determinant());
    #s.compute(M, Eigen::EigenvaluesOnly);
    #return s.eigenvalues()[1];
#}
#""")

ROOT.gInterpreter.Declare("""

typedef ROOT::Math::SMatrix<double,5> SMatrix55;
typedef ROOT::Math::SVector<double,5> SVector5;    

double eigval(const std::tuple<SVector5, SMatrix55> &eig) {
    
    const auto &e = std::get<0>(eig);
    const auto &v = std::get<1>(eig);
    
    double maxval = 0.;
    unsigned int imax = 0;
    for (unsigned int i=0; i<5; ++i) {
        double val = v(0,i)*v(0,i) + v(2,i)*v(2,i);
        if (val > maxval && v(0,i)*v(2,i)<0.) {
            maxval = val;
            imax = i;
        }
    }
    return e(imax);
    
}
""")


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
    #cut = f"1./reco_curv>{ptmin} && fabs(reco_eta)<2.4 && reco_charge>-2. && 1./reco_curv<150."
    #cut = f"1./reco_curv>{ptmin} && fabs(reco_eta)<2.4 && reco_charge>-2."
    
    
    
    #cut = f"mcpt{idx} > {ptmin} && fabs(eta{idx})<2.4"
    #cut += f" && gen_phi>=0. && gen_phi<0.4"
    #cut += f" && gen_phi>={-np.pi} && gen_phi<{np.pi/4.}"
    
    #cut += f" && eta{idx}>2.3"
    
    #print(cut)

    d = d.Filter(cut)
    
    #d = d.Filter("gen_eta<-2.3")

    #d = d.Define('krec', f'1./pt{idx}*(1.-cErr{idx}*cErr{idx})')
    #d = d.Define('kgen', f'1./mcpt{idx}')

    #d = d.Define('krec', f'1./pt{idx}')
    #d = d.Define('kr', "log(reco_curv/gen_curv)")
    #d = d.Define('kr', "reco_charge*reco_curv/gen_curv/gen_charge")
    #d = d.Define('kr', "tkptErr*tkptErr*reco_curv*reco_curv")
    #d = d.Define('kr', "tkptErr*tkptErr")
    #d = d.Define('kr', "tkptErr")
    #d = d.Define("kr", "symdeteigen(cov00,cov01,cov02,cov03,cov04,cov11,cov12,cov13,cov14,cov22,cov23,cov24,cov33,cov34,cov44)")
    #d = d.Define("kr", "std::get<0>(eig)[1]")
    #d = d.Define("kr", "cov00 + cov22 - 2.*cov02")
    d = d.Define("kr", "eigval(eig)")

    #d = d.Define('kr', "cov03/reco_curv/reco_curv")
    #d = d.Define('kr', "cov23/sqrt(cov22*cov33)")
    #d = d.Define('kr', "cov02/sqrt(cov00*cov22)")
    #d = d.Define('kr', "gen_curv/reco_curv")
    d = d.Define('kgen', "gen_curv")
    #d = d.Define('kgen', "reco_curv")
    
    #d = d.Filter("kr>=0. && kr<0.02")
    #d = d.Filter("kr>=(-log(3.)) && kr<log(3.)")
    
    #d = d.Define('krec', f'pt{idx}')
    #d = d.Define('kgen', f'mcpt{idx}')
    
    d = d.Define('eta', "gen_eta")
    d = d.Define('phi', "gen_phi")
    #d = d.Define('eta', "reco_eta")
    #d = d.Define('phi', "reco_phi")
    #d = d.Define('q', "reco_charge")
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
    #cut = f"pt{idx} > {ptmin} && fabs(eta{idx})<2.4"
    #cut += f" && phi{idx}>=0. && phi{idx}<0.4"
    #cut += f" && phi{idx}>={-np.pi} && phi{idx}<{np.pi/4.}"
    
    #cut += f" && eta{idx}>2.3"
    
    #print(cut)

    d = d.Filter(cut)
    #d = d.Filter(f"mceta{idx}<-2.3")

    #d = d.Define('krec', f'1./pt{idx}*(1.-cErr{idx}*cErr{idx})')
    #d = d.Define('kgen', f'1./mcpt{idx}')

    #d = d.Define('krec', f'1./pt{idx}')
    #d = d.Define('kr', f'mcpt{idx}/pt{idx}')
    #d = d.Define('kr', f'cErr{idx}*cErr{idx}/pt{idx}/pt{idx}')
    #d = d.Define('kr', f'cErr{idx}*cErr{idx}')
    #d = d.Define('kr', f'cErr{idx}')
    #d = d.Define("kr", f"symdeteigen(cov{idx}_00,cov{idx}_01,cov{idx}_02,cov{idx}_03,cov{idx}_04,cov{idx}_11,cov{idx}_12,cov{idx}_13,cov{idx}_14,cov{idx}_22,cov{idx}_23,cov{idx}_24,cov{idx}_33,cov{idx}_34,cov{idx}_44)")
    #d = d.Define('kr', f'std::get<0>(eig{idx})[1]')
    #d = d.Define('kr', f'cov{idx}_00 + cov{idx}_22 - 2.*cov{idx}_02')
    #d = d.Define('kr', f'cov{idx}_03*pt{idx}*pt{idx}')
    #d = d.Define('kr', f'cov{idx}_03')
    d = d.Define('kr', f'eigval(eig{idx})')
    #d = d.Define('kr', f'cov{idx}_23/sqrt(cov{idx}_22*cov{idx}_33)')
    #d = d.Define('kr', f'cov{idx}_02/sqrt(cov{idx}_00*cov{idx}_22)')
    #d = d.Define('kr', f'dxy{idx}_mcvtx')
    #d = d.Define('kr', f'cov{idx}_00')
    #d = d.Define('kr', f'cErr{idx}*cErr{idx}/pt{idx}/pt{idx}')
    #d = d.Define('kr', f'pt{idx}/mcpt{idx}')
    #d = d.Filter("kr>=0. && kr<0.05")
    
    #d = d.Define('kr', f'log(mcpt{idx}/pt{idx})')
    d = d.Define('kgen', f'1./mcpt{idx}')
    #d = d.Define('kgen', f'1./pt{idx}')
    
    #d = d.Filter("kr>=0. && kr<2.")
    
    #d = d.Filter("kr>=-log(3.) && kr<log(3.)")
    
    #d = d.Define('krec', f'pt{idx}')
    #d = d.Define('kgen', f'mcpt{idx}')
    
    #d = d.Define('eta', f'eta{idx}')
    #d = d.Define('phi', f'phi{idx}')
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


def scale(A,e,M,W,Y,Z,V,e2,Z2,a,b,c,d,k,q,eta):
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
    #e2 = np.sqrt(100.+e2**2) - 1.
    #Y = Y**2
    #e2 = (5000.+e2)**2
    #Z2 = (10.+Z2)**2
    ##Z = (1.+Z)**2
    #e2 = (1.+e2)**2
    #Z = 1e-4*Z
    
    #W  = 1e-4 + W
    #Z = 1e-4 + Z
    #Z = 0.
    #V = (1.+V)**2
    #Z2 = (1.+Z2)**2
    
    #res = (1.+A)*(1.+np.exp(W+Y)*k**2)/(1.+np.exp(Y)*k**2)*(1. +q*M/k)/(1. + e*k)
    #res = (1. + A + np.exp(W+Y)*k**2)/(1.+np.exp(Y)*k**2)*(1. +q*M/k)/(1. + e*k)
    #res = (1. + A + (1.+W)*Y*k**2)/(1.+Y*k**2)*(1. +q*M/k)
    
    #res = 1./(1.+e*k) + (A + W*Y*k**2)/(1. + Y*k**2) + q*M/k
    #res = 1. + (A + W*Y*k**2 * Z*e2*k**4)/(1. + Y*k**2 + e2*k**4) + q*M/k
    #res = 1. + (A + W*Y*k**2)/(1. + Y*k**2) + q*M/k
    #res = (1. + A + (1.+W)*Y*k**2 + (1.+Z)*e2*k**4)/(1. + Y*k**2 + e2*k**4) + q*M/k 
    #res = 1. + A + q*M/k + W/(1.+Y*k**2) + q*Z*k/(1.+Y*k**2) - e*k - e2*k/(1.+Y*k**2)
    #res = 1. + A + q*M/k + W/(1.+Y*k**2) + q*Z*k/(1.+e2*k**2)
    
    #res = 1. + A + W/(1.+Y*k**2) + Z2/k**2 + q*M/k + q*Z*k/(1.+Y*k**2) + q*V*k
    
    #res = 1. + A + W/(1.+Y*k**2) + Z2/k**2 + q*M/k + q*Z*k/(1.+Y*k**2) + q*V*k
    
    res = 1. + A + W/(1.+Y*k**2) + q*M/k + q*Z*k/(1.+Y*k**2) + q*V*k - e*k - e2*k/(1.+Y*k**2)
                                                                           
    #res = 1./(1.+e*k) + (A + W*Y*k**2)/(1. + Y*k**2) + q*M/k
    
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



def logsigpdfnorestrict(vals, mu, sigma):

    #mu = scale*kgen
    #sigma = res*kgen
    
    alpha = 1.
    alpha1 = alpha
    alpha2 = alpha
    
    A1 = np.exp(0.5*alpha1**2)
    A2 = np.exp(0.5*alpha2**2)
    
    t = (vals - mu)/sigma
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
    
    pdf = np.where(t<-alpha1, pdfleft, np.where(t<alpha2, pdfcore, pdfright))
    
    Icore = (scipy.special.ndtr(alpha2) - scipy.special.ndtr(-alpha1))*sigma*np.sqrt(2.*np.pi)
    Ileft = (sigma/alpha1)*np.exp(-0.5*alpha1**2)
    Iright = (sigma/alpha2)*np.exp(-0.5*alpha2**2)
    
    I = Icore + Ileft + Iright
    
    return pdf - np.log(I)

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
    
    #thigh = np.inf
    #tlow = -np.inf
    
    tlow = (0.-mu)/sigma
    thigh = (0.05-mu)/sigma
    
    
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
    
    
def logsigpdfbinnedmoy(mu,sigma,krs):
    
    krs = krs[np.newaxis,np.newaxis,np.newaxis,:]
    width = krs[...,1:] - krs[...,:-1]

    kr = 0.5*(krs[...,1:] + krs[...,:-1])
    
    t = (kr - mu)/sigma
    
    logpdf = np.exp(-0.5*(t+np.exp(-t)))
    
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
    
    #alpha = 1.0
    #alpha1 = alpha
    #alpha2 = alpha
    
    alpha1 = 1.
    alpha2 = 1.
    
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
    
    mu *= 1e-6
    #sigma *= 1e-6
    
    #sigma = np.sqrt(sigmasq)
    sigma = np.where(sigma>0.,sigma,np.nan)
    
    #mu = 1. + 0.1*np.tanh(mu)
    #sigma = 1e-3*(1. + np.exp(sigma))
    
    
    mu = mu[...,np.newaxis]
    sigma = sigma[...,np.newaxis]
    
    logpdf = logsigpdfbinnedmoy(mu,sigma,krs)
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

#parmscale = np.array([1e-4, 1e-3, 1e-5,1e-3, 1e-3, 1e-7,1.,1.,1.,1.,1.,1.,1.])
#parmscale = np.array([1e-4,1., 1e-5,1e-3, 1e-3, 1e-7,1.,1.,1.,1.,1.,1.,1.])

#parmscale = np.array([1.,1.,1.,1e-3, 1e-3, 1e-7,1.,1.,1.,1.,1.,1.,1.])

#parmscale = 1.

#parmscale  = np.ones((5,), dtype=np.float64)

#parmscale = np.array([1e-3, 1e-3, 1e-7,1.,1e-7,1.,1e-2])
#parmscale = np.array([1e-2, 1e-2, 1e-4,1.,1e-2,1.,1e-2])

#parmscale = np.array([1e-3, 1e-3, 1e-7,1.,1e-3,1.])

def nll(parms,dataset,eta,ntotal):
    parms = parms*parmscale
    
    a = parms[0]
    b = parms[1]
    c = parms[2]
    d = parms[3]
    e = parms[4]
    f = parms[5]
    s = parms[6]
    
    #s = s**2
    
    #e2 = parms[10]
    #Z2 = parms[11]
    
    #V = parms[10]
    #e2 = parms[11]
    ##W2 = parms[12]
    #Z2 = parms[13]

    #V = np.zeros_like(A)
    #W2 = np.zeros_like(A)

    #a = a**2
    ##b = b**2
    #c = c**2
    #d = d**2
    
    
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
    scalem = sigmasq(a,b,c,d,e,f,kgen,eta) + 0*q
    resm = s*scalem
    #resm = s
    #scalem = scale(A,e,M,W,Y,Z,V,e2,Z2,a,b,c,d,kgen, q, eta)
        
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
    #nll += 0.5*cweight*(Z**2 +  e2**2 + Z2**2 + V**2)
    #nll += 0.5*cweight*(Z**2 +  e2**2)
    
    #nll += 0.5*cweight/10.**2*W**2
    #nll += 0.5*cweight/1e6**2*Y**2
    #nll += 0.5*cweight/0.01**2*e**2
    #nll += 0.5*cweight*e**2
    #nll += 0.5*cweight*Z**2
    #nll += 0.5*cweight*e2**2
    #nll += 0.5*cweight*W**2
    #nll += 0.5*cweight*Y**2
    #nll += 0.5*cweight*Z**2
    
    #nll += 0.5*cweight*e**2
    #nll += 0.5*cweight*A**2
    #nll += 0.5*cweight*M**2
    #nll += 0.5*cweight*W**2
    #nll += 0.5*cweight*Z**2
    ##nll += 0.5*cweight*Y**2
    #nll += 0.5*cweight*e2**2
    
    #nll += 0.5*cweight*V**2
    ##nll += 0.5*cweight*b**2
    ##nll += 0.5*cweight*a
    ##nll += 0.5*cweight*c
    #nll += 0.5*cweight*Z2**2
    
    nll += 0.5*cweight*e**2
    nll += 0.5*cweight*f**2
    #nll += 0.5*cweight*b**2
    #nll += 0.5*cweight*d**2
    
    #nll += 0.5*cweight/1e-3**2*e**2
    #nll += 0.5*cweight*(Z**2)
    
    #nll += 0.5*cweight/1000.**2*(A**2 + W**2)
    #nll += 0.5*cweight/100.**2*W**2
    #nll += 0.5*cweight/100.**2*Y**2
    
    #nll = np.where(np.any(sigma<=0.,), np.nan, nll)
    
    return nll


def sigmasq(a,c,g1,d1,g2,d2,g3,d3,k,q):
    #l = computeTrackLength(eta)
    #res1 = a1*np.power(l1,2) + c1*np.power(l1,4)*np.power(p1,2) + b1*np.power(l1,2)/(1+d1/(np.power(p1,2)*np.power(l1,2)))
    #return a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    #return a*l**2 + c*l**4/k**2*(1.+b*k**2/l**2)/(1.+d*k**2/l**2)
    
    
    #a = a**2
    ###b = b**2
    #c = c**2
    #d = d**2
    
    #d = d**2
    ###e = e**2
    #f = f**2
    
    #a = a**2
    #c = c**2
    #g1 = g1**2
    #d1 = d1**2
    ##g2 = g2**2
    #d2 = d2**2
    ##g3 = g3**2
    #d3 = d3**2
    
    #d = np.sqrt(1.+d**2)-1.
    
    ##res = a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    #res = a*l**2 + c*l**4/k**2*(1.+b*k**2*l**2)/(1.+d*k**2*l**2)
    
    #res = a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    #res = (a + b*k**2 + c*k**4)/(1.+ d*k**2)/k**2
    #res = a + c/k**2 + b/(1.+d*k**2)
    ##V = (10.+V)**2
    #Z2 = (10.+Z2)**2
    
    #V = np.exp(V)
    #Z2 = np.exp(Z2)
    
    #V = (1.+V)**2
    #Z2 = (1.+Z2)**2
    
    #res = a + c/k**2 + b/(1.+d*k**2) + e/(1.+f*k**2)
    #res = a + c/k**2 + g1/(1.+d1*k**2) + g2/(1.+d2*k**2) + g3/(1.+d3*k**2)
    #res =  -a -c/k**2 - g2/(1.+d2*k**2) - g3/(1.+d3*k**2)
    #res =  -a*k**2 -c - g2/(1.+d2*k**2) - g3/(1.+d3*k**2)
    #res =  -a*k**2  -c - g2/(1.+d2*k**2) - g3/(1.+d3*k**2)
    #res =  a + c/k**2 + g2/(1.+d2*k**2) + g3/(1.+d3*k**2)
    #res = a*np.log(k) -g3 - np.log1p(d3*k**2)
    #res = a + g3/(1.+d3*k**2) + c*q/k + g1*q*k/(1.+d3*k**2) + g2/(1.+d2*k**2) + d1*q*k/(1.+d2*k**2)
    #res =  -a -c/k**2 - g2/(1.+d2*k**2) - g3/(1.+d3*k**2)
    res = a*k**2 + c + g2/(1.+d2*k**2)  + g3/(1.+d3*k**2)
    #res = a*k**2 + c + g1/(1.+d1*k**2) + g2/(1.+d2*k**2)  + g3/(1.+d3*k**2)


    #res = a + c/k**2
    
    #res = a + c/k**2 + b/(1.+d*k**2) + V/(1.+Z2*k**2)
    #res = a + c/k**2 + b/(1.+d*k**2)
    
    #res = a + 1./k**2*(c + b*k**2 + V*k**4)/(1.+d*k**2+Z2*k**4)
    #res = a**2 + c**2/k**2 + b/k + d**2*k**2
    #res = a**2 + c**2/k**2 + b**2/k + d**2*k**2
    
    return res

#parmscale = np.array([1e-3 ,1e-7, 1e-3, 1.,1e-3,1.,1e-3, 1.])
parmscale = np.ones((8,),dtype=np.float64)

def nllbinnedmodel(parms, scales, errs, ks, qs):
    parms = parms*parmscale
    
    a = parms[0]
    c = parms[1]
    g1 = parms[2]
    d1 = parms[3]
    g2 = parms[4]
    d2 = parms[5]
    g3 = parms[6]
    d3 = parms[7]
    
    #mval = a + c/ks**2 + b/(1.+d*ks**2)
    
    mval = sigmasq(a,c,g1,d1,g2,d2,g3,d3,ks,qs)
    
    #nll = 0.5*(mval-scales)**2/errs**2
    nll = 0.5*(mval-scales)**2/errs**2
    nll = np.sum(nll)
    
    #nll += a**2
    #nll += c**2
    nll += g1**2
    nll += d1**2
    #nll += g2**2
    #nll += d2**2
    #nll += g3**2
    #nll += d3**2
    
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

f = f"{dataDir}/muGunCov_eig.root"
#fdj = f"{dataDir}/muonTree.root"
fdj = f"{dataDir}/JpsiToMuMu_JpsiPt8_pythia8_eig.root"
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

#nEtaBins = 24
#etas = onp.linspace(-1.2,1.2, nEtaBins+1, dtype=np.float64)

#nEtaBins = 28
#nEtaBins = 24
####nEtaBins = 480
#etas = onp.linspace(-1.4,1.4, nEtaBins+1, dtype=np.float64)

#nEtaBins = 10
#etas = onp.linspace(-2.4,-1.4, nEtaBins+1, dtype=np.float64)

#nEtaBins = 1
#etas = np.linspace(-2.4,-2.3, nEtaBins+1)
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
ptmax = 150.

nPhiBins = 1
phis = onp.linspace(-np.pi,np.pi, nPhiBins+1, dtype=np.float64)

nkbins = 25
ks = onp.linspace(1./20., 1./5.5, nkbins+1, dtype=np.float64)

#nptbins = 40
#ks = 1./onp.linspace(150., 5.5, nptbins+1, dtype=np.float64)

nptbins = 100
#nptbins = 20
pts = 1./onp.linspace(ptmax, 20., nptbins+1, dtype=np.float64)

ks = onp.concatenate((pts,ks[1:]),axis=0)

#nkbins = 40
#ks = 1./onp.linspace(150.,33.,nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./5.5, nkbins+1, dtype=np.float64)
##ks = onp.linspace(1./150., 1./12., nkbins+1, dtype=np.float64)



nkbins = ks.shape[0]-1


nkbinsfine = 1000
#ksfine = 1./onp.linspace(100., 3.3, nkbinsfine+1, dtype=np.float64)
ksfine = onp.linspace(1./ptmax, 1./5.5, nkbinsfine+1, dtype=np.float64)

qs = onp.array([-1.5,0.,1.5], dtype=np.float64)
#qs = onp.array([0.,1.5], dtype=np.float64)

nqrbins = 100000
#qrmin = -5e-3
#qrmax = 5e-3

#qrmin = -100.
#qrmax = -60.

#qrmin = -5e-7
#qrmax = 5e-7

qrmin = 0.
qrmax = 5e-7

#qrmin = -5e-8
#qrmax = 5e-8

#qrmin = 0.
#qrmax = 1.0
#qrmin = -2.
#qrmax = 2.
qrs = onp.linspace(qrmin,qrmax,nqrbins+1,dtype=np.float64)
qrsingle = onp.array([qrmin, qrmax],dtype=np.float64)
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

fgbinned = lbatch(fgbinned, batch_size=int(1), in_axes=(0,0,None))
hbinned = lbatch(hbinned, batch_size=int(1), in_axes=(0,0,None))


#xmu = np.zeros((nEtaBins,2,nkbins),dtype=np.float64)
xmu = 1e-4*np.ones((nEtaBins,2,nkbins),dtype=np.float64)
xsigma = 0.01*np.ones((nEtaBins,2,nkbins),dtype=np.float64)
xbinned = np.stack((xmu,xsigma),axis=-1)

#val = nllbinned(xbinned, hdset, qrs)
#assert(0)

htest = hbinned(xbinned,hdset,qrs)
print(htest.shape)



#xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=None, edmtol = 1e-3, reqposdef = False)

#xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=None)
xbinned = pmin(fgbinned, xbinned, (hdset,qrs), jac=True, h=hbinned)

hessbinned = hbinned(xbinned, hdset, qrs)
covbinned = np.linalg.inv(hessbinned)

errsbinned =  np.sqrt(np.diagonal(covbinned, offset=0, axis1=-1, axis2=-2))

#errsbinned = 2.*xbinned*errsbinned
#xbinned = xbinned**2


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




fg = jax.value_and_grad(nllbinnedmodel)
h = jax.hessian(nllbinnedmodel)



#fg = jax.value_and_grad(nll)
#h = jax.hessian(nll)
##h = jax.jacrev(jax.grad(nll))

fg = jax.jit(fg)
h = jax.jit(h)

#if forcecpu:
    ##fg = pbatch_accumulate(fg, batch_size=int(1e9),ncpu=32, in_axes=(None,0,None))
    ##h = pbatch_accumulate(h, batch_size=int(1e9),ncpu=32, in_axes=(None,0,None))

    #fg = lpbatch_accumulate(fg, batch_size=int(1e6),ncpu=32, in_axes=(None,0,None,None))
    #h = lpbatch_accumulate(h, batch_size=int(1e5),ncpu=32, in_axes=(None,0,None,None))
    
    ##fg = jax.jit(fg)
    ##h = jax.jit(h)
    #pass
#else:
    #fg = lbatch_accumulate(fg, batch_size=int(1e6), in_axes=(None,0,None,None))
    #h = lbatch_accumulate(h, batch_size=int(1e5), in_axes=(None,0,None,None))


#fg = lpbatch_accumulate(fg, batch_size=16384, ncpu=32, in_axes=(None,0,None))

#fg = batch_fun(fg, batch_size=10e3)
#h = batch_fun(h, batch_size=10e3)

#h = jax.hessian(nll)
#h = jax.jit(jax.hessian(nll))
#h = jax.jit(jax.jacrev(jax.grad(nll)))



def scalesigma(x, etasc, ks):
    
    #A = x[...,0,np.newaxis,np.newaxis]
    #e = x[...,1,np.newaxis,np.newaxis]
    #M = x[...,2,np.newaxis,np.newaxis]
    ##W = x[...,3,np.newaxis,np.newaxis]
    ##Y = x[...,4,np.newaxis,np.newaxis]
    ##Z = x[...,5,np.newaxis,np.newaxis]
    #a = x[...,3,np.newaxis,np.newaxis]
    #b = x[...,4,np.newaxis,np.newaxis]
    #c = x[...,5,np.newaxis,np.newaxis]
    #d = x[...,6,np.newaxis,np.newaxis]
    #W = x[...,7,np.newaxis,np.newaxis]
    #Y = x[...,8,np.newaxis,np.newaxis]
    #Z = x[...,9,np.newaxis,np.newaxis]
    #V = x[...,10,np.newaxis,np.newaxis]
    #e2 = x[...,11,np.newaxis,np.newaxis]
    #Z2 = x[...,12,np.newaxis,np.newaxis]
    ##e2 = x[...,10,np.newaxis,np.newaxis]
    ##Z2 = x[...,11,np.newaxis,np.newaxis]

    ##V = np.zeros_like(A)
    ##W2 = np.zeros_like(A)
    ##e2 = np.zeros_like(A)
    ##Z2 = np.zeros_like(A)

    ##a = a**2
    ##b = b**2
    ##c = c**2
    ##d = d**2
    
        
    #a = parms[0]
    #b = parms[1]
    #c = parms[2]
    #d = parms[3]
    #s = parms[4]
    
    a = x[...,0,np.newaxis,np.newaxis]
    c = x[...,1,np.newaxis,np.newaxis]
    g1 = x[...,2,np.newaxis,np.newaxis]
    d1 = x[...,3,np.newaxis,np.newaxis]
    g2 = x[...,4,np.newaxis,np.newaxis]
    d2 = x[...,5,np.newaxis,np.newaxis]
    g3 = x[...,6,np.newaxis,np.newaxis]
    d3 = x[...,7,np.newaxis,np.newaxis]
    
    #s = x[...,6,np.newaxis,np.newaxis]
    

    qs = np.array([-1.,1.],dtype=np.float64)
    #TODO make this more elegant and dynamic
    if len(x.shape)>1:
        qs = qs[np.newaxis,:,np.newaxis]
    else:
        qs = qs[:,np.newaxis]

    #scaleout = scale(A,e,M,W,Y,Z,ks, qs, etasc[...,np.newaxis,np.newaxis])
    scaleout = sigmasq(a,c,g1,d1,g2,d2,g3,d3, ks, qs) + 0.*qs
    #sigmaout = np.sqrt(sigmasqout)
    sigmaout = np.zeros_like(scaleout)
    #scaleout = scale(A,e,M,W,Y,Z,V,e2,Z2,a,b,c,d,ks, qs, etasc[...,np.newaxis,np.newaxis])
    
    
    #sigmaout = sigmaout*np.ones_like(qs)
    
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
    scales = xbinned[ieta,...,0]
    errs = errsbinned[ieta,...,0]
    ksfit = hdsetks[ieta,...]
    
    #print("scales errs ks")
    #print(scales.shape)
    #print(errs.shape)
    #print(ksfit.shape)
    
    #assert(0)
    
    a = 1.
    c = 1.
    g1 = 0.
    d1 = 50.
    g2 = 0.
    d2 = 200.
    g3 = 0.
    d3 = 5000.

    #s = 1.

    x = np.stack((a,c,g1,d1,g2,d2,g3,d3)).astype(np.float64)

    qs = np.array([-1.,1.],dtype=np.float64)
    qs = qs[:,np.newaxis]
    
    x = pmin(fg, x, (scales,errs,ksfit,qs), doParallel=False, jac=True, h=None,xtol=1e-14,edmtol=1e-3)
    x = pmin(fg, x, (scales,errs,ksfit,qs), doParallel=False, jac=True, h=h, edmtol = 1e-4)
    
    print("computing hess")
    hess = h(x, scales, errs,ksfit,qs)
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
                     
onp.savez_compressed("unbinnedfiterrseig.npz",
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

