import os
import multiprocessing

#ncpu = multiprocessing.cpu_count()

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32" 

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
from obsminimization import pmin,batch_vmap,jacvlowmemb, batch_accumulate, lbatch_accumulate, pbatch_accumulate,lpbatch_accumulate, pbatch_accumulate_simple
#from calInput import makeData
import argparse
import functools
import time
import sys
from header import CastToRNode


ROOT.gROOT.ProcessLine(".L src/module.cpp+")
ROOT.gROOT.ProcessLine(".L src/applyCalibration.cpp+")

ROOT.ROOT.EnableImplicitMT()


def makeData(inputFile, genMass=False, smearedMass=False, isData=False):

    RDF = ROOT.ROOT.RDataFrame
    isJ = True
    restrictToBarrel = False
    dataDir = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata"
    runClosure = False

    if isJ:
        cut = 'pt1>4.3 && pt2>4.3 && pt1<25. && pt2<25.'# && mass>2.9 && mass<3.3'
    else:
        cut = 'pt1>20.0 && pt2>20.0 '#&& mass>75. && mass<115.'
    if restrictToBarrel:
        cut+= '&& fabs(eta1)<0.8 && fabs(eta2)<0.8'
    else:
        cut+= '&& fabs(eta1)<2.4 && fabs(eta2)<2.4' 
    if not isData:
        cut+= '&& mcpt1>0. && mcpt2>0.'

    d = RDF('tree',inputFile)

    if smearedMass:
        NSlots = d.GetNSlots()
        ROOT.gInterpreter.ProcessLine('''
                        std::vector<TRandom3> myRndGens({NSlots});
                        int seed = 1; // not 0 because seed 0 has a special meaning
                        for (auto &&gen : myRndGens) gen.SetSeed(seed++);
                        '''.format(NSlots = NSlots))


    if isJ:
        cut = 'pt1>4.3 && pt2>4.3 && pt1<25. && pt2<25.'# && mass>2.9 && mass<3.3'
        
    else:
        cut = 'pt1>20.0 && pt2>20.0 && mass>80. && mass<100.'

    if restrictToBarrel:
        cut+= '&& fabs(eta1)<0.8 && fabs(eta2)<0.8'

    else:
        cut+= '&& fabs(eta1)<2.4 && fabs(eta2)<2.4' 

    if genMass:

        cut+= '&& mcpt1>0. && mcpt2>0.'

    print(cut)

    

    d = d.Filter(cut)\
        .Define('v1', 'ROOT::Math::PtEtaPhiMVector(pt1,eta1,phi1,0.105658)')\
        .Define('v2', 'ROOT::Math::PtEtaPhiMVector(pt2,eta2,phi2,0.105658)')
    
    
    
    if smearedMass:
        d = d.Define('v1sm', 'ROOT::Math::PtEtaPhiMVector(mcpt1+myRndGens[rdfslot_].Gaus(0., cErr1*mcpt1),eta1,phi1,0.105658)')\
            .Define('v2sm', 'ROOT::Math::PtEtaPhiMVector(mcpt2+myRndGens[rdfslot_].Gaus(0., cErr2*mcpt2),eta2,phi2,0.105658)')\
            .Define('smearedgenMass', '(v1sm+v2sm).M()')
    else:
        d = d.Define('calcMass','(v1+v2).M()')
        d = d.Filter("calcMass>=2.9 && calcMass<=3.3")

    f = ROOT.TFile.Open('%s/bFieldMap.root' % dataDir)
    bFieldMap = f.Get('bfieldMap')

    if runClosure: print('taking corrections from', '{}/scale_{}_80X_13TeV.root'.format(dataDir, 'DATA' if isData else 'MC'))

    f2 = ROOT.TFile.Open('{}/scale_{}_80X_13TeV.root'.format(dataDir, 'DATA' if isData else 'MC'))
    A = f2.Get('magnetic')
    e = f2.Get('e')
    M = f2.Get('B')

    module = ROOT.applyCalibration(bFieldMap, A, e, M, isData, runClosure)

    d = module.run(CastToRNode(d))
    
    mass = 'calcMass'
    #mass = 'mass'

    if smearedMass:
        mass = 'smearedgenMass'
    
    cols=[mass,'eta1', 'pt1', 'phi1', 'eta2', 'pt2', 'phi2']
    
    if genMass:
        cols.append('genMass')
    
    data = d.AsNumpy(columns=cols)

    return data

def scale(A,e,M,pt,q):
    return 1. + A - e/pt + q*pt*M

def sigmasq(a,c,pt,eta):
    l = computeTrackLength(eta)
    #res1 = a1*np.power(l1,2) + c1*np.power(l1,4)*np.power(p1,2) + b1*np.power(l1,2)/(1+d1/(np.power(p1,2)*np.power(l1,2)))
    return a*l**2 + c*l**4*pt**2

def sigpdf(mass, scale, res):
    ml = 2.9
    mh = 3.3
    
    m0 = 3.09692
    
    mu = scale*m0
    sigma = res*m0
    
    pdf = np.exp(-0.5*(mass-mu)**2/sigma**2)/sigma/np.sqrt(2.*np.pi)
    
    #I=1.
    
    #I = 0.5*(scipy.special.erf((mh-mu)/sigma/np.sqrt(2.)) - scipy.special.erf((ml-mu)/sigma/np.sqrt(2.)))
    
    I = scipy.special.ndtr((mh-mu)/sigma) - scipy.special.ndtr((ml-mu)/sigma)
    #I = 1.
    
    return pdf/I
    

def nll(parms,dataset,etas):
    A = parms[0]
    e = parms[1]
    M = parms[2]
    a = parms[3]
    c = parms[4]
    
    A = 0.01*np.tanh(A)
    e = 0.01*np.tanh(e)
    M = 0.01*np.tanh(M)
    a = 1e-6 + 0.07e-3*np.exp(a)
    c = 1e-8 + 1e-8*np.exp(c)
    
    m0 = 3.09692
    mu = 105.6583755e-3
    
    m2 = m0**2 - 2.*mu**2
    
    dataset = dataset.astype(np.float64)
    
    eta1 = dataset[...,0]
    pt1 = dataset[...,1]
    phi1 = dataset[...,2]
    eta2 = dataset[...,3]
    pt2 = dataset[...,4]
    phi2 = dataset[...,5]
    ieta1 = dataset[...,6].astype(np.int32)
    ieta2 = dataset[...,7].astype(np.int32)
    ##ieta1 = np.ones_like(pt1).astype(np.int32)
    #ieta2 = np.ones_like(pt1).astype(np.int32)
    #ieta1 = np.ones(shape = pt1.shape, dtype=np.int32)
    #ieta2 = np.ones(shape = pt1.shape, dtype=np.int32)
    #mass = dataset[...,8]
    
    px1 = pt1*np.cos(phi1)
    py1 = pt1*np.sin(phi1)
    pz1 = pt1*np.sinh(eta1)
    p1 = pt1*np.cosh(eta1)
    e1 = np.sqrt(p1**2 + mu**2)

    px2 = pt2*np.cos(phi2)
    py2 = pt2*np.sin(phi2)
    pz2 = pt2*np.sinh(eta2)
    p2 = pt2*np.cosh(eta2)
    e2 = np.sqrt(p2**2 + mu**2)
    
    p1p2 = (px1*px2 + py1*py2 + pz1*pz2)
    
    cosalpha = p1p2/p1/p2
    
    mass = np.sqrt(2.*(-p1p2 + mu**2 + e1*e2))
    #lr = np.log(pt1) - np.log(pt2)
    r = p1/p2

    #m0 = mass
    
    #p2alt = np.sqrt((m0**2 - 2.*mu**2)/(2.*r*(1.-cosalpha)))
    p2alt = np.sqrt((m0**2 - 2.*mu**2 - mu**2/r - r*mu**2)/(2.*r*(1.-cosalpha)))
    p1alt = r*p2alt
    
    pt1alt = p1alt/np.cosh(eta1)
    pt2alt = p2alt/np.cosh(eta2)
    
    #ieta1 = np.digitize(eta1,etas)
    #ieta2 = np.digitize(eta2,etas)
    
    A1 = A[ieta1]
    e1 = e[ieta1]
    M1 = M[ieta1]
    a1 = a[ieta1]
    c1 = c[ieta1]

    A2 = A[ieta2]
    e2 = e[ieta2]
    M2 = M[ieta2]
    a2 = a[ieta2]
    c2 = c[ieta2]
    
    
    #A1 = np.polyval(A,eta1)
    print("A1.shape", A1.shape)
    #e1 = np.polyval(e,eta1)
    #M1 = np.polyval(M,eta1)
    #a1 = np.polyval(a,eta1)
    #c1 = np.polyval(c,eta1)

    #A2 = np.polyval(A,eta2)
    #e2 = np.polyval(e,eta2)
    #M2 = np.polyval(M,eta2)
    #a2 = np.polyval(a,eta2)
    #c2 = np.polyval(c,eta2)
    
    #A1 = np.sum(A[np.newaxis,...]*eta1[...,np.newaxis],axis=-1)
    #e1 = np.sum(e[np.newaxis,...]*eta1[...,np.newaxis],axis=-1)
    #M1 = np.sum(M[np.newaxis,...]*eta1[...,np.newaxis],axis=-1)
    #a1 = np.sum(a[np.newaxis,...]*eta1[...,np.newaxis],axis=-1)
    #c1 = np.sum(c[np.newaxis,...]*eta1[...,np.newaxis],axis=-1)

    #A2 = np.sum(A[np.newaxis,...]*eta2[...,np.newaxis],axis=-1)
    #e2 = np.sum(e[np.newaxis,...]*eta2[...,np.newaxis],axis=-1)
    #M2 = np.sum(M[np.newaxis,...]*eta2[...,np.newaxis],axis=-1)
    #a2 = np.sum(a[np.newaxis,...]*eta2[...,np.newaxis],axis=-1)
    #c2 = np.sum(c[np.newaxis,...]*eta2[...,np.newaxis],axis=-1)
    
    #A1 = np.zeros_like(eta1)
    #e1 = np.zeros_like(eta1)
    #M1 = np.zeros_like(eta1)
    #a1 = np.ones_like(eta1)
    #c1 = np.zeros_like(eta1)
    
    #A2 = np.zeros_like(eta1)
    #e2 = np.zeros_like(eta1)
    #M2 = np.zeros_like(eta1)
    #a2 = np.ones_like(eta1)
    #c2 = np.zeros_like(eta1)
    
    scale1 = scale(A1,e1,M1, pt1alt, 1)
    scale2 = scale(A2,e2,M2, pt2alt, -1)
    scalem = np.sqrt(scale1*scale2)
    
    ressq1 = sigmasq(a1,c1,pt1alt,eta1)
    ressq2 = sigmasq(a2,c2,pt2alt,eta2)
    resm = np.sqrt(ressq1+ressq2)
    
    pdf = sigpdf(mass, scalem, resm)
    
    print("pdf.shape", pdf.shape)
    
    nll = -np.sum(np.log(pdf))
    #return nll
    #diff1 = (pt1alt-pt1)/pt1
    #diff2 = (pt2alt-pt2)/pt2
    
    #print(np.mean(diff1), np.std(diff1), np.mean(diff2), np.std(diff2))
    
    return nll
    
    #massalt = np.sqrt(2.*(-p1p2 + mu**2 + p1*p2))
    #massalt = np.sqrt(2.*p1*p2*(1.-cosalpha))
   
    #diff = massalt-mass
   
    #return np.mean(diff1), np.std(diff1), np.mean(diff2), np.std(diff2)
   
    #return mass
   

   
#h = jax.hessian(nll)

fg = jax.hessian(nll)
#fg = jax.value_and_grad(nll)
#fg = jax.grad(nll)
#fg = jax.jacrev(nll)
#fg = lambda *args: nll(*args), jax.jacfwd(*args)

#def fg(*args):
    #return nll(*args), jax.jacrev(nll)(*args)

h = jax.hessian(nll)
#h = jax.jacfwd(jax.jacrev(nll))

#h = jax.jit(h)
    
#def fun(x, dataset, *args):
    #def _fun(dataset):
        #return h(x, dataset,*args)
    #return batch_accumulate(_fun, batch_size=8192)(dataset)

def batch_fun(f, batch_size=int(16384)):
    def _fun(x, dataset, *args):
        def _fun2(dataset):
            return f(x, dataset,*args)
        #return batch_accumulate(_fun2, batch_size=int(1e6))(dataset)
        return lbatch_accumulate(_fun2, batch_size=batch_size)(dataset)
        #return lpbatch_accumulate(_fun2, batch_size=int(1e5), ncpu=8)(dataset)
        #return pbatch_accumulate(_fun2, batch_size=batch_size, ncpu=32)(dataset)
        #return pbatch_accumulate_simple(_fun2, batch_size=batch_size, ncpu=32)(dataset)
    return _fun


#fg = jax.jit(batch_fun(fg))
#h = jax.jit(batch_fun(h))
#fg = batch_fun(fg)
#h = batch_fun(h)

#fg = jax.jit(fg)

fg = jax.jit(fg)
fgsimple = fg
fg = batch_fun(fg,batch_size=int(50e3))
#fg = jax.jit(fg)

#fg = jax.jit(fg)
h = jax.jit(h)
#h = batch_fun(h,batch_size=1024)

#fun = jax.jit(fun)

#fg = jax.jit(jax.value_and_grad(fun))
#h = jax.jit(jax.hessian(fun))

#f = functools.partial

infile = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muonTree.root"
#infile = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata/muonTreeData.root"

d = makeData(infile, isData=False)

nEtaBins = 48
etas = np.linspace(-2.4, 2.4, nEtaBins+1, dtype='float64')
#x = (0.,0.,0.,1.,0.)

A = np.zeros(shape=(nEtaBins,), dtype=np.float64)
e = np.zeros(shape=(nEtaBins,), dtype=np.float64)
M = np.zeros(shape=(nEtaBins,), dtype=np.float64)
a = np.zeros(shape=(nEtaBins,), dtype=np.float64)
c = np.zeros(shape=(nEtaBins,), dtype=np.float64)

x = (A,e,M,a,c)

x = np.zeros(shape=(5,nEtaBins), dtype=np.float64)


print(type(d))
print(type(d["eta1"]))
#print(d)
ieta1 = onp.digitize(d["eta1"],etas) - 1
ieta2 = onp.digitize(d["eta2"],etas) - 1

assert(np.min(ieta1)==0)
assert(np.min(ieta2)==0)
assert(np.max(ieta1)==(nEtaBins-1))
assert(np.max(ieta2)==(nEtaBins-1))

#print(etas)
#print(np.min(ieta1), np.max(ieta1))
#print(ieta1)
#assert(0)
dataset = np.stack( (d["eta1"], d["pt1"], d["phi1"],d["eta2"], d["pt2"], d["phi2"],ieta1,ieta2), axis=-1)
d = None
print("dataset.shape",dataset.shape)
#dataset = dataset[:int(100e3)]
for i in range(1000):
    print(i)
    valgrad = fg(x, dataset,etas)
    
    #print(val)
    #print(grad[0][0])
    ##valgrad[0].block_until_ready()
    ##valgrad[1].block_until_ready()
    #valgrad.block_until_ready()
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), valgrad)

#assert(0)

#for j in range(10000):
    #ncpu = 32
    #batch_size = 16384
    #for i in range(ncpu):
                #length = args_flat[0].shape[0]
        
        #nbatchfull = ncpu
        #batch_size = length//nbatchfull
        #batched_length = nbatchfull*batch_size
        #remainder = length%batch_size
        
        #idxstart = np.arange(0,length,length//n)

##fgtest = jax.value_and_grad(nll)
##fgtest = jax.jit(fgtest)
#outvals = []
#for j in range(10000):
    #print("test run j", j)
    #for i in range(32):
        ##data = jax.device_put(dataset[:int(32*32768)],device=jax.devices()[i])
        #data = dataset[:int(1e5)]
        #ftest = jax.jit(fg,device=jax.devices()[i])
        #outval = ftest(x, data,etas)
        #outvals.append(outval)
#for outval in outvals:
    #jax.tree_util.tree_map(lambda x: x.block_until_ready(), outval)
        #outval[0].block_until_ready()
        #outval[1].block_until_ready()

##print(nll(x, dataset[:int(50e3)],etas))

#assert(0)


#dataset = dataset[:int(50e3)]




x = pmin(fgsimple, x, (dataset[:int(100e3)],etas), doParallel=False, jac=True, h=None)
x = pmin(fg, x, (dataset,etas), doParallel=False, jac=True, h=None)

#nll(x, dataset[:500000], etas)
#assert(0)


#print("starting computation")
#print(fun(x,dataset,etas))
#print("starting computation")
#for i in range(10):
    #print(fun(x, dataset, etas))
#print(type(dataset))
#print(dataset.shape)
#print(dataset.dtype)
