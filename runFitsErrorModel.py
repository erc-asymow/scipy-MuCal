import os
import multiprocessing

ncpu = multiprocessing.cpu_count()

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32" 

os.environ["OMP_NUM_THREADS"] = str(ncpu)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncpu)
os.environ["MKL_NUM_THREADS"] = str(ncpu)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncpu)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncpu)

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


def makeData(inputFile, idx, ptmin):

    RDF = ROOT.ROOT.RDataFrame
    
    if idx==1:
        q = 1.0
    elif idx==2:
        q = -1.0
        

    d = RDF('tree',inputFile)
    
    cut = f"pt{idx} > {ptmin} && fabs(eta{idx})<2.4"
    
    #cut += f" && eta{idx}>2.3"
    
    #print(cut)

    d = d.Filter(cut)

    #d = d.Define('krec', f'1./pt{idx}/(1.+2.*cErr{idx}*cErr{idx})')
    #d = d.Define('kgen', f'1./mcpt{idx}')

    d = d.Define('krec', f'1./pt{idx}')
    d = d.Define('kgen', f'1./mcpt{idx}')
    d = d.Define('kerr', f'cErr{idx}')
    
    #d = d.Define('krec', f'pt{idx}')
    #d = d.Define('kgen', f'mcpt{idx}')
    
    d = d.Define('eta', f'eta{idx}')
    d = d.Define('q', f'{q}')
    #d = d.Filter("krec/kgen>0.5 && krec/kgen<2.0")

    
    cols=['eta','q','kgen','krec', 'kerr']
    
    data = d.AsNumpy(columns=cols)

    return data

def scale(A,e,M,W,a,b,c,d,k,q,eta):
    sintheta = np.sin(2*np.arctan(np.exp(-eta)))
    l = computeTrackLength(eta)
    
    #delta = A - e*sintheta*k + q*M/k + W*l**4/k**2
    #delta = A - e*sintheta*k + q*M/k + W/k**2
    g = b/c + d
    
    delta = A - e*sintheta*k + q*M/k - W*l**4/k**2*(1.+g*k**2/l**2)/(1.+d*k**2/l**2)
    return 1.-delta
    #return 1.+delta

def sigmasq(a,b,c,d,k,eta):
    l = computeTrackLength(eta)
    #res1 = a1*np.power(l1,2) + c1*np.power(l1,4)*np.power(p1,2) + b1*np.power(l1,2)/(1+d1/(np.power(p1,2)*np.power(l1,2)))
    #return a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    #return a*l**2 + c*l**4/k**2*(1.+b*k**2/l**2)/(1.+d*k**2/l**2)
    
    res = a + c/k**2 + b/(1.+d*k**2)
    
    #res = a*l**2 + c*l**4/k**2 + b*l**2/(1.+d*k**2/l**2)
    #res = a*l**2 + c*l**4/k**2*(1.+b*k**2/l**2)/(1.+d*k**2/l**2)
    
    return res

def logsigpdf(vals, mu, sigma):

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

def loggauspdf(vals, mu, sigma):
    
    t = (vals - mu)/sigma
    
    pdf = -0.5*t**2
    logI = np.log(sigma) + 0.5*np.log(2.*np.pi)
    
    return pdf - logI

#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1e2])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1.])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1.])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-6, 1e-3, 1e-3, 1e-7,1e2])
#parmscale = np.array([1e-4, 1e-3, 1e-5, 1e-3, 1e-3, 1e-7,1.])

#parmscale = np.array([1e-3, 1e-3, 1e-7,1.])


#parmscale = np.array([1e-3, 1e-3, 1e-7,1.,1.])
#parmscale = np.array([1e-3, 1e-3, 1e-7,1.,1e-5])
parmscale = np.array([1e-3, 1e-3, 1e-7,1.,1e-2])
#parmscale = np.array([1e-3, 1e-3, 1e-7,1.,1e-4])


#parmscale = np.array([1e-3, 1., 1e-7,1.])

#parmscale = np.array([1e-3, 1e-3, 1e-7])
#parmscale = np.array([1e-3, 1e-3])
#parmscale = np.array([1e-3, 1e-3, 1e-3])

def nll(parms,dataset,eta):
    parms = parms*parmscale
    
    #eta = dataset[...,0]
    #q = dataset[...,1]
    #kgen = dataset[...,2]
    krec = dataset[...,1]
    kerr = dataset[...,2]
    #ieta = dataset[...,4]
    
    a = parms[0]
    b = parms[1]
    c = parms[2]
    d = parms[3]
    s = parms[4]
    
    #c = parms[1]
    #e = parms[2]
    
    #b = np.zeros_like(a)
    #d = 370.*np.ones_like(a)
    
    #scalem = scale(A,e,M,W, kgen, q, eta)
    resm = sigmasq(a,b,c,d,krec,eta)
    #resm = np.sqrt(resm)
    vals = kerr**2
    #vals = kerr
    
        
    pdf = logsigpdf(vals, resm, s*resm)
    #pdf = loggauspdf(vals/resm, 1., 1e-2)
    #pdf = loggauspdf(krec, kgen, scalem, resm)
    nll = -np.sum(pdf)
    
    #nll = np.where(np.any(sigma<=0.,), np.nan, nll)
    
    return nll


#def gfunc(b,c,d):
    #return b/c + d

#jacgfunc = jax.jacrev(gfunc)

def batch_fun(f, batch_size=int(16384)):
    def _fun(x, dataset, *args):
        def _fun2(dataset):
            return f(x, dataset, *args)
        #return batch_accumulate(_fun2, batch_size=batch_size)(dataset,ieta)
        return lbatch_accumulate(_fun2, batch_size=batch_size)(dataset)
        #return lpbatch_accumulate(_fun2, batch_size=int(1e5), ncpu=8)(dataset)
        #return pbatch_accumulate(_fun2, batch_size=batch_size, ncpu=32)(dataset,ieta)
        #return pbatch_accumulate_simple(_fun2, batch_size=batch_size, ncpu=32)(dataset)
    return _fun


def parmsg(parms):
    a = parms[0]
    b = parms[1]
    c = parms[2]
    d = parms[3]
    s = parms[4]
    
    g = b/c + d
    
    parms = np.stack((a,g,c,d,s),axis=0)
    return parms
    
jacg = jax.jit(jax.jacfwd(parmsg))
   
dataDir = "/data/bendavid/cmsdocker7/home/cmsusr/muoncaldata"

files = [f"{dataDir}/muonTree.root", f"{dataDir}/muonTreeMCZ.root"]
minpts = [3., 10.]

dsets = []
for f, minpt in zip(files,minpts):
    for idx in [1,2]:
        d = makeData(f,idx,minpt)
        dset = onp.stack( (d["eta"],d["krec"], d['kerr']), axis=-1)
        #dset = np.stack( (d["eta"],d["kgen"], d['kerr']), axis=-1)
        print(dset.dtype)
        #dset = dset[:int(10e3)]
        dsets.append(dset)

dataset = onp.concatenate(dsets,axis=0)
dsets = None
deta = dataset[:,0]

#print(dataset.shape)


#nEtaBins = 48
#etas = onp.linspace(-2.4,2.4, nEtaBins+1)

#nEtaBins = 24
#etas = np.linspace(0.,2.4, nEtaBins+1)

nEtaBins = 1
etas = np.linspace(-2.4,-2.3, nEtaBins+1)


#etas = np.linspace(0.,0.1, nEtaBins+1)
etasL = etas[:-1]
etasH = etas[1:]
etasC = 0.5*(etas[1:]+etas[:-1])

fg = jax.jit(jax.value_and_grad(nll))
#h = jax.hessian(nll)
h = jax.jit(jax.hessian(nll))
#h = batch_fun(h, batch_size=int(50e3))
#h = jax.jit(jax.jacrev(jax.grad(nll)))

forcecpu = False
if forcecpu:
    fg = lpbatch_accumulate(fg, batch_size=int(1e6),ncpu=32, in_axes=(None,0,None))
    h = lpbatch_accumulate(h, batch_size=int(1e5),ncpu=32, in_axes=(None,0,None))
else:
    fg = lbatch_accumulate(fg, batch_size=int(1e6), in_axes=(None,0,None))
    h = lbatch_accumulate(h, batch_size=int(1e5), in_axes=(None,0,None))

xs = []
covs = []
xerrs = []

for etaL, etaH, etaC in zip(etasL, etasH, etasC):
    #dseteta = dataset[np.where(dataset[:,0]>2.3)]
    dseteta = dataset[np.where(np.logical_and(deta>=etaL, deta<etaH))]
    #dseteta = dseteta.astype(np.float64)
    #ieta = onp.digitize(dseteta[:,0],etas)-1
    
    print(etaC)
    
    #A = 0.
    #e = 0.
    #M = 0.
    #W = 0.
    #a = 1e-3
    #b = 1e-3
    #c = 1e-5
    #d = 370.
    a = 1.
    b = 1.
    c = 1.
    d = 100.
    s = 1.
    

    x = np.stack((a,b,c,d,s)).astype(np.float64)
    #x = np.stack((a,b,c,d)).astype(np.float64)
    #x = np.stack((a,b,c)).astype(np.float64)
    #x = np.stack((a,c)).astype(np.float64)

    
    #x = pmin(fg, x, (dseteta,etaC), doParallel=False, jac=True, h=None, lb=lb,ub=ub)
    #x = pmin(fg, x, (dseteta,etaC), doParallel=False, jac=True, h=None)
    x = pmin(fg, x, (dseteta,etaC), doParallel=False, jac=True, h=h)
    
    print("computing hess")
    hess = h(x, dseteta, etaC)
    dseteta = None
    #hess = np.eye(x.shape[0])
    cov = np.linalg.inv(hess)
    #xerr = np.sqrt(np.diag(cov))
    

    
    x = x*parmscale
    

    cov = cov*parmscale[:,np.newaxis]*parmscale[np.newaxis,:]
    
    #convert from abcds to agcds
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
    
nparms = xs.shape[-1]
cov = onp.zeros(shape=(nEtaBins*nparms,nEtaBins*nparms), dtype=xs.dtype)
print("nparms", nparms)
print("xs.shape", xs.shape)
print("cov.shape", cov.shape)
for i, icov in enumerate(covs):
    cov[i*nparms:(i+1)*nparms, i*nparms:(i+1)*nparms] = icov
    
print(cov)

covdiag = np.diag(cov)
cor = cov/np.sqrt(covdiag[:,np.newaxis]*covdiag[np.newaxis,:])
print(cor)


a = xs[:,0]
b = xs[:,1]
c = xs[:,2]
d = xs[:,3]
s = xs[:,4]
#c = xs[:,1]
#c = xs[:,1]

#a = xs[:,3]
#b = xs[:,4]
#c = xs[:,5]
#d = xs[:,6]


aerr = xerrs[:,0]
berr = xerrs[:,1]
cerr = xerrs[:,2]
derr = xerrs[:,3]
serr = xerrs[:,4]

#cerr = xerrs[:,1]

#aerr = xerrs[:,3]
#berr = xerrs[:,4]
#cerr = xerrs[:,5]
#derr = xerrs[:,6]

#b = np.zeros_like(a)
#d = np.zeros_like(a)
#berr = np.zeros_like(aerr)
#derr = np.zeros_like(aerr)


etaarr = onp.array(etas.tolist())
ha = ROOT.TH1D("a", "a", nEtaBins, etaarr)
hc = ROOT.TH1D("c", "c", nEtaBins, etaarr)
hb = ROOT.TH1D("g", "g", nEtaBins, etaarr)
hd = ROOT.TH1D("d", "d", nEtaBins, etaarr)
hs = ROOT.TH1D("s", "s", nEtaBins, etaarr)

ha = array2hist(a, ha, aerr)
hc = array2hist(c, hc, cerr)
hb = array2hist(b, hb, berr)
hd = array2hist(d, hd, derr)
hs = array2hist(s, hs, serr)

ha.GetYaxis().SetTitle('material correction (resolution) a^2')
hc.GetYaxis().SetTitle('hit position (resolution) c^2')

ha.GetXaxis().SetTitle('#eta')
hc.GetXaxis().SetTitle('#eta')

hCov = ROOT.TH2D("cov","cov", nEtaBins*nparms, -0.5, nEtaBins*nparms-0.5, nEtaBins*nparms, -0.5, nEtaBins*nparms-0.5)
hCov = array2hist(cov, hCov)

hCor = ROOT.TH2D("cor","cor", nEtaBins*nparms, -0.5, nEtaBins*nparms-0.5, nEtaBins*nparms, -0.5, nEtaBins*nparms-0.5)
hCor = array2hist(cor, hCor)


f = ROOT.TFile("calibrationMCErrs.root", 'recreate')
f.cd()


ha.Write()
hc.Write()
hb.Write()
hd.Write()
hs.Write()
hCov.Write()
hCor.Write()

assert(0)

#dataset = dataset[np.where(dataset[:,0]>2.3)]

#dataset = onp.array(dataset)
#onp.random.shuffle(dataset)
#dataset = dataset[:int(1e6)]

#jax doesn't implement this, so use original numpy version
ieta = onp.digitize(dataset[:,0],etas)-1

#dataset = np.stack((dataset,(ieta,)),axis=-1)

A = np.full((nEtaBins,),0.,dtype=np.float64)
e = np.full((nEtaBins,),0.,dtype=np.float64)
M = np.full((nEtaBins,),0.,dtype=np.float64)
W = np.full((nEtaBins,),0.,dtype=np.float64)
a = np.full((nEtaBins,),1e-6,dtype=np.float64)
b = np.full((nEtaBins,),0.,dtype=np.float64)
c = np.full((nEtaBins,),0.,dtype=np.float64)
d = np.full((nEtaBins,),370.,dtype=np.float64)

x = np.stack((A,e,M,W,a,b,c,d), axis=-1)
print("x shape")
print(x.shape)

#val = nllpll(x,dataset[:1000],ieta[:1000],etas)
#print(val.shape)
#print(val)

#grad = jax.grad(nllpll)(x,dataset[:1000],ieta[:1000],etas)
#print(grad.shape)
#print(grad)
#assert(0)


#fg = jax.jit(jax.value_and_grad(nllfull))
#h = jax.jit(jax.hessian(nllfull))

#fg = batch_fun(fg, batch_size=int(10e6))
#h = batch_fun(h, batch_size=int(1e6))

#x = pmin(fg, x, (dataset,ieta,etas), doParallel=False, jac=True, h=None)  

#fg = jax.jit(jax.value_and_grad(nllpll))
#h = jax.jit(jax.hessian(nllpll))

#nllpartial = functools.partial(nllpll, dataset=dataset,ieta=ieta, etas=etas)

#fg = batch_fun(fg, batch_size=int(10e6))
#h = batch_fun(h, batch_size=int(1e6))
def nllpllsum(*args):
    return np.sum(nllpll(*args))

g = jax.grad(nllpllsum)
def fg(*args):
    return nllpll(*args), g(*args)
#fg = lambda args: nllpll(*args), g(*args)

fg = jax.jit(fg)




fg = batch_fun(fg, batch_size=int(10e6))
fg = jax.jit(fg)

#val,grad = fg(x,dataset,ieta,etas)
#print(val)
#print(grad)
#assert(0)

#fg = lambda: jax.grad(lambda *args: np.sum(nllpll(*args)))

#x = pmin(nllpartial, x, doParallel=True, jac=False, h=None)  
x = pmin(fg, x, (dataset,ieta,etas), doParallel=True, jac=True, h=None)  

print(x)

#nll(parms,dataset, eta)

#fg = jax.value_and_grad(nll)


#for i in range(1000):
    #val,grad = fg(parms,dataset,-2.35)
    #print(val)
    #print(grad)

#for i in range(1000):
    #hess = h(parms,dataset,-2.35)

    #print(hess)



#print(x)
         
