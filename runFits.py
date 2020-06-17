import os
import multiprocessing

#ncpu = multiprocessing.cpu_count()

#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8" 

#os.environ["OMP_NUM_THREADS"] = str(ncpu)
#os.environ["OPENBLAS_NUM_THREADS"] = str(ncpu)
#os.environ["MKL_NUM_THREADS"] = str(ncpu)
#os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncpu)
#os.environ["NUMEXPR_NUM_THREADS"] = str(ncpu)

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
#os.environ["XLA_FLAGS"]="--xla_hlo_profile"

#os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=32"

#os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import jax
import jax.numpy as np
import numpy as onp
from jax import grad, hessian, jacobian, config
from jax.scipy.special import erf
config.update('jax_enable_x64', True)

import ROOT
import pickle
from termcolor import colored
from scipy.optimize import minimize, SR1, LinearConstraint, check_grad, approx_fprime
from scipy.optimize import Bounds
import itertools
from root_numpy import array2hist, fill_hist

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from fittingFunctionsBinned import defineStatePars, nllPars, defineState, nll, defineStateParsSigma, nllParsSigma, plots, plotsPars, plotsParsBkg, scaleFromModelPars, splitTransformPars, nllBinsFromBinPars, chi2LBins, scaleSqSigmaSqFromBinsPars,scaleSqFromModelPars,sigmaSqFromModelPars,modelParsFromParVector,scaleSigmaFromModelParVector, plotsBkg, bkgModelFromBinPars
from obsminimization import pmin,batch_vmap,jacvlowmemb
import argparse
import functools
import time
import sys

##slower but lower memory usage calculation of hessian which
##explicitly loops over hessian rows
#def hessianlowmem(fun):
    #def _hessianlowmem(x, f):
        #_, hvp = jax.linearize(jax.grad(f), x)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        #basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    #return functools.partial(_hessianlowmem, f=fun)

#compromise version which vectorizes the calculation, but only partly to save memory
def hessianoptsplit(fun, vsize=4):
    def _hessianopt(x, f):
        _, hvp = jax.linearize(jax.grad(f), x)
        hvp = jax.jit(hvp)
        n = np.prod(x.shape)
        idxs =  np.arange(vsize, n, vsize)
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        splitbasis = np.split(basis,idxs)
        vhvp = jax.vmap(hvp)
        vhvp = jax.jit(vhvp)
        return np.concatenate([vhvp(b) for b in splitbasis]).reshape(x.shape + x.shape)
    return functools.partial(_hessianopt, f=fun)

#optimized version which is faster than the built-in hessian for some reason
# **TODO** follow up with jax authors to understand why
def hessianopt(fun):
    def _hessianopt(x, f):
        _, hvp = jax.linearize(jax.grad(f), x)
        hvp = jax.jit(hvp)
        vhvp = jax.vmap(hvp)
        vhvp = jax.jit(vhvp)
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return vhvp(basis).reshape(x.shape + x.shape)
    return functools.partial(_hessianopt, f=fun)


#def vgrad(fun):
    #g = jax.grad(fun)
    #return jax.vmap(g)

def hvp(fun):
    def _hvp(x, v, f):
        return jax.jvp(jax.grad(f), (x,), (v,))[1]
    return functools.partial(_hvp, f=fun)

class CachingHVP():
    def __init__(self,fun):
        self.grad = jax.jit(jax.grad(fun))
        self.x = None
        self.flin = None
        
    def hvp(self,x,v):
        if self.x is None or not np.equal(x,self.x).all():
            _,flin = jax.linearize(self.grad,x)
            self.flin = jax.jit(flin)
            #self.flin = flin
            self.x = x
        return self.flin(v)
    
#def blockfgrad(fun):
    #g = jax.grad(fun)
    #return jax.jit(jax.vmap(g))
    
class CachingBlockGrads():
    def __init__(self,fun,nblocks,static_argnums=None,):
        hess = jax.hessian(fun)
        vhess = jax.vmap(hess)
        self._vhess = jax.jit(vhess,static_argnums=static_argnums)
        self._vmatmul = jax.jit(jax.vmap(np.matmul))
        self._fgrad = jax.jit(jax.vmap(jax.value_and_grad(fun)),static_argnums=static_argnums)
        self.vhessres = None
        self.x = None
        self.nblocks = nblocks
        
    def hvp(self,x,v, *args):
        if self.x is None or not np.equal(x,self.x).all():
            self.vhessres = self._vhess(x.reshape((self.nblocks,-1)),*args)
            self.x = x
        return self._vmatmul(self.vhessres,v.reshape((self.nblocks,-1))).reshape((-1,))
    
    def fgrad(self, x, *args):
        f,g = self._fgrad(x.reshape((self.nblocks,-1)), *args)
        f = np.sum(f, axis=0)
        g = g.reshape((-1,))
        return f,g
   
#wrapper to handle printing which otherwise doesn't work properly from bfgs apparently
class NLLHandler():
    def __init__(self, fun, fundebug = None):
        self.fun = fun
        self.fundebug = fundebug
        self.iiter = 0
        self.f = 0.
        self.grad = np.array(0.)
        
    def wrapper(self, x, *args):
        f,grad = self.fun(x,*args)
        if np.isnan(f) or np.any(np.isnan(grad)):
            print("nan detected")
            print(x)
            print(f)
            print(grad)
            
            if self.fundebug is not None:
                self.fundebug(x)
                
            assert(0)    
        self.f = f
        self.grad = grad
        return f,grad
    
    def callback(self,x):
        print(self.iiter, self.f, np.linalg.norm(self.grad))
        self.iiter += 1


    
    

parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-runCalibration', '--runCalibration', default=False, action='store_true', help='Use to fit corrections, omit to fit scale parameter')
parser.add_argument('-fitResolution', '--fitResolution', default=False, action='store_true', help='Use to fit resolution paramter, omit to fit sigma')


args = parser.parse_args()
isJ = args.isJ
runCalibration = args.runCalibration
fitResolution = args.fitResolution

#file = open("calInput{}MC_48etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
file = open("calInput{}DATA_48etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
pkg = pickle.load(file)

dataset = pkg['dataset']
etas = pkg['edges'][0]
pts = pkg['edges'][2]
masses = pkg['edges'][-1]
binCenters1 = pkg['binCenters1']
binCenters2 = pkg['binCenters2']
good_idx = pkg['good_idx']

#print("good_idx shape", good_idx[0].shape)
#print("dataset shape", dataset.shape)

filegen = open("calInput{}MCgen_48etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
datasetgen = pickle.load(filegen)

nEtaBins = len(etas)-1
nPtBins = len(pts)-1
nBins = dataset.shape[0]

#print(pts)


scale = np.ones((nBins,),dtype='float64')
sigma = 6e-3*np.ones((nBins,),dtype='float64')
fbkg = 0.05*np.ones((nBins,),dtype='float64')
slope = 0.02*np.ones((nBins,),dtype='float64')

xscale = np.stack([scale,sigma,fbkg,slope],axis=-1)

xscale = np.zeros_like(xscale)

nllBinspartial = functools.partial(nllBinsFromBinPars, masses=masses)

minbin = 0
#maxbin = 10
maxbin = int(5000)
#maxbins = int(10e3)
#pmin(nllBinspartial, xscale[minbin:maxbin], args=(dataset[minbin:maxbin],datasetgen[minbin:maxbin]))

#np.set_printoptions(threshold=sys.maxsize)
#for i in range(4):
    #print(good_idx[i])
#assert(0)
#print("first minimization")


xres = xscale
#parallel fit for scale, sigma, fbkg, slope in bins
xres = pmin(nllBinspartial, xscale, args=(dataset,datasetgen))
#xres = xscale

#compute covariance matrices of scale and sigma from binned fit
def hnll(x,dataset,datasetgen):
    #compute the hessian wrt internal fit parameters in each bin
    hess = jax.hessian(nllBinspartial)
    #hess = jax.hessianlowmem(nllBinspartial)
    #invert to get the hessian
    cov = np.linalg.inv(hess(x,dataset,datasetgen))
    #compute the jacobian for scale and sigma squared wrt internal fit parameters
    jacscalesq, jacsigmasq = jax.jacfwd(scaleSqSigmaSqFromBinsPars)(x)
    jac = np.stack((jacscalesq,jacsigmasq),axis=-1)
    #compute covariance matrix for scalesq and sigmasq
    covscalesigmasq = np.matmul(jac.T,np.matmul(cov,jac))
    #invert again to get the hessian
    hscalesigmasq = np.linalg.inv(covscalesigmasq)
    return hscalesigmasq, covscalesigmasq
#fh = jax.jit(jax.vmap(hnll))
fh = jax.jit(batch_vmap(hnll, batch_size=256))
#fh = beval(fh, batch_size=256, accumulator = lambda x: np.concatenate(x,axis=0))

#print("run fh")
        
    #vinverse = jax.vmap
#assert(0)
hScaleSqSigmaSqBinned, hCovScaleSqSigmaSqBinned = fh(xres,dataset,datasetgen)

fbkg, slope = bkgModelFromBinPars(xres)


scaleSqBinned, sigmaSqBinned = scaleSqSigmaSqFromBinsPars(xres)

scaleBinned = np.sqrt(scaleSqBinned)
sigmaBinned = np.sqrt(sigmaSqBinned)

scaleSqSigmaSqErrorsBinned = np.sqrt(np.diagonal(hCovScaleSqSigmaSqBinned, axis1=-1, axis2=-2))

scaleSqErrorBinned = scaleSqSigmaSqErrorsBinned[:,0]
sigmaSqErrorBinned = scaleSqSigmaSqErrorsBinned[:,1]

scaleErrorBinned = 0.5*scaleSqErrorBinned/scaleBinned
sigmaErrorBinned = 0.5*sigmaSqErrorBinned/sigmaBinned



#scaleSigmaErrorBinned = 0.5*

#np.set_printoptions(threshold=sys.maxsize)

#print(binScale)
#print(binSigma)
#assert(0)

#print(covScaleSqSigmaSq)
#print(covScaleSqSigmaSq.shape)

#assert(0)

nModelParms = 7

#xmodel = np.zeros((nEtaBins,nModelParms),dtype=np.float64)

#A = np.zeros((nEtaBins),dtype=np.float64)
#e = np.zeros((nEtaBins),dtype=np.float64)
#M = np.zeros((nEtaBins),dtype=np.float64)
#a = 1e-2**2*np.ones((nEtaBins),dtype=np.float64)

#xmodel = np.stack((A,e,M,a),axis=-1)

#xmodel = np.zeros((nEtaBins,nModelParms),dtype=np.float64)

parmones = np.ones((nEtaBins),dtype=np.float64)

A = 0.*parmones
e = 0.*parmones
M = 0.*parmones
W = 0.*parmones
#a = 1e-3
#b = 1e-3
#c = 1e-5
#d = 370.
a = 1.*parmones
b = 1.*parmones
c = 1.*parmones
d = 1.*parmones


#xmodel = np.array((A,e,M,a,c,b,d),dtype=np.float64)
xmodel = np.stack((A,e,M,a,c,b,d),axis=-1)


chi2 = chi2LBins(xmodel, scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas, binCenters1, binCenters2, good_idx)


#print(chi2)

#assert(0)


#chi2 = jax.jit(chi2SumBins)(xmodel, binScaleSq, binSigmaSq, covScaleSqSigmaSq, etas, binCenters1, binCenters2, good_idx)

#

#print(chi2)

print("second minimization")

#chi2LBins
xmodel = pmin(chi2LBins, xmodel.flatten(), args=(scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas, binCenters1, binCenters2, good_idx), doParallel=False)

xmodel = xmodel.reshape((-1,nModelParms))


fgchi2 = jax.jit(jax.value_and_grad(chi2LBins))
hchi2 = jax.jit(jax.hessian(chi2LBins))

print("chi2 hess/grad")

chi2,chi2grad = fgchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas, binCenters1, binCenters2, good_idx)

chi2hess = hchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas, binCenters1, binCenters2, good_idx)

#xmodel = -np.linalg.solve(chi2hess,chi2grad)

hmodel = chi2hess
covmodel = np.linalg.inv(chi2hess)
invhess = covmodel

valmodel,gradmodel = fgchi2(xmodel.flatten(), scaleSqBinned, sigmaSqBinned, hScaleSqSigmaSqBinned, etas, binCenters1, binCenters2, good_idx)
ndof = 2*nBins - nEtaBins*nModelParms

edm = 0.5*np.matmul(np.matmul(gradmodel.T,covmodel),gradmodel)

chi2val = 2.*valmodel

print("nEtaBins", nEtaBins)
print("nBins", nBins)
print("chi2/dof = %f/%d = %f" % (chi2val,ndof,chi2val/float(ndof)))

#chi2grad = chi2grad.reshape((nEtaBins,nModelParms))

#print(chi2grad)
#print(etas)
#assert(0)

#gtest = scaleSqFromModelPars(A,e,M,etas, binCenters1, binCenters2, good_idx, linearize=False)
#gtest = jax.jacrev(scaleSqFromModelPars)

#gtestval = gtest(A,e,M,etas, binCenters1, binCenters2, good_idx, linearize=False)
#print(gtestval.shape)
#print(gtestval)

#print(np.max(gtestval[:,1]), np.min(gtestval[:,1]))

#assert(0)

#hmodel = hchi2(res.x, binScaleSq, binSigmaSq, hScaleSqSigmaSq, etas, binCenters1, binCenters2, good_idx)
#covmodel = np.linalg.inv(hchi2(res.x, binScaleSq, binSigmaSq, hScaleSqSigmaSq, etas, binCenters1, binCenters2, good_idx))
errsmodel = np.sqrt(np.diag(covmodel)).reshape((nEtaBins,nModelParms))

#np.set_printoptions(threshold=sys.maxsize)
#xmodel = np.reshape(res.x,(nEtaBins,nModelParms))
#xmodel = np.reshape(res.x,(nEtaBins,nModelParms))

    
#A = xmodel[...,0]
#e = xmodel[...,1]
#M = xmodel[...,2]
#a = xmodel[...,3]
##c = x[...,4]
##b = x[...,5]

#c = np.zeros_like(a)
#b = np.zeros_like(a)
#d = 370.*np.ones_like(a)

A,e,M,a,b,c,d = modelParsFromParVector(xmodel)

scaleSqModel = scaleSqFromModelPars(A, e, M, etas, binCenters1, binCenters2, good_idx)
sigmaSqModel = sigmaSqFromModelPars(a, b, c, d, etas, binCenters1, binCenters2, good_idx)

scaleModel = np.sqrt(scaleSqModel)
sigmaModel = np.sqrt(sigmaSqModel)

#print(binScaleSq)
#print(binSigmaSq)
#print(scalesqmodel)
#print(sigmasqmodel)
#assert(0)

#print(np.linalg.eigvalsh(hmodel))

##print(hmodel)
#print(covmodel)

#for i in range(nModelParms):
    #print(i)
    #print(xmodel[:,i])
    #print(errsmodel[:,i])

#assert(0)

print(xmodel, "+/-", errsmodel)
print(edm, "edm")

#errs = np.sqrt(np.diag(invhess))
#erridx = np.where(np.isnan(errs))
#print(erridx)
#binidx = (erridx[0] - nEtaBins-nBins,)
#print(binidx)
#for gid in good_idx:
    #print(gid[binidx])



diag = np.diag(np.sqrt(np.diag(invhess)))
diag = np.linalg.inv(diag)
corr = np.dot(diag,invhess).dot(diag)

#only for model parameters
#corr = corr[:4*nEtaBins,:4*nEtaBins]

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.clf()
plt.pcolor(corr, cmap='jet', vmin=-1, vmax=1)
plt.colorbar()
plt.savefig("corrMC.pdf")



#plotsPars(res.x,nEtaBins,nPtBins,dataset,datasetgen,masses,isJ,etas, binCenters1, binCenters2, good_idx)

#plotsParsBkg(res.x,nEtaBins,nPtBins,dataset,datasetgen,masses,isJ,etas, binCenters1, binCenters2, good_idx)

plotsBkg(scaleBinned,sigmaBinned,fbkg,slope,dataset,datasetgen,masses,isJ,etas, binCenters1, binCenters2, good_idx)

print("computing scales and errors:")

ndata = np.sum(dataset,axis=-1)



Aerr = errsmodel[:,0]
eerr = errsmodel[:,1]
Merr = errsmodel[:,2]
aerr = errsmodel[:,3]
cerr = errsmodel[:,4]

etaarr = onp.array(etas.tolist())
hA = ROOT.TH1D("A", "A", nEtaBins, etaarr)
he = ROOT.TH1D("e", "e", nEtaBins, etaarr)
hM = ROOT.TH1D("M", "M", nEtaBins, etaarr)
ha = ROOT.TH1D("a", "a", nEtaBins, etaarr)
hc = ROOT.TH1D("c", "c", nEtaBins, etaarr)

hA = array2hist(A, hA, Aerr)
he = array2hist(e, he, eerr)
hM = array2hist(M, hM, Merr)
ha = array2hist(a, ha, aerr)
hc = array2hist(c, hc, cerr)

hA.GetYaxis().SetTitle('b field correction')
he.GetYaxis().SetTitle('material correction')
hM.GetYaxis().SetTitle('alignment correction')
ha.GetYaxis().SetTitle('material correction (resolution) a^2')
hc.GetYaxis().SetTitle('hit position (resolution) c^2')

hA.GetXaxis().SetTitle('#eta')
he.GetXaxis().SetTitle('#eta')
hM.GetXaxis().SetTitle('#eta')
ha.GetXaxis().SetTitle('#eta')
hc.GetXaxis().SetTitle('#eta')

#assert(0)
print("evaluate scale sigma jac from model")
#scalejac,sigmajac = jax.jit(jax.jacfwd(scaleSigmaFromModelParVector))(xmodel.flatten(),etas, binCenters1, binCenters2, good_idx)
fjac = jax.jit(jax.jacfwd(scaleSigmaFromModelParVector))
#fjac = jax.jit(jacvlowmemb(scaleSigmaFromModelParVector, batch_size=1))
scalejac,sigmajac = fjac(xmodel.flatten(),etas, binCenters1, binCenters2, good_idx)
#print(scalejac,sigmajac)
#assert(0)
scalesigmajac = np.stack((scalejac,sigmajac),axis=1)
scalesigmajac = np.reshape(scalesigmajac, (-1,covmodel.shape[0]))
#print(scalesigmajac)

#compute error one element at a time to avoid computing full 24k^2 covariance matrix
def scalesigmaerr(scalesigmajac):
    jcol = np.expand_dims(scalesigmajac,axis=-1)
    res = np.matmul(jcol.T, np.matmul(covmodel, jcol))
    err = np.sqrt(np.squeeze(res,axis=-1))
    return err
    
scaleSigmaErrsModel = jax.jit(batch_vmap(scalesigmaerr, batch_size=256))(scalesigmajac)

#covScaleSigmaModel = np.matmul(scalesigmajac,np.matmul(covmodel,scalesigmajac.T))
#print(covScaleSigmaModel)
#print(scalesigmajac.shape)
#print(covScaleSigmaModel.shape)
#assert(0)
#scaleSigmaErrsModel = np.sqrt(np.diag(covScaleSigmaModel))
scaleSigmaErrsModel = np.reshape(scaleSigmaErrsModel, (-1,2))

#assert(0)

#have to use original numpy to construct the bin edges because for some reason this doesn't work with the arrays returned by jax
scaleplotBinned = ROOT.TH1D("scaleBinned", "scale", nBins, onp.linspace(0, nBins, nBins+1))
scaleplotBinned = array2hist(scaleBinned, scaleplotBinned, scaleErrorBinned)

print(scaleModel.shape, scaleSigmaErrsModel[:,0].shape)
scaleplotModel = ROOT.TH1D("scaleModel", "scale", nBins, onp.linspace(0, nBins, nBins+1))
scaleplotModel = array2hist(scaleModel, scaleplotModel, scaleSigmaErrsModel[:,0])

sigmaplotBinned = ROOT.TH1D("sigmaBinned", "sigma", nBins, onp.linspace(0, nBins, nBins+1))
sigmaplotBinned = array2hist(sigmaBinned, sigmaplotBinned, sigmaErrorBinned)

sigmaplotModel = ROOT.TH1D("sigmaModel", "sigma", nBins, onp.linspace(0, nBins, nBins+1))
sigmaplotModel = array2hist(sigmaModel, sigmaplotModel, scaleSigmaErrsModel[:,1])

plots = [scaleplotBinned,scaleplotModel, sigmaplotBinned, sigmaplotModel]

for ibin in range(nBins):
    ieta1 = good_idx[0][ibin]
    ieta2 = good_idx[1][ibin]
    ipt1 = good_idx[2][ibin]
    ipt2 = good_idx[3][ibin]
    for plot in plots:
        plot.GetXaxis().SetBinLabel(ibin+1,'eta1_{}_eta2_{}_pt1_{}_pt2_{}'.format(ieta1,ieta2,ipt1,ipt2))

for plot in plots:
    plot.GetXaxis().LabelsOption("v")

f = ROOT.TFile("calibrationMC.root", 'recreate')
f.cd()

hA.Write()
he.Write()
hM.Write()
ha.Write()
hc.Write()
for plot in plots:
    plot.Write()

