import os
import multiprocessing

ncpu = multiprocessing.cpu_count()

os.environ["OMP_NUM_THREADS"] = str(ncpu)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncpu)
os.environ["MKL_NUM_THREADS"] = str(ncpu)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncpu)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncpu)

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

from fittingFunctionsBinned import defineStatePars, nllPars, defineState, nll, plots, plotsPars
import argparse
import functools

#slower but lower memory usage calculation of hessian which
#explicitly loops over hessian rows
def hessianlowmem(fun):
    def _hessianlowmem(x, f):
        _, hvp = jax.linearize(jax.grad(f), x)
        hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return functools.partial(_hessianlowmem, f=fun)

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

def scaleFromPars(AeM, etas, binCenters1, binCenters2, good_idx):

    A = AeM[:nEtaBins]
    e = AeM[nEtaBins:2*nEtaBins]
    M = AeM[2*nEtaBins:3*nEtaBins]

    etasC = (etas[:-1] + etas[1:]) / 2.

    sEta = np.sin(2*np.arctan(np.exp(-etasC)))
    s1 = sEta[good_idx[0]]
    s2 = sEta[good_idx[1]]
    
    c1 = binCenters1
    c2 = binCenters2

    # select the model parameters from the eta bins corresponding
    # to each kinematic bin
    A1 = A[good_idx[0]]
    e1 = e[good_idx[0]]
    M1 = M[good_idx[0]]

    A2 = A[good_idx[1]]
    e2 = e[good_idx[1]]
    M2 = M[good_idx[1]]

    term1 = A1-s1*e1*c1+M1/c1
    term2 = A2-s2*e2*c2-M2/c2
    combos = term1*term2
    scale = np.sqrt(combos)

    return scale.flatten()


parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-runCalibration', '--runCalibration', default=False, action='store_true', help='Use to fit corrections, omit to fit scale parameter')

args = parser.parse_args()
isJ = args.isJ
runCalibration = args.runCalibration

file = open("calInput{}MC_4etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
pkg = pickle.load(file)

dataset = pkg['dataset']
etas = pkg['edges'][0]
pts = pkg['edges'][3]
binCenters1 = pkg['binCenters1']
binCenters2 = pkg['binCenters2']
good_idx = pkg['good_idx']

filegen = open("calInput{}MCgen_4etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
datasetgen = pickle.load(filegen)

nEtaBins = len(etas)-1
nPtBins = len(pts)-1
nBins = dataset.shape[0]

print(pts)

if runCalibration:
    x = defineStatePars(nEtaBins,nPtBins, dataset, isJ)
else:
    x = defineState(nEtaBins,nPtBins, dataset)


print("minimising")

xtol = np.finfo('float64').eps
#btol = 1.e-8
btol = 0.1
maxiter = 100000
#maxiter = 87

if runCalibration:

    lb_scale = np.concatenate((0.95*np.ones(nEtaBins),-0.05*np.ones(nEtaBins), -1e-2*np.ones(nEtaBins)),axis=0)
    ub_scale = np.concatenate((1.05*np.ones(nEtaBins),0.05*np.ones(nEtaBins), 1e-2*np.ones(nEtaBins)),axis=0)
else:   
    lb_scale = np.full((nBins),0.5)
    ub_scale = np.full((nBins),2.)

lb_sigma = np.full((nBins),-np.inf)
lb_nsig = np.full((nBins),-np.inf)

ub_sigma = np.full((nBins),np.inf)
ub_nsig = np.full((nBins),np.inf)

lb = np.concatenate((lb_scale,lb_sigma,lb_nsig),axis=0)
ub = np.concatenate((ub_scale,ub_sigma,ub_nsig),axis=0)

constraints = LinearConstraint( A=np.eye(x.shape[0]), lb=lb, ub=ub,keep_feasible=True )

if runCalibration:
    fnll = nllPars
    #convert fnll to single parameter function fnllx(x)
    fnllx = functools.partial(fnll, nEtaBins=nEtaBins, nPtBins=nPtBins, dataset=dataset, datasetGen=datasetgen, isJ=isJ, etas=etas, binCenters1=binCenters1, binCenters2=binCenters2, good_idx=good_idx)
else:
    fnll = nll
    #convert fnll to single parameter function fnllx(x)
    fnllx = functools.partial(fnll, nEtaBins=nEtaBins, nPtBins=nPtBins, dataset=dataset, datasetGen=datasetgen, isJ=isJ)


fgradnll = jax.jit(jax.value_and_grad(fnllx))

def fgradnlldebug(x):
    f,grad = fgradnll(x)
    if np.isnan(f) or np.any(np.isnan(grad)):
        print("nan detected")
        print(x)
        print(f)
        print(grad)
        
        fnllx(x)
        
    return f,grad

hessnll = hessianlowmem(fnllx)


res = minimize(fgradnll, x,\
    method = 'trust-constr',jac = True, hess=SR1(),constraints=constraints,\
    options={'verbose':3,'disp':True,'maxiter' : maxiter, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

print(res)


fitres = res.x

val,gradient = fgradnll(res.x)
gradfinal = gradient

print(val)
print(gradient)

#print gradient, "gradient"

hessian = hessnll(res.x)
#hessian = np.eye(x.shape[0])

hessfinal = hessian

#print np.linalg.eigvals(hessfinal), "eigenvalues"

invhess = np.linalg.inv(hessfinal)

edm = 0.5*np.matmul(np.matmul(gradfinal.T,invhess),gradfinal)

print(fitres, "+/-", np.sqrt(np.diag(invhess)))
print(edm, "edm")

diag = np.diag(np.sqrt(np.diag(invhess)))
diag = np.linalg.inv(diag)
corr = np.dot(diag,invhess).dot(diag)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.clf()
plt.pcolor(corr, cmap='jet', vmin=-1, vmax=1)
plt.colorbar()
plt.savefig("corrMC.pdf")

if runCalibration:
    plotsPars(res.x,nEtaBins,nPtBins,dataset,datasetgen,isJ,etas, binCenters1, binCenters2, good_idx)

    A = res.x[:nEtaBins, np.newaxis]
    e = res.x[nEtaBins:2*nEtaBins]
    M = res.x[2*nEtaBins:3*nEtaBins]

    hA = ROOT.TH1D("A", "A", nEtaBins, onp.array(etas.tolist()))
    he = ROOT.TH1D("e", "e", nEtaBins, onp.array(etas.tolist()))
    hM = ROOT.TH1D("M", "M", nEtaBins, onp.array(etas.tolist()))

    hA = array2hist(A[:,0], hA, np.sqrt(np.diag(invhess)[:nEtaBins]))
    he = array2hist(e, he, np.sqrt(np.diag(invhess)[nEtaBins:2*nEtaBins]))
    hM = array2hist(M, hM, np.sqrt(np.diag(invhess)[2*nEtaBins:3*nEtaBins]))

    hA.GetYaxis().SetTitle('b field correction')
    he.GetYaxis().SetTitle('material correction')
    hM.GetYaxis().SetTitle('alignment correction')

    hA.GetXaxis().SetTitle('#eta')
    he.GetXaxis().SetTitle('#eta')
    hM.GetXaxis().SetTitle('#eta')

    AeM = res.x[:3*nEtaBins]
    scale = scaleFromPars(AeM, etas, binCenters1, binCenters2, good_idx)
    
    print(scale.shape)
    
    jacobianscale = jax.jit(jax.jacfwd(scaleFromPars))
    jac = jacobianscale(AeM,etas, binCenters1, binCenters2, good_idx)
    invhessAeM = invhess[:3*nEtaBins,:3*nEtaBins]
    scale_invhess = np.matmul(np.matmul(jac,invhessAeM),jac.T)
    scale_err = np.sqrt(np.diag(scale_invhess))
    
    #have to use original numpy to construct the bin edges because for some reason this doesn't work with the arrays returned by jax
    scaleplot = ROOT.TH1D("scale", "scale", nBins, onp.linspace(0, nBins, nBins+1))
    scaleplot = array2hist(scale, scaleplot, scale_err)

    for ibin in range(nBins):
        ieta1 = good_idx[0][ibin]
        ieta2 = good_idx[1][ibin]
        ipt1 = good_idx[2][ibin]
        ipt2 = good_idx[3][ibin]
        scaleplot.GetXaxis().SetBinLabel(ibin+1,'eta1_{}_eta2_{}_pt1_{}_pt2_{}'.format(ieta1,ieta2,ipt1,ipt2))

    scaleplot.GetXaxis().LabelsOption("v")


    f = ROOT.TFile("calibrationMC.root", 'recreate')
    f.cd()

    hA.Write()
    he.Write()
    hM.Write()
    scaleplot.Write()

else:
    plots(res.x,nEtaBins,nPtBins,dataset,datasetgen,isJ)

    f = ROOT.TFile("scaleMC.root", 'recreate')
    f.cd()

    scaleplot = ROOT.TH1D("scale", "scale", nBins, onp.linspace(0, nBins, nBins+1))
    scaleplot.GetYaxis().SetTitle('scale')

    scaleplot = array2hist(fitres[:nBins], scaleplot, np.sqrt(np.diag(invhess)[:nBins]))

    for ibin in range(nBins):
        ieta1 = good_idx[0][ibin]
        ieta2 = good_idx[1][ibin]
        ipt1 = good_idx[2][ibin]
        ipt2 = good_idx[3][ibin]
        scaleplot.GetXaxis().SetBinLabel(ibin+1,'eta1_{}_eta2_{}_pt1_{}_pt2_{}'.format(ieta1,ieta2,ipt1,ipt2))

    scaleplot.GetXaxis().LabelsOption("v")
    scaleplot.Write()

