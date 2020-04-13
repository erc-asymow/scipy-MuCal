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

def scaleFromPars(AeM, etas, binCenters1, binCenters2):

    A = AeM[:nEtaBins]
    e = AeM[nEtaBins:2*nEtaBins]
    M = AeM[2*nEtaBins:3*nEtaBins]

    etasC = (etas[:-1] + etas[1:]) / 2.

    s = np.sin(2*np.arctan(np.exp(-etasC)))
    
    c1 = binCenters1
    c2 = binCenters2

    #print c1, c2

    A1 = A[:,np.newaxis,np.newaxis,np.newaxis]
    e1 = e[:,np.newaxis,np.newaxis,np.newaxis]
    M1 = M[:,np.newaxis,np.newaxis,np.newaxis]

    A2 = A[np.newaxis,:,np.newaxis,np.newaxis]
    e2 = e[np.newaxis,:,np.newaxis,np.newaxis]
    M2 = M[np.newaxis,:,np.newaxis,np.newaxis]

    term1 = A1-s[:,np.newaxis,np.newaxis,np.newaxis]*e1*c1+M1/c1
    term2 = A2-s[np.newaxis,:,np.newaxis,np.newaxis]*e2*c2-M2/c2
    #combos = np.swapaxes(np.tensordot(term1,term2, axes=0),1,2)
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

filegen = open("calInput{}MCgen_4etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
datasetgen = pickle.load(filegen)

nEtaBins = len(etas)-1
nPtBins = len(pts)-1

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
#maxiter = 2

sep = nEtaBins*nEtaBins*nPtBins*nPtBins

idx = np.where((np.sum(datasetgen,axis=2)<=4000.).flatten())[0]

good_idx = np.where((np.sum(datasetgen,axis=2)>4000.).flatten())[0]

if runCalibration:

    bad_idx = np.concatenate((idx, idx+sep), axis=0)
    lb_scale = np.concatenate((0.009*np.ones(nEtaBins),-0.01*np.ones(nEtaBins), -1e-5*np.ones(nEtaBins)),axis=0)
    ub_scale = np.concatenate((1.001*np.ones(nEtaBins),0.01*np.ones(nEtaBins), 1e-5*np.ones(nEtaBins)),axis=0)
    pars_idx = np.linspace(0, nEtaBins-1,nEtaBins,dtype=np.int16)
    good_idx = np.concatenate((pars_idx,nEtaBins+pars_idx,2*nEtaBins+pars_idx,3*nEtaBins+good_idx, 3*nEtaBins+good_idx+sep), axis=0)
else:   
    bad_idx = np.concatenate((idx, idx+sep,idx+2*sep), axis=0)
    lb_scale = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),0.).flatten()
    ub_scale = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),2.).flatten()
    good_idx = np.concatenate((good_idx, good_idx+sep,good_idx+2*sep), axis=0)

lb_sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-np.inf).flatten()
lb_nsig = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-np.inf).flatten()

ub_sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),np.inf).flatten()
ub_nsig = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),np.inf).flatten()

lb = np.concatenate((lb_scale,lb_sigma,lb_nsig),axis=0)
ub = np.concatenate((ub_scale,ub_sigma,ub_nsig),axis=0)

#bounds for fixed parameters must be equal to the starting values
if runCalibration:
    lb = jax.ops.index_update(lb, 3*nEtaBins+bad_idx, x[3*nEtaBins+bad_idx])
    ub = jax.ops.index_update(ub, 3*nEtaBins+bad_idx, x[3*nEtaBins+bad_idx])
else:
    lb = jax.ops.index_update(lb, bad_idx, x[bad_idx])
    ub = jax.ops.index_update(ub, bad_idx, x[bad_idx])

constraints = LinearConstraint( A=np.eye(x.shape[0]), lb=lb, ub=ub,keep_feasible=True )

if runCalibration:
    fnll = nllPars
    #convert fnll to single parameter function fnllx(x)
    fnllx = functools.partial(fnll, nEtaBins=nEtaBins, nPtBins=nPtBins, dataset=dataset, datasetGen=datasetgen, isJ=isJ, etas=etas, binCenters1=binCenters1, binCenters2=binCenters2)
else:
    fnll = nll
    #convert fnll to single parameter function fnllx(x)
    fnllx = functools.partial(fnll, nEtaBins=nEtaBins, nPtBins=nPtBins, dataset=dataset, datasetGen=datasetgen, isJ=isJ)

fgradnll = jax.jit(jax.value_and_grad(fnllx))
#fgradnll = jax.value_and_grad(fnllx)

hessnll = hessianlowmem(fnllx)

res = minimize(fgradnll, x,\
    method = 'trust-constr',jac = True, hess=SR1(),constraints=constraints,\
    options={'verbose':3,'disp':True,'maxiter' : maxiter, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

print(res)


fitres = res.x[good_idx]

val,gradient = fgradnll(res.x)
gradfinal = gradient[good_idx]

#print gradient, "gradient"

hessian = hessnll(res.x)
#hessian = np.eye(x.shape[0])

hessmod = hessian[good_idx,:]
hessfinal = hessmod[:,good_idx] 

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
    plotsPars(res.x,nEtaBins,nPtBins,dataset,datasetgen,isJ,etas, binCenters1, binCenters2)

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

    scale_idx = np.where((np.sum(datasetgen,axis=2)>4000.).flatten())[0]

    AeM = res.x[:3*nEtaBins]
    scale = scaleFromPars(AeM, etas, binCenters1, binCenters2)
    
    jacobianscale = jax.jit(jax.jacfwd(scaleFromPars))
    jac = jacobianscale(AeM,etas, binCenters1, binCenters2)
    jac = jac[scale_idx,:]
    invhessAeM = invhess[:3*nEtaBins,:3*nEtaBins]
    scale_invhess = np.matmul(np.matmul(jac,invhessAeM),jac.T)
    scale_err = np.sqrt(np.diag(scale_invhess))
    
    #have to use original numpy to construct the bin edges because for some reason this doesn't work with the arrays returned by jax
    scaleplot = ROOT.TH1D("scale", "scale", scale_idx.shape[0], onp.linspace(0, scale_idx.shape[0], scale_idx.shape[0]+1))

    #stuff for assigning the correct label to bins in the unrolled plot
    scale_good = scale[scale_idx]
    scale_new = np.zeros_like(scale)
    scale_new = jax.ops.index_update(scale_new, scale_idx, scale_good)
    scale_4d = np.reshape(scale_new,(nEtaBins,nEtaBins,nPtBins,nPtBins))

    scaleplot.GetYaxis().SetTitle('scale')
    scaleplot = array2hist(scale_good, scaleplot, scale_err)

    bin1D = 1
    for ieta1 in range(nEtaBins):
        for ieta2 in range(nEtaBins):
            for ipt1 in range(nPtBins):
                for ipt2 in range(nPtBins):

                    if scale_4d[ieta1,ieta2,ipt1,ipt2] == 0.: continue
                    scaleplot.GetXaxis().SetBinLabel(bin1D,'eta1_{}_eta2_{}_pt1_{}_pt2_{}'.format(ieta1,ieta2,ipt1,ipt2))
                    bin1D = bin1D+1

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

    scaleplot = ROOT.TH1D("scale", "scale", good_idx.shape[0]/3, onp.linspace(0, good_idx.shape[0]/3, good_idx.shape[0]/3+1))
    scaleplot.GetYaxis().SetTitle('scale')

    scaleplot = array2hist(fitres[:good_idx.shape[0]/3], scaleplot, np.sqrt(np.diag(invhess)[:good_idx.shape[0]/3]))

    scale_idx = np.where((np.sum(datasetgen,axis=2)>4000.).flatten())[0]

    scale = res.x[:sep]
    scale_good = scale[scale_idx]
    scale_new = np.zeros_like(scale)
    scale_new = jax.ops.index_update(scale_new, scale_idx, scale_good)
    scale_4d = np.reshape(scale_new,(nEtaBins,nEtaBins,nPtBins,nPtBins))

    bin1D = 1
    for ieta1 in range(nEtaBins):
        for ieta2 in range(nEtaBins):
            for ipt1 in range(nPtBins):
                for ipt2 in range(nPtBins):

                    if scale_4d[ieta1,ieta2,ipt1,ipt2] == 0.: continue
                    scaleplot.GetXaxis().SetBinLabel(bin1D,'eta1_{}_eta2_{}_pt1_{}_pt2_{}'.format(ieta1,ieta2,ipt1,ipt2))
                    bin1D = bin1D+1

    scaleplot.GetXaxis().LabelsOption("v")
    scaleplot.Write()

