import os
os.environ["OMP_NUM_THREADS"] = "32" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "32" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "32" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "32" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "32" # export NUMEXPR_NUM_THREADS=6

#prepare possible migration to jax
#import jax.numpy as np
#from jax import grad, hessian, jacobian, config
#from jax.scipy.special import erf
#config.update('jax_enable_x64', True)

from autograd import grad, hessian, jacobian
import autograd.numpy as np
from autograd.scipy.special import erf

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

from fittingFunctionsBinned import defineStatePars, nllPars
from binning import etas, ptsJ, ptsJC, ptsZ
import argparse


def scaleFromPars(x):

    A = x[:nEtaBins, np.newaxis]
    e = x[nEtaBins:2*nEtaBins]
    M = x[2*nEtaBins:3*nEtaBins]

    etasC = (etas[:-1] + etas[1:]) / 2.

    s = np.sin(2*np.arctan(np.exp(-etasC)))
    
    c = np.array((0.1703978,0.21041214,0.26139158),dtype='float64') #bin centers in curvature

    term1 = A-s[:,np.newaxis]*np.tensordot(e,c,axes=0)+np.tensordot(M,1./c,axes=0)
    term2 = A-s[:,np.newaxis]*np.tensordot(e,c,axes=0)-np.tensordot(M,1./c,axes=0)

    scale = np.sqrt(np.swapaxes(np.tensordot(term1,term2, axes=0),1,2))

    return scale.flatten()


parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-runCalibration', '--runCalibration', default=False, action='store_true', help='Use to fit corrections, omit to fit scale parameter')

args = parser.parse_args()
isJ = args.isJ
runCalibration = args.runCalibration

fileJ = open("calInput{}MC_4etaBins_3ptBins.pkl".format('J' if isJ else 'Z'), "rb")
datasetJ = pickle.load(fileJ)
fileJgen = open("calInput{}MCgen_4etaBins_3ptBins.pkl".format('J' if isJ else 'Z'), "rb")
datasetJgen = pickle.load(fileJgen)

pts = ptsJ if isJ else ptsZ
print pts

nEtaBins = len(etas)-1
nPtBins = len(pts)-1

if runCalibration:
    x = defineStatePars(nEtaBins,nPtBins, datasetJ, isJ)
else:
    x = defineState(nEtaBins,nPtBins, datasetJ)


print "minimising"

xtol = np.finfo('float64').eps
#btol = 1.e-8
btol = 0.1
#maxiter = 100000
maxiter = 2

sep = nEtaBins*nEtaBins*nPtBins*nPtBins

idx = np.where((np.sum(datasetJgen,axis=2)<=1000.).flatten())[0]

good_idx = np.where((np.sum(datasetJgen,axis=2)>1000.).flatten())[0]

if runCalibration:

    bad_idx = np.concatenate((idx, idx+sep), axis=None)
    lb_scale = np.concatenate((0.009*np.ones(nEtaBins),-0.01*np.ones(nEtaBins), -1e-5*np.ones(nEtaBins)),axis=None)
    ub_scale = np.concatenate((1.001*np.ones(nEtaBins),0.01*np.ones(nEtaBins), 1e-5*np.ones(nEtaBins)),axis=None)
    pars_idx = np.linspace(0, nEtaBins-1,nEtaBins,dtype=np.int16)
    good_idx = np.concatenate((pars_idx,nEtaBins+pars_idx,2*nEtaBins+pars_idx,3*nEtaBins+good_idx, 3*nEtaBins+good_idx+sep), axis=None)

else:   
    bad_idx = np.concatenate((idx, idx+sep,idx+2*sep), axis=None)
    lb_scale = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),0.).flatten()
    ub_scale = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),2.).flatten()
    good_idx = np.concatenate((good_idx, good_idx+sep,good_idx+2*sep), axis=None)

lb_sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-np.inf).flatten()
lb_nsig = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-np.inf).flatten()

ub_sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),np.inf).flatten()
ub_nsig = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),np.inf).flatten()

lb = np.concatenate((lb_scale,lb_sigma,lb_nsig),axis=None)
ub = np.concatenate((ub_scale,ub_sigma,ub_nsig),axis=None)

#bounds for fixed parameters must be equal to the starting values
if runCalibration:
    lb[3*nEtaBins+bad_idx] = x[3*nEtaBins+bad_idx]
    ub[3*nEtaBins+bad_idx] = x[3*nEtaBins+bad_idx]
else:
    lb[bad_idx] = x[bad_idx]
    ub[bad_idx] = x[bad_idx]

constraints = LinearConstraint( A=np.eye(x.shape[0]), lb=lb, ub=ub,keep_feasible=True )

if runCalibration:
    gradnll = grad(nllPars)
    hessnll = hessian(nllPars)

    res = minimize(nllPars, x, args=(nEtaBins,nPtBins,datasetJ,datasetJgen,isJ),\
        method = 'trust-constr',jac = gradnll, hess=SR1(),constraints=constraints,\
        options={'verbose':3,'disp':True,'maxiter' : maxiter, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})
else:
    gradnll = grad(nll)
    hessnll = hessian(nll)

    res = minimize(nll, x, args=(nEtaBins,nPtBins,datasetJ,datasetJgen,isJ),\
        method = 'trust-constr',jac = gradnll, hess=SR1(),constraints=constraints,\
        options={'verbose':3,'disp':True,'maxiter' : maxiter, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

print res


fitres = res.x[good_idx]

gradient = gradnll(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen,isJ)
gradfinal = gradient[good_idx]

#print gradient, "gradient"

hessian = hessnll(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen,isJ)
#hessian = np.eye(x.shape[0])

hessmod = hessian[good_idx,:]
hessfinal = hessmod[:,good_idx] 

#print np.linalg.eigvals(hessfinal), "eigenvalues"

invhess = np.linalg.inv(hessfinal)

edm = 0.5*np.matmul(np.matmul(gradfinal.T,invhess),gradfinal)

print fitres, "+/-", np.sqrt(np.diag(invhess))
print edm, "edm"

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
    #plotsPars(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen,isJ)

    A = res.x[:nEtaBins, np.newaxis]
    e = res.x[nEtaBins:2*nEtaBins]
    M = res.x[2*nEtaBins:3*nEtaBins]

    hA = ROOT.TH1D("A", "A", nEtaBins, etas)
    he = ROOT.TH1D("e", "e", nEtaBins, etas)
    hM = ROOT.TH1D("M", "M", nEtaBins, etas)

    hA = array2hist(A[:,0], hA, np.sqrt(np.diag(invhess)[:nEtaBins]))
    he = array2hist(e, he, np.sqrt(np.diag(invhess)[nEtaBins:2*nEtaBins]))
    hM = array2hist(M, hM, np.sqrt(np.diag(invhess)[2*nEtaBins:3*nEtaBins]))

    hA.GetYaxis().SetTitle('b field correction')
    he.GetYaxis().SetTitle('material correction')
    hM.GetYaxis().SetTitle('alignment correction')

    hA.GetXaxis().SetTitle('#eta')
    he.GetXaxis().SetTitle('#eta')
    hM.GetXaxis().SetTitle('#eta')

    scale_idx = np.where((np.sum(datasetJgen,axis=2)>1000.).flatten())[0]

    scale = scaleFromPars(res.x)[scale_idx]
    jacobianscale = jacobian(scaleFromPars)
    jac = jacobianscale(res.x)
    jac = jac[scale_idx,:]
    jac = jac[:,good_idx]
    scale_invhess = np.matmul(np.matmul(jac,invhess),jac.T)
    scale_err = np.sqrt(np.diag(scale_invhess))
    print("scale_err:")
    print(scale_err)
    scaleplot = ROOT.TH1D("scale", "scale", scale_idx.shape[0], np.linspace(0, scale_idx.shape[0], scale_idx.shape[0]+1))
    scaleplot.GetYaxis().SetTitle('scale')
    scaleplot = array2hist(scale, scaleplot, np.sqrt(scale_err))


    f = ROOT.TFile("calibrationMC.root", 'recreate')
    f.cd()

    hA.Write()
    he.Write()
    hM.Write()
    scaleplot.Write()

else:
    plots(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen,isJ)

    f = ROOT.TFile("scaleMC.root", 'recreate')
    f.cd()

    scaleplot = ROOT.TH1D("scale", "scale", good_idx.shape[0]/3, np.linspace(0, good_idx.shape[0]/3, good_idx.shape[0]/3+1))
    scaleplot.GetYaxis().SetTitle('scale')

    scaleplot = array2hist(fitres[:good_idx.shape[0]/3], scaleplot, np.sqrt(np.diag(invhess)[:good_idx.shape[0]/3]))
    scaleplot.Write()

