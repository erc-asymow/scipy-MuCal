from autograd import grad, hessian, jacobian
import autograd.numpy as np
import ROOT
import pickle
from termcolor import colored
from scipy.optimize import minimize, SR1, LinearConstraint, check_grad, approx_fprime
from scipy.optimize import Bounds
import itertools
from root_numpy import array2hist, fill_hist
from autograd.scipy.special import erf

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from fittingFunctionsBinned import *

fileJ = open("calInputJMC_4etaBins_4ptBins.pkl", "rb")
datasetJ = pickle.load(fileJ)
fileJgen = open("calInputJMCgen_4etaBins_4ptBins.pkl", "rb")
datasetJgen = pickle.load(fileJgen)

etas = np.arange(-0.8, 1.2, 0.4)
#etas = np.array((-0.8,0.8))
pts = np.array((3.,4.5,5.5,7.,20.))
#pts = np.array((3.,20.))

nEtaBins = len(etas)-1
nPtBins = len(pts)-1

x = defineState(nEtaBins,nPtBins, datasetJ)

print "minimising"

xtol = np.finfo('float64').eps

btol = 1.e-8

sep = nEtaBins*nEtaBins*nPtBins*nPtBins
idx = np.where((np.sum(datasetJgen,axis=2)<1000.).flatten())[0]
bad_idx = np.concatenate((idx, idx+sep,idx+2*sep), axis=None)

lb_scale = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),0.).flatten()
lb_sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-np.inf).flatten()
lb_nsig = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-np.inf).flatten()

#lb_scale[idx] = np.full(len(idx),1.)
#lb_sigma[idx] = np.full(len(idx),-3.5)
#lb_nsig[idx] = np.full(len(idx),6.9)

ub_scale = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),2.).flatten()
ub_sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),np.inf).flatten()
ub_nsig = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),np.inf).flatten()

#ub_scale[idx] = np.full(len(idx),1.)
#ub_sigma[idx] = np.full(len(idx),-3.5)
#ub_nsig[idx] = np.full(len(idx),6.9)

lb = np.concatenate((lb_scale,lb_sigma,lb_nsig),axis=None)
ub = np.concatenate((ub_scale,ub_sigma,ub_nsig),axis=None)

#bounds for fixed parameters must be equal to the starting values
lb[bad_idx] = x[bad_idx]
ub[bad_idx] = x[bad_idx]

constraints = LinearConstraint( A=np.eye(x.shape[0]), lb=lb, ub=ub,keep_feasible=True )

grad = grad(nll)
hess = hessian(nll)

res = minimize(nll, x, args=(nEtaBins,nPtBins,datasetJ,datasetJgen),\
	method = 'trust-constr',jac = grad, hess=SR1(), constraints = constraints,\
	options={'verbose':3,'disp':True,'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

print res

good_idx = np.where((np.sum(datasetJgen,axis=2)>1000.).flatten())[0]
good_idx = np.concatenate((good_idx, good_idx+sep,good_idx+2*sep), axis=None)

fitres = res.x[good_idx]

gradient = grad(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen)
gradfinal = gradient[good_idx]

print gradient, "gradient"

hessian = hess(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen)

hessmod = hessian[good_idx,:]
hessfinal = hessmod[:,good_idx]

hessfinal = 0.5*(hessfinal + np.transpose(hessfinal))

print np.linalg.eigvals(hessfinal), "eigenvalues"

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

plots(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen)

scaleplot = ROOT.TH1D("scale", "scale", good_idx.shape[0]/3, np.linspace(0, good_idx.shape[0]/3, good_idx.shape[0]/3+1))
scaleplot.GetYaxis().SetTitle('scale')

scaleplot = array2hist(fitres[:good_idx.shape[0]/3], scaleplot, np.sqrt(np.diag(invhess)[:good_idx.shape[0]/3]))

f = ROOT.TFile("scaleMC.root", 'recreate')
f.cd()

scaleplot.Write()


