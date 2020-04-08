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
from binning import *

parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-runCalibration', '--runCalibration', default=False, action='store_true', help='Use to fit corrections, omit to fit scale parameter')

args = parser.parse_args()
isJ = args.isJ
runCalibration = args.runCalibration

fileJ = open("calInput{}MC_4etaBins_4ptBins.pkl".format('J' if isJ else 'Z'), "rb")
datasetJ = pickle.load(fileJ)
fileJgen = open("calInput{}MCgen_4etaBins_4ptBins.pkl".format('J' if isJ else 'Z'), "rb")
datasetJgen = pickle.load(fileJgen)

pts = ptsJ if isJ else ptsZ

nEtaBins = len(etas)-1
nPtBins = len(pts)-1

if runCalibration:
	x = defineStatePars(nEtaBins,nPtBins, datasetJ)
else:
	x = defineState(nEtaBins,nPtBins, datasetJ)


print "minimising"

xtol = np.finfo('float64').eps

#btol = 1.e-8
btol = 0.1

sep = nEtaBins*nEtaBins*nPtBins*nPtBins

idx = np.where((np.sum(datasetJgen,axis=2)<=1000.).flatten())[0]

good_idx = np.where((np.sum(datasetJgen,axis=2)>1000.).flatten())[0]

if runCalibration:
	bad_idx = np.concatenate((idx, idx+sep), axis=None)
	lb_scale = np.concatenate((0.9*np.ones(nEtaBins),-0.01*np.ones(nEtaBins), -1e-5*np.ones(nEtaBins)),axis=None)
	ub_scale = np.concatenate((1.1*np.ones(nEtaBins),0.01*np.ones(nEtaBins), 1e-5*np.ones(nEtaBins)),axis=None)
	good_idx = np.concatenate((good_idx, good_idx+sep), axis=None)
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
	grad = grad(nllPars)
	hess = hessian(nllPars)

	res = minimize(nllPars, x, args=(nEtaBins,nPtBins,datasetJ,datasetJgen,isJ),\
		method = 'trust-constr',jac = grad, hess=SR1(),constraints=constraints,\
		options={'verbose':3,'disp':True,'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})
else:
	grad = grad(nll)
	hess = hessian(nll)

	res = minimize(nll, x, args=(nEtaBins,nPtBins,datasetJ,datasetJgen,isJ),\
		method = 'trust-constr',jac = grad, hess=SR1(),constraints=constraints,\
		options={'verbose':3,'disp':True,'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

print res

if runCalibration:
	fitres = np.concatenate(res.x[:3*nEtaBins],res.x[3*nEtaBins:][good_idx])
else:
	fitres = res.x[good_idx]

gradient = grad(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen)
gradfinal = gradient[good_idx]

print gradient, "gradient"

hessian = hess(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen)

hessmod = hessian[3*nEtaBins:,:][good_idx,:]
hessfinal = hessmod[:,3*nEtaBins:][:,good_idx]

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

if runCalibration:
	plotsPars(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen,isJ)

	A = ROOT.TH1D("A", "A", nEtaBins, etas)
	e = ROOT.TH1D("e", "e", nEtaBins, etas)
	M = ROOT.TH2D("M", "M", nEtaBins, etas)

	A = array2hist(res.x[:nEtaBins], A, np.sqrt(np.diag(invhess)[:nEtaBins]))
	e = array2hist(res.x[nEtaBins:2*nEtaBins], e, np.sqrt(np.diag(invhess)[nEtaBins:2*nEtaBins]))
	M = array2hist(res.x[2*nEtaBins:3*nEtaBins], e, np.sqrt(np.diag(invhess)[2*nEtaBins:3*nEtaBins]))

	A.GetYaxis().SetTitle('b field correction')
	e.GetYaxis().SetTitle('material correction')
	M.GetYaxis().SetTitle('alignment correction')

	A.GetXaxis().SetTitle('#eta')
	e.GetXaxis().SetTitle('#eta')
	M.GetXaxis().SetTitle('#eta')

	f = ROOT.TFile("calibrationMC.root", 'recreate')
	f.cd()

	A.Write()
	e.Write()
	M.Write()

else:
	plots(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen,isJ)

	f = ROOT.TFile("scaleMC.root", 'recreate')
	f.cd()

	scaleplot = ROOT.TH1D("scale", "scale", good_idx.shape[0]/3, np.linspace(0, good_idx.shape[0]/3, good_idx.shape[0]/3+1))
	scaleplot.GetYaxis().SetTitle('scale')

	scaleplot = array2hist(fitres[:good_idx.shape[0]/3], scaleplot, np.sqrt(np.diag(invhess)[:good_idx.shape[0]/3]))
	scaleplot.Write()

