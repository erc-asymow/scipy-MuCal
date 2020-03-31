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
from autograd.scipy.stats import poisson
import threading
import Queue
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def defineState(nEtaBins,nPtBins):

    #eta1,eta2,mass,phi2,phi1,pt2,pt1

    scale = np.ones((nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),0.03)
    nsig = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),1000)
    
    x = np.concatenate((scale.flatten(),sigma.flatten(), nsig.flatten()),axis=None)

    print x.shape, 'x.shape'
                
    return x.astype("float64")


def nllJ(x,etas,phis,pts,dataset,datasetGen):

    #etas = np.arange(-0.8, 1.2, 0.4)
    #pts = np.array((3.,7.,15.,20.))
    etas = np.array((-0.8,0.8))
    pts = np.array((3.,20.))
    #phis = np.arange(-np.pi, np.pi+2.*np.pi/6.,2.*np.pi/6.)
    #etas = np.array((-0.8,-0.4))
    phis = np.array((-np.pi,np.pi))

    #print datasetGen.shape, "datasetGen"

    #dataset is binned as eta1,eta2,mass,phi2,phi1,pt2,pt1

    sep = np.power(len(etas)-1,2)*np.power(len(pts)-1,2)
    
    scale = x[:sep].reshape((len(etas)-1,len(etas)-1,len(pts)-1,len(pts)-1))
    sigma = x[sep:2*sep].reshape((len(etas)-1,len(etas)-1,len(pts)-1,len(pts)-1))
    nsig = x[2*sep:].reshape((len(etas)-1,len(etas)-1,len(pts)-1,len(pts)-1))

    #print dataset.shape, 'should be eta1,eta2,mass,phi2,phi1,pt2,pt1'

    vals = np.linspace(2.9,3.3,100)

    h= np.tensordot(scale,vals,axes=0) #get a 7-D vector with np.newaxis with all possible combos of kinematics and mass values
    #(eta1, eta2, mass, phi1, phi2, pt1, pt2) h.shape
    
    #print h.shape, "h.shape"
    
    h_ext = np.swapaxes(np.swapaxes(h,2,4),3,4)[:,:,:,np.newaxis,np.newaxis,:,:]
    
    #print h_ext.shape, "h_ext.shape"

    sigma_ext = sigma[:,:,np.newaxis,np.newaxis,np.newaxis,:,:]

    xscale = np.sqrt(2.)*sigma_ext

    #print xscale.shape, "xscale.shape"

    maxZ = ((3.3-h_ext.astype('float64'))/xscale)
    minZ = ((2.9-h_ext.astype('float64'))/xscale)

    #print maxZ.shape, "maxZ.shape"

    arg = np.sqrt(np.pi/2.)*sigma_ext*(erf(maxZ)-erf(minZ))
    
    #print arg.shape, "arg.shape"

    #take tensor product between mass and genMass dimensions and sum over gen masses
    #divide each bin by the sum of gen events in that bin
    den = np.where(np.sum(datasetGen,axis=2)!=0.,np.sum(datasetGen,axis=2),-1)[:,:,np.newaxis,:,:,:,:]

    I = np.sum(np.einsum("ijplmnk,ijqlmnk->ijpqlmnk",arg,datasetGen),axis=3)/den

    #print I.shape, "I.shape"
    
    #eta1,eta2,mass,phi2,phi1,pt2,pt1
    #print scale, sigma, nsig, "pars"

    #give vals the right shape
    vals_ext = vals[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]

    gaus = np.exp(-np.power(vals_ext  -h_ext.astype('float64'), 2.)/(2 * np.power(sigma_ext, 2.)))
    
    #print gaus.shape, "gaus.shape"

    #take tensor product between mass and genMass dimensions and sum over gen masses
    #divide each bin by the sum of gen events in that bin
    den2 = np.where(np.sum(datasetGen,axis=2)!=0.,np.sum(datasetGen,axis=2),1)[:,:,np.newaxis,:,:,:,:]
    pdf = np.sum(np.einsum("ijplmnk,ijqlmnk->ijpqlmnk",gaus,datasetGen),axis=3)/den2/np.where(I>0.,I,-1)

    pdf = np.where(pdf>0.,pdf,0.)

    #print pdf.shape, "pdf.shape"

    #print I
    #print pdf
    massbinwidth = (3.3-2.9)/100

    norm_pdf = nsig*pdf*massbinwidth

    #print norm_pdf.shape, "norm pdf.shape"

    #nexp - nobs*ln(nexp)
    #nexp = Nsig*masspdf(mass|parameters)*massbinwidth

    nll = np.sum(norm_pdf-dataset*np.log(np.where(norm_pdf>0.,norm_pdf,1.)), axis =2)

    #print nll.shape, "nll.shape"

    #print np.sum(nll), "final nll"
        
    return np.sum(nll)


fileJ = open("calInputJMC.pkl", "rb")
datasetJ = pickle.load(fileJ)
fileJgen = open("calInputJMCgen.pkl", "rb")
datasetJgen = pickle.load(fileJgen)

#etas = np.arange(-0.8, 1.2, 0.4)
#pts = np.array((3.,7.,15.,20.))
etas = np.array((-0.8,0.8))
pts = np.array((3.,20.))
#phis = np.arange(-np.pi, np.pi+2.*np.pi/6.,2.*np.pi/6.)
#etas = np.array((-0.8,-0.4))
phis = np.array((-np.pi,np.pi))


x = defineState(len(etas)-1,len(pts)-1)

print "minimising"

xtol = np.finfo('float64').eps

grad = grad(nllJ)
hess = hessian(nllJ)

btol = 1.e-8

res = minimize(nllJ, x, args=(etas,phis,pts,datasetJ,datasetJgen),method = 'trust-constr',jac = grad, hess = hess,options={'verbose':3,'disp':True,'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

print res

hessian = hess(res.x,etas,phis,pts,datasetJ,datasetJgen)
print np.linalg.eigvals(hessian), "eigenvalues"

invhess = np.linalg.inv(hessian)

edm = 0.5*np.matmul(np.matmul(grad(res.x,etas,phis,pts,datasetJ,datasetJgen).T,invhess),grad(res.x,etas,phis,pts,datasetJ,datasetJgen))

print res.x, "+/-", np.sqrt(np.diag(invhess))
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
plt.savefig("corr.pdf")
