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

def defineState(nEtaBins,nPtBins,dataset):

    #eta1,eta2,mass,phi2,phi1,pt2,pt1

    scale = np.ones((nEtaBins,nEtaBins,nPtBins,nPtBins)) + np.random.normal(0, 0.0005, (nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-3.9) 
    nsig = np.log(np.where(np.sum(dataset,axis=2)>0.,np.sum(dataset,axis=2),2.))
    
    x = np.concatenate((scale.flatten(), sigma.flatten(), nsig.flatten()),axis=None)
    #x = np.concatenate((scale.flatten(), sigma.flatten()),axis=None)
    #x = nsig.flatten()
    
    print x.shape, 'x.shape'
                
    return x.astype('float64')

def kernelpdf(scale, sigma, dataset, datasetGen):

    #print np.sum(dataset), "n events"

    #print datasetGen.shape, "datasetGen"

    #dataset is binned as eta1,eta2,mass,pt2,pt1

    maxR = np.full((100),3.15)
    minR = np.full((100),3.05)

    valsReco = np.linspace(minR[0],maxR[0],100)
    valsGen = valsReco

    h= np.tensordot(scale,valsGen,axes=0) #get a 7-D vector with np.newaxis with all possible combos of kinematics and gen mass values
    #(eta1, eta2, mass, pt1, pt2) h.shape
    
    #print h.shape, "h.shape"
    
    h_ext = np.swapaxes(np.swapaxes(h,2,4),3,4)[:,:,np.newaxis,:,:,:]
    
    #print h_ext.shape, "h_ext.shape"

    sigma_ext = sigma[:,:,np.newaxis,np.newaxis,:,:]

    xscale = np.sqrt(2.)*sigma_ext

    #print xscale.shape, "xscale.shape"

    maxR_ext = maxR[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
    minR_ext = minR[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]

    #print maxR_ext.shape, "maxR.shape"

    maxZ = ((maxR_ext-h_ext.astype('float64'))/xscale)
    minZ = ((minR_ext-h_ext.astype('float64'))/xscale)

    #print maxZ.shape, "maxZ.shape"

    arg = np.sqrt(np.pi/2.)*sigma_ext*(erf(maxZ)-erf(minZ))
    
    #print arg.shape, "arg.shape"

    #take tensor product between mass and genMass dimensions and sum over gen masses
    #divide each bin by the sum of gen events in that bin
    den = np.where(np.sum(datasetGen,axis=2)>5000.,np.sum(datasetGen,axis=2),-1)[:,:,np.newaxis,:,:]

    #print den.shape, "den.shape"

    I = np.sum(arg*datasetGen[:,:,np.newaxis,:,:,:],axis=3)/den

    #print I.shape, "I.shape"
    
    #eta1,eta2,mass,phi2,phi1,pt2,pt1
    #print scale, sigma,  "pars"

    #dim vals = (mass,)
    #dim h_ext = (eta1, eta2, genMass, phi1, phi2, pt1, pt2)

    #give vals the right shape -> add dimension for gen mass (axis = 3)
    vals_ext = valsReco[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis, np.newaxis]

    gaus = np.exp(-np.power(vals_ext  -h_ext.astype('float64'), 2.)/(2 * np.power(sigma_ext, 2.)))
    
    #print gaus.shape, "gaus.shape"

    #take tensor product between mass and genMass dimensions and sum over gen masses
    #divide each bin by the sum of gen events in that bin
    den2 = np.where(np.sum(datasetGen,axis=2)>5000.,np.sum(datasetGen,axis=2),1)[:,:,np.newaxis,:,:]

    #print den2.shape, "den2.shape"

    pdf = np.sum(gaus*datasetGen[:,:,np.newaxis,:,:,:],axis=3)/den2/np.where(I>0.,I,-1)

    pdf = np.where(pdf>0.,pdf,0.)

    #print pdf.shape, "pdf.shape"

    #print I
    #print pdf
    massbinwidth = (maxR[0]-minR[0])/100

    #print pdf.shape, "pdf.shape", nsig.shape, "nsig.shape"
    norm_pdf = pdf*massbinwidth
    
    return norm_pdf

def bkgpdf(slope):

    maxR = np.full((100),3.15)
    minR = np.full((100),3.05)

    valsReco = np.linspace(minR[0],maxR[0],100)

    #take all the combinations mass-kinematics

    combos = np.tensordot(slope,valsReco,axes=0) 
    combos_ext = np.swapaxes(np.swapaxes(combos,2,4),3,4)[:,:,np.newaxis,:,np.newaxis,np.newaxis,:,:]

    maxR_ext = maxR[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
    minR_ext = minR[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]

    I = (np.exp(maxR_ext) - np.exp(minR_ext))/slope[:,:,np.newaxis,:,:]

    pdf = np.exp(combos)/I[:,:,:,np.newaxis,np.newaxis,:,:]

    return pdf


def nll(x,nEtaBins,nPtBins,dataset,datasetGen):

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)
    
    scale = x[:sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.exp(x[sep:2*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))

    #scale = np.ones((nEtaBins,nEtaBins,nPtBins,nPtBins))
    #sigma = np.exp(np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-3.5))

    nsig = np.exp(x[2*sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    #nsig = np.sum(datasetGen,axis=2)

    norm_pdf = nsig[:,:,np.newaxis,:,:]*kernelpdf(scale, sigma, dataset, datasetGen)

    #nexp - nobs*ln(nexp)
    #nexp = Nsig*masspdf(mass|parameters)*massbinwidth

    nll = nsig - np.sum(dataset*np.log(np.where(norm_pdf>0.,norm_pdf,1.)), axis =2)

    #nsig - nobs*ln(nsig)
    #nll = nsig-np.sum(dataset*np.log(nsig), axis=2)

    #print nll.shape, "nll.shape"
    #print np.sum(nll), "final nll"
        
    return np.sum(nll)

def plots(x,nEtaBins,nPtBins,dataset,datasetGen):

    maxR = 3.15
    minR = 3.05

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)
    
    scale = x[:sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.exp(x[sep:2*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    #scale = np.ones((nEtaBins,nEtaBins,nPtBins,nPtBins))
    #sigma = np.exp(np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-3.5))
    nsig = np.exp(x[2*sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    #nsig = np.sum(dataset,axis=2)

    #print dataset.shape, 'eta1,eta2,mass,phi2,phi1,pt2,pt1'

    pdf = kernelpdf(scale, sigma, dataset, datasetGen)
    pdf = nsig[:,:,np.newaxis,:,:]*pdf

    #mass plots for each bin

    mass = np.linspace(minR,maxR,100)

    for ieta1 in range(nEtaBins):
        for ieta2 in range(nEtaBins):
            for ipt1 in range(nPtBins):
                for ipt2 in range(nPtBins):

                    if not pdf[ieta1,ieta2,:,ipt1,ipt2].any()>0.: continue

                    scale_bin = scale[ieta1,ieta2,ipt1,ipt2]
                    sigma_bin = sigma[ieta1,ieta2,ipt1,ipt2]
                    #nsig_bin = nsig[ieta1,ieta2,0,0,0,ipt1,ipt2]
                    nsig_bin = nsig[ieta1,ieta2,ipt1,ipt2]
                    nsig_true = np.sum(dataset,axis=2)[ieta1,ieta2,ipt1,ipt2]


                    plt.clf()
                  
                    fig, (ax1, ax2) = plt.subplots(nrows=2)
                    ax1.errorbar(mass, dataset[ieta1,ieta2,:,ipt1,ipt2], yerr=np.sqrt(dataset[ieta1,ieta2,:,ipt1,ipt2]), fmt='.')
                    ax1.set_ylabel('number of events')
                    ax1.text(0.95, 0.95, 'scale: {:.5f}\n sigma: {:.2f}\n nsig: {}\n nsig true {}'.format(scale_bin,sigma_bin,nsig_bin,nsig_true),
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax1.transAxes,
                    color='black', fontsize=12)
                    
                    ax1.plot(mass, pdf[ieta1,ieta2,:,ipt1,ipt2])
                    plt.xlim(minR, maxR)
        
                    ax2.errorbar(mass,dataset[ieta1,ieta2,:,ipt1,ipt2]/pdf[ieta1,ieta2,:,ipt1,ipt2],yerr=np.sqrt(dataset[ieta1,ieta2,:,ipt1,ipt2])/pdf[ieta1,ieta2,:,ipt1,ipt2], fmt='.')
                    ax2.set_xlabel('dimuon mass')
                    ax2.set_ylabel('ratio data/pdf')
                    plt.xlim(minR, maxR)

                    plt.savefig('PLOTSJ/plot_{}{}{}{}.pdf'.format(ieta1,ieta2,ipt1,ipt2))
    

fileJ = open("calInputJMC.pkl", "rb")
datasetJ = pickle.load(fileJ)
fileJgen = open("calInputJMCgen.pkl", "rb")
datasetJgen = pickle.load(fileJgen)

etas = np.arange(-0.8, 1.2, 0.4)
#pts = np.array((3.,7.,15.,20.))
#etas = np.array((-0.8,0.8))
pts = np.array((3.,20.))
#phis = np.arange(-np.pi, np.pi+2.*np.pi/6.,2.*np.pi/6.)
#etas = np.array((-0.8,-0.4))
phis = np.array((-np.pi,np.pi))

nEtaBins = len(etas)-1
nPtBins = len(pts)-1

x = defineState(nEtaBins,nPtBins, datasetJ)

#plots(x,nEtaBins,nPtBins,datasetJ,datasetJgen)

print "minimising"

xtol = np.finfo('float64').eps

grad = grad(nll)
hess = hessian(nll)

btol = 1.e-8

idx = np.where((np.sum(datasetJgen,axis=2)<5000.).flatten())[0]

lb_scale = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),0.).flatten()
lb_sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-np.inf).flatten()
lb_nsig = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-np.inf).flatten()

lb_scale[idx] = np.full(len(idx),1.)
lb_sigma[idx] = np.full(len(idx),-3.5)
lb_nsig[idx] = np.full(len(idx),6.9)

ub_scale = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),np.inf).flatten()
ub_sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),np.inf).flatten()
ub_nsig = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),20.).flatten()

ub_scale[idx] = np.full(len(idx),1.)
ub_sigma[idx] = np.full(len(idx),-3.5)
ub_nsig[idx] = np.full(len(idx),6.9)

lb = np.concatenate((lb_scale,lb_sigma,lb_nsig),axis=None)
ub = np.concatenate((ub_scale,ub_sigma,ub_nsig),axis=None)
#lb = np.concatenate((lb_scale,lb_sigma),axis=None)
#ub = np.concatenate((ub_scale,ub_sigma),axis=None)
#lb = lb_nsig
#ub = ub_nsig

constraints = LinearConstraint( A=np.eye(x.shape[0]), lb=lb, ub=ub,keep_feasible=True )

res = minimize(nll, x, args=(nEtaBins,nPtBins,datasetJ,datasetJgen),method = 'trust-constr',jac = grad, hess=SR1(), constraints = constraints,options={'verbose':3,'disp':True,'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

print res

plots(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen)

good_idx = np.where((np.sum(datasetJgen,axis=2)>5000.).flatten())[0]

#good_idx = np.concatenate((good_idx, good_idx+nEtaBins*nEtaBins*nPtBins*nPtBins,good_idx+2*nEtaBins*nEtaBins*nPtBins*nPtBins), axis=None)
good_idx = np.concatenate((good_idx, good_idx+nEtaBins*nEtaBins*nPtBins*nPtBins,good_idx), axis=None)


fitres = res.x[good_idx]

gradient = grad(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen)
gradfinal = gradient[good_idx]

print gradient, "gradient"

hessian = hess(res.x,nEtaBins,nPtBins,datasetJ,datasetJgen)

hessmod = hessian[good_idx,:]
hessfinal = hessmod[:,good_idx]

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
plt.savefig("corr.pdf")
