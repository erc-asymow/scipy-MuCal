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

    scale = np.ones((nEtaBins,nEtaBins,nPtBins,nPtBins)) + np.random.normal(0, 0.0005, (nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-3.9) 
    nsig = np.log(np.where(np.sum(dataset,axis=2)>0.,np.sum(dataset,axis=2),2.))
        
    x = np.concatenate((scale.flatten(), sigma.flatten(), nsig.flatten()),axis=None)
        
    print x.shape, 'x.shape'
                
    return x.astype('float64')

def defineStatebkg(nEtaBins,nPtBins,dataset):

    scale = np.ones((nEtaBins,nEtaBins,nPtBins,nPtBins)) + np.random.normal(0, 0.0005, (nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-3.9) 
    nsig = np.log(np.where(np.sum(dataset,axis=2)>0.,0.9*np.sum(dataset,axis=2),2.))
    slope = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-0.1) 
    nbkg = np.log(np.where(np.sum(dataset,axis=2)>0.,0.1*np.sum(dataset,axis=2),2.))
    
    x = np.concatenate((scale.flatten(), sigma.flatten(), nsig.flatten(),slope.flatten(),nbkg.flatten()),axis=None)
    
    print x.shape, 'x.shape'
                
    return x.astype('float64')

def kernelpdf(scale, sigma, dataset, datasetGen):

    #dataset is binned as eta1,eta2,mass,pt2,pt1

    maxR = np.full((100),3.3)
    minR = np.full((100),2.9)

    valsReco = np.linspace(minR[0],maxR[0],100)
    valsGen = valsReco

    h= np.tensordot(scale,valsGen,axes=0) #get a 5D vector with np.newaxis with all possible combos of kinematics and gen mass values
    h_ext = np.swapaxes(np.swapaxes(h,2,4),3,4)[:,:,np.newaxis,:,:,:]

    sigma_ext = sigma[:,:,np.newaxis,np.newaxis,:,:]

    xscale = np.sqrt(2.)*sigma_ext

    maxR_ext = maxR[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
    minR_ext = minR[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]

    maxZ = ((maxR_ext-h_ext.astype('float64'))/xscale)
    minZ = ((minR_ext-h_ext.astype('float64'))/xscale)

    arg = np.sqrt(np.pi/2.)*sigma_ext*(erf(maxZ)-erf(minZ))

    #take tensor product between mass and genMass dimensions and sum over gen masses
    #divide each bin by the sum of gen events in that bin
    den = np.where(np.sum(datasetGen,axis=2)>1000.,np.sum(datasetGen,axis=2),-1)[:,:,np.newaxis,:,:]

    I = np.sum(arg*datasetGen[:,:,np.newaxis,:,:,:],axis=3)/den

    #give vals the right shape -> add dimension for gen mass (axis = 3)
    vals_ext = valsReco[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis, np.newaxis]

    gaus = np.exp(-np.power(vals_ext  -h_ext.astype('float64'), 2.)/(2 * np.power(sigma_ext, 2.)))

    #take tensor product between mass and genMass dimensions and sum over gen masses
    #divide each bin by the sum of gen events in that bin
    den2 = np.where(np.sum(datasetGen,axis=2)>1000.,np.sum(datasetGen,axis=2),1)[:,:,np.newaxis,:,:]

    pdf = np.sum(gaus*datasetGen[:,:,np.newaxis,:,:,:],axis=3)/den2/np.where(I>0.,I,-1)

    pdf = np.where(pdf>0.,pdf,0.)

    massbinwidth = (maxR[0]-minR[0])/100

    pdf = pdf*massbinwidth
    
    return pdf

def exppdf(slope):

    maxR = 3.3
    minR = 2.9

    valsReco = np.linspace(minR,maxR,100)

    I = (np.exp(-slope*minR) - np.exp(-slope*maxR))/slope

    massbinwidth = (maxR-minR)/100

    h = np.tensordot(slope,valsReco,axes=0) 
    h_ext = np.swapaxes(np.swapaxes(h,2,4),3,4)

    pdf = np.exp(-h_ext)/I

    return pdf*massbinwidth


def nll(x,nEtaBins,nPtBins,dataset,datasetGen):

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)
    
    scale = x[:sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.exp(x[sep:2*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))

    nsig = np.exp(x[2*sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    sigpdf = nsig[:,:,np.newaxis,:,:]*kernelpdf(scale, sigma, dataset, datasetGen)

    nll = nsig - np.sum(dataset*np.log(np.where(sigpdf>0.,sigpdf,1.)), axis =2)
        
    return np.sum(nll)

def nllbkg(x,nEtaBins,nPtBins,dataset,datasetGen):

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)
    
    scale = x[:sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.exp(x[sep:2*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))

    nsig = np.exp(x[2*sep:3*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    slope = x[3*sep:4*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    nbkg = np.exp(x[4*sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))

    sigpdf = nsig[:,:,np.newaxis,:,:]*kernelpdf(scale, sigma, dataset, datasetGen)
    bkgpdf = nbkg[:,:,np.newaxis,:,:]*exppdf(slope)

    nll = nsig+nbkg - np.sum(dataset*np.log(sigpdf+ bkgpdf), axis =2)
        
    return np.sum(nll)

def plots(x,nEtaBins,nPtBins,dataset,datasetGen):

    maxR = 3.3
    minR = 2.9

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)
    
    scale = x[:sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.exp(x[sep:2*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    nsig = np.exp(x[2*sep:3*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    n_true = np.sum(dataset,axis=2)
    
    sigpdf = nsig[:,:,np.newaxis,:,:]*kernelpdf(scale, sigma, dataset, datasetGen)

    pdf = sigpdf

    mass = np.linspace(minR,maxR,100)

    for ieta1 in range(nEtaBins):
        for ieta2 in range(nEtaBins):
            for ipt1 in range(nPtBins):
                for ipt2 in range(nPtBins):

                    if not pdf[ieta1,ieta2,:,ipt1,ipt2].any()>0.: continue

                    scale_bin = scale[ieta1,ieta2,ipt1,ipt2]
                    sigma_bin = sigma[ieta1,ieta2,ipt1,ipt2]
                    nsig_bin = nsig[ieta1,ieta2,ipt1,ipt2]
                    n_true_bin = n_true[ieta1,ieta2,ipt1,ipt2]

                    plt.clf()
                  
                    fig, (ax1, ax2) = plt.subplots(nrows=2)
                    ax1.errorbar(mass, dataset[ieta1,ieta2,:,ipt1,ipt2], yerr=np.sqrt(dataset[ieta1,ieta2,:,ipt1,ipt2]), fmt='.')
                    ax1.set_ylabel('number of events')
                    ax1.text(0.95, 0.95, 'scale: {:.5f}\n sigma: {:.3f}\n nsig: {:.0f}\n ntrue: {:.0f}'\
                        .format(scale_bin,sigma_bin,nsig_bin, n_true_bin),
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax1.transAxes,
                    color='black', fontsize=12)
                    
                    ax1.plot(mass, pdf[ieta1,ieta2,:,ipt1,ipt2])
                    plt.xlim(minR, maxR)
        
                    ax2.errorbar(mass,dataset[ieta1,ieta2,:,ipt1,ipt2]/pdf[ieta1,ieta2,:,ipt1,ipt2],yerr=np.sqrt(dataset[ieta1,ieta2,:,ipt1,ipt2])/pdf[ieta1,ieta2,:,ipt1,ipt2], fmt='.')
                    ax2.set_xlabel('dimuon mass')
                    ax2.set_ylabel('ratio data/pdf')
                    plt.xlim(minR, maxR)

                    plt.savefig('PLOTSJMC/plot_{}{}{}{}.pdf'.format(ieta1,ieta2,ipt1,ipt2))


def plotsbkg(x,nEtaBins,nPtBins,dataset,datasetGen):

    maxR = 3.3
    minR = 2.9

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)
    
    scale = x[:sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.exp(x[sep:2*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    nsig = np.exp(x[2*sep:3*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    slope = x[3*sep:4*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    nbkg = np.exp(x[4*sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    sigpdf = nsig[:,:,np.newaxis,:,:]*kernelpdf(scale, sigma, dataset, datasetGen)
    bkgpdf = nbkg[:,:,np.newaxis,:,:]*exppdf(slope)
    n_true = np.sum(dataset,axis=2)

    pdf = sigpdf+bkgpdf

    mass = np.linspace(minR,maxR,100)

    for ieta1 in range(nEtaBins):
        for ieta2 in range(nEtaBins):
            for ipt1 in range(nPtBins):
                for ipt2 in range(nPtBins):

                    if not pdf[ieta1,ieta2,:,ipt1,ipt2].any()>0.: continue

                    scale_bin = scale[ieta1,ieta2,ipt1,ipt2]
                    sigma_bin = sigma[ieta1,ieta2,ipt1,ipt2]
                    nsig_bin = nsig[ieta1,ieta2,ipt1,ipt2]
                    slope_bin = slope[ieta1,ieta2,ipt1,ipt2]
                    nbkg_bin = nbkg[ieta1,ieta2,ipt1,ipt2]
                    n_true_bin = n_true[ieta1,ieta2,ipt1,ipt2]


                    plt.clf()
                  
                    fig, (ax1, ax2) = plt.subplots(nrows=2)
                    ax1.errorbar(mass, dataset[ieta1,ieta2,:,ipt1,ipt2], yerr=np.sqrt(dataset[ieta1,ieta2,:,ipt1,ipt2]), fmt='.')
                    ax1.set_ylabel('number of events')
                    ax1.text(0.95, 0.95, 'scale: {:.5f}\n sigma: {:.3f}\n nsig: {:.0f}\n slope: {:.3f}\n nbkg: {:.0f} \n ntrue: {:.0f}'\
                        .format(scale_bin,sigma_bin,nsig_bin,slope_bin,nbkg_bin,n_true_bin),
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax1.transAxes,
                    color='black', fontsize=12)
                    
                    ax1.plot(mass, pdf[ieta1,ieta2,:,ipt1,ipt2])
                    ax1.plot(mass, bkgpdf[ieta1,ieta2,:,ipt1,ipt2], ls='--')
                    plt.xlim(minR, maxR)
        
                    ax2.errorbar(mass,dataset[ieta1,ieta2,:,ipt1,ipt2]/pdf[ieta1,ieta2,:,ipt1,ipt2],yerr=np.sqrt(dataset[ieta1,ieta2,:,ipt1,ipt2])/pdf[ieta1,ieta2,:,ipt1,ipt2], fmt='.')
                    ax2.set_xlabel('dimuon mass')
                    ax2.set_ylabel('ratio data/pdf')
                    plt.xlim(minR, maxR)

                    plt.savefig('PLOTSJDATA/plot_{}{}{}{}.pdf'.format(ieta1,ieta2,ipt1,ipt2))