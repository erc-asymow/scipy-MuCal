import jax.numpy as np
from jax import grad, hessian, jacobian, config, random
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

def defineState(nEtaBins,nPtBins,dataset):
    
    nBins = dataset.shape[0]
    ndata = np.sum(dataset,axis=-1)

    scale = np.ones((nBins,),dtype='float64') #+ np.random.normal(0, 0.0005, (nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.full((nBins,),-4.9, dtype='float64')
    nsig = np.log(np.where(ndata>0.,ndata,2.))
        
    x = np.concatenate((scale,sigma,nsig),axis=0)
        
    print(x.shape, 'x.shape')
                
    return x.astype('float64')

def defineStatebkg(nEtaBins,nPtBins,dataset):

    nEtaBins = dataset.shape[0]
    ndata = np.sum(dataset,axis=-1)

    scale = np.ones((nBins,),dtype='float64') #+ np.random.normal(0, 0.0005, (nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.full((nBins,),-4.9, dtype='float64')
    nsig = np.log(np.where(ndata>0.,0.9*ndata,2.))
    slope = np.full((nBins),-0.1, dtype='float64') 
    nbkg = np.log(np.where(ndata>0.,0.1*ndata,2.))
    
    x = np.concatenate((scale.flatten(), sigma.flatten(), nsig.flatten(),slope.flatten(),nbkg.flatten()),axis=0)
    
    print(x.shape, 'x.shape')
                
    return x.astype('float64')

def defineStatePars(nEtaBins,nPtBins,dataset, isJ):
    
    nBins = dataset.shape[0]
    ndata = np.sum(dataset,axis=-1)    

    A = np.ones((nEtaBins))
    e = np.zeros((nEtaBins))
    M = np.zeros((nEtaBins))
    sigma = np.full((nBins,),-4.9, dtype='float64')
    nsig = np.log(np.where(ndata>0.,ndata,2.))

    if isJ:
        x = np.concatenate((A, e, M, sigma, nsig),axis=0)
    else:
        x = np.concatenate((A, M, sigma, nsig),axis=0)

    print(x.shape, 'x.shape')
                
    return x.astype('float64')

def defineStateParsSigma(nEtaBins,nPtBins,dataset, isJ):

    nBins = dataset.shape[0]
    ndata = np.sum(dataset,axis=-1)    

    A = np.ones((nEtaBins))
    e = np.zeros((nEtaBins))
    M = np.zeros((nEtaBins))
    a = 0.07e-3*np.ones((nEtaBins))
    #b = 0.02e-3*np.ones((nEtaBins))
    #c = 1.5e-8*np.ones((nEtaBins))
    #d = 370.*np.ones((nEtaBins))
    nsig = np.log(np.where(ndata>0.,ndata,2.))

    if isJ:
        x = np.concatenate((A, e, M, a, nsig),axis=0)
    else:
        x = np.concatenate((A, M, a, nsig),axis=0)

    print(x.shape, 'x.shape')
                
    return x.astype('float64')

def kernelpdf(scale, sigma, datasetGen, masses):

    valsMass = 0.5*(masses[:-1]+masses[1:])
    massWidth = masses[1:]-masses[:-1]
    massWidth = massWidth[np.newaxis,:]

    
    valsReco = valsMass[np.newaxis,:,np.newaxis]
    valsGen = valsMass[np.newaxis,np.newaxis,:]
    
    scale_ext = scale[:,np.newaxis,np.newaxis]
    sigma_ext = valsGen*sigma[:,np.newaxis,np.newaxis]

    h = scale_ext*valsGen
    xscale = np.sqrt(2.)*sigma_ext

    #contribution from each gen mass bin with correct relative normalization
    pdfarg = datasetGen[:,np.newaxis,:]*np.exp(-np.power(valsReco  -h, 2.)/(2 * np.power(sigma_ext, 2.)))/sigma_ext/np.sqrt(2.*np.pi)
    #sum over gen mass bins
    pdf = np.sum(pdfarg,axis=-1)
    #numerical integration over reco mass bins
    I = np.sum(massWidth*pdf, axis=-1, keepdims=True)

    pdf = pdf/I

    return pdf

def kernelpdfPars(A, e, M, sigma, datasetGen, masses, etas, binCenters1, binCenters2, good_idx):

    #compute scale from physics parameters
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
    
    return kernelpdf(scale, sigma, datasetGen, masses)

def computeTrackLength(eta):

    L0=108.-4.4 #max track length in cm. tracker radius - first pixel layer

    tantheta = 2/(np.exp(eta)-np.exp(-eta))
    r = 267.*tantheta #267 cm: z position of the outermost disk of the TEC 
    #L = np.where(np.absolute(eta) <= 1.4, L0, (np.where(eta > 1.4, min(r, 108.)-4.4, min(-r, 108.)-4.4)))

    #print(L)
    
    return np.ones((eta.shape[0]))

def kernelpdfParsSigma(A, e, M, a, b, c, d, datasetGen, masses, etas, binCenters1, binCenters2, good_idx):

    #compute scale from physics parameters
    etasC = (etas[:-1] + etas[1:]) / 2.

    sEta = np.sin(2*np.arctan(np.exp(-etasC)))
    s1 = sEta[good_idx[0]]
    s2 = sEta[good_idx[1]]
    
    k1 = binCenters1
    k2 = binCenters2

    # select the model parameters from the eta bins corresponding
    # to each kinematic bin
    A1 = A[good_idx[0]]
    e1 = e[good_idx[0]]
    M1 = M[good_idx[0]]

    A2 = A[good_idx[1]]
    e2 = e[good_idx[1]]
    M2 = M[good_idx[1]]

    term1 = A1-s1*e1*k1+M1/k1
    term2 = A2-s2*e2*k2-M2/k2
    combos = term1*term2
    scale = np.sqrt(combos)

    #rough approximation for now
    p1 = 1./binCenters1
    p2 = 1./binCenters2

    L = computeTrackLength(etasC)
    l1 = L[good_idx[0]]
    l2 = L[good_idx[1]]

    a1 = a[good_idx[0]]
    b1 = b[good_idx[0]]
    c1 = c[good_idx[0]]
    d1 = d[good_idx[0]]

    a2 = a[good_idx[1]]
    b2 = b[good_idx[1]]
    c2 = c[good_idx[1]]
    d2 = d[good_idx[1]]

    res1 = a1*np.power(l1,2) #+ c1*np.power(l1,4)*np.power(p1,2) + b1*np.power(l1,2)/(1+d1/(np.power(p1,2)*np.power(l1,2)))
    res2 = a2*np.power(l2,2) #+ c2*np.power(l2,4)*np.power(p2,2) + b2*np.power(l2,2)/(1+d2/(np.power(p2,2)*np.power(l2,2)))

    sigma = 1./2.*np.sqrt(res1+res2)
    
    return kernelpdf(scale, sigma, datasetGen, masses)

def nll(x,nEtaBins,nPtBins,dataset,datasetGen, masses):

    nBins = dataset.shape[0]
    
    scale = x[:nBins]
    sigma = np.exp(x[nBins:2*nBins])

    nsig = np.exp(x[2*nBins:])
    sigpdf = nsig[:,np.newaxis]*kernelpdf(scale, sigma, datasetGen, masses)

    #TODO revisit this protection
    nll = nsig - np.sum(dataset*np.log(np.where(dataset>0.,sigpdf,1.)), axis =-1)
        
    return np.sum(nll)

def nllbkg(x,nEtaBins,nPtBins,dataset,datasetGen, masses):

    nBins = dataset.shape[0]
    
    scale = x[:nBins]
    sigma = np.exp(x[nBins:2*nBins])

    nsig = np.exp(x[2*nBins:])
    slope = x[3*nBins:4*nBins]
    nbkg = np.exp(x[4*nBins:])

    sigpdf = nsig[:,np.newaxis]*kernelpdf(scale, sigma, datasetGen, masses)
    bkgpdf = nbkg[:,np.newaxis]*exppdf(slope)

    #TODO revisit this protection
    nll = nsig+nbkg - np.sum(dataset*np.log(np.where(dataset>0.,sigpdf+bkgpdf,1.)), axis =-1)
        
    return np.sum(nll)

def nllPars(x,nEtaBins,nPtBins,dataset,datasetGen, masses,isJ, etas, binCenters1, binCenters2, good_idx):

    nBins = dataset.shape[0]

    A = x[:nEtaBins]

    if isJ:
        e = x[nEtaBins:2*nEtaBins]
        M = x[2*nEtaBins:3*nEtaBins]
        sigma = np.exp(x[3*nEtaBins:3*nEtaBins+nBins])
        nsig = np.exp(x[3*nEtaBins+nBins:])
    else: 
        e = np.zeros((nEtaBins))
        M = x[nEtaBins:2*nEtaBins]
        sigma = np.exp(x[2*nEtaBins:2*nEtaBins+nBins])
        nsig = np.exp(x[2*nEtaBins+nBins:])
    
    sigpdf = nsig[:,np.newaxis]*kernelpdfPars(A, e, M, sigma, datasetGen, masses, etas, binCenters1, binCenters2, good_idx)

    #TODO revisit this protection
    nll = nsig - np.sum(dataset*np.log(np.where(dataset>0.,sigpdf,1.)), axis = -1)

    return np.sum(nll)

def nllParsSigma(x,nEtaBins,nPtBins,dataset,datasetGen, masses,isJ, etas, binCenters1, binCenters2, good_idx):

    nBins = dataset.shape[0]

    A = x[:nEtaBins]

    if isJ:
        e = x[nEtaBins:2*nEtaBins]
        M = x[2*nEtaBins:3*nEtaBins]
        a = x[3*nEtaBins:4*nEtaBins]
        nsig = np.exp(x[4*nEtaBins:])
    else: 
        e = np.zeros((nEtaBins))
        M = x[nEtaBins:2*nEtaBins]
        a = x[2*nEtaBins:3*nEtaBins]
        nsig = np.exp(x[3*nEtaBins:])

    b = 0.03e-3*np.ones((nEtaBins))
    c = 15.e-9*np.ones((nEtaBins))
    d = 370.*np.ones((nEtaBins))
    
    sigpdf = nsig[:,np.newaxis]*kernelpdfParsSigma(A, e, M, a, b, c, d, datasetGen, masses, etas, binCenters1, binCenters2, good_idx)

    #TODO revisit this protection
    nll = nsig - np.sum(dataset*np.log(np.where(dataset>0.,sigpdf,1.)), axis = -1)

    return np.sum(nll)

def plots(x,nEtaBins,nPtBins,dataset,datasetGen,masses,isJ, good_idx):

    if isJ:
        maxR = 3.3
        minR = 2.9
    else:
        maxR = 75.
        minR = 115.

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)
    nBins = dataset.shape[0]
    
    scale = x[:nBins]
    sigma = np.exp(x[nBins:2*nBins])
    nsig = np.exp(x[2*nBins:3*nBins])
    n_true = np.sum(dataset,axis=-1)
    
    sigpdf = nsig[:,np.newaxis]*kernelpdf(scale, sigma, datasetGen, masses)

    pdf = sigpdf

    mass = np.linspace(minR,maxR,100)

    for ibin in range(nBins):

        ieta1 = good_idx[0][ibin]
        ieta2 = good_idx[1][ibin]
        ipt1 = good_idx[2][ibin]
        ipt2 = good_idx[3][ibin]     
        
        scale_bin = scale[ibin]
        sigma_bin = sigma[ibin]
        nsig_bin = nsig[ibin]
        n_true_bin = n_true[ibin]

        plt.clf()
        
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.errorbar(mass, dataset[ibin,:], yerr=np.sqrt(dataset[ibin,:]), fmt='.')
        ax1.set_ylabel('number of events')
        ax1.text(0.95, 0.95, 'scale: {:.5f}\n sigma: {:.3f}\n nsig: {:.0f}\n ntrue: {:.0f}'\
            .format(scale_bin,sigma_bin,nsig_bin, n_true_bin),
        verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=12)
        
        ax1.plot(mass, pdf[ibin,:])
        ax1.set_xlim([minR, maxR])

        ax2.errorbar(mass,dataset[ibin,:]/pdf[ibin,:],yerr=np.sqrt(dataset[ibin,:])/pdf[ibin,:], fmt='.')
        ax2.set_xlabel('dimuon mass')
        ax2.set_ylabel('ratio data/pdf')
        
        ax2.set_xlim([minR, maxR])
        ax2.set_ylim([0., 2.5])

        plt.savefig('PLOTS{}MC/plot_{}{}{}{}.pdf'.format('J' if isJ else 'Z',ieta1,ieta2,ipt1,ipt2))
        plt.close(fig)

#TODO this function needs to be updated to the new binning scheme
def plotsbkg(x,nEtaBins,nPtBins,dataset,datasetGen,masses, isJ):

    if isJ:
        maxR = 3.3
        minR = 2.9
    else:
        maxR = 75.
        minR = 115.

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)
    
    scale = x[:sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.exp(x[sep:2*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    nsig = np.exp(x[2*sep:3*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    slope = x[3*sep:4*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    nbkg = np.exp(x[4*sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    sigpdf = nsig[:,:,np.newaxis,:,:]*kernelpdf(scale, sigma, datasetGen, masses)
    bkgpdf = nbkg[:,:,np.newaxis,:,:]*exppdf(slope)
    n_true = np.sum(dataset,axis=-1)

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

                    plt.savefig('PLOTS{}DATA/plot_{}{}{}{}.pdf'.format('J' if isJ else 'Z',ieta1,ieta2,ipt1,ipt2))


def plotsPars(x,nEtaBins,nPtBins,dataset,datasetGen,masses,isJ,etas, binCenters1, binCenters2, good_idx):

    if isJ:
        maxR = 3.3
        minR = 2.9
    else:
        maxR = 75.
        minR = 115.

    nBins = dataset.shape[0]

    A = x[:nEtaBins]

    if isJ:
        e = x[nEtaBins:2*nEtaBins]
        M = x[2*nEtaBins:3*nEtaBins]
        sigma = np.exp(x[3*nEtaBins:3*nEtaBins+nBins])
        nsig = np.exp(x[3*nEtaBins+nBins:])
    else: 
        e = np.zeros((nEtaBins))
        M = x[nEtaBins:2*nEtaBins]
        sigma = np.exp(x[2*nEtaBins:2*nEtaBins+nBins])
        nsig = np.exp(x[2*nEtaBins+nBins:])

    n_true = np.sum(dataset,axis=-1)    
    
    sigpdf = nsig[:,np.newaxis]*kernelpdfPars(A, e, M, sigma, datasetGen,masses,etas, binCenters1, binCenters2, good_idx)

    pdf = sigpdf

    mass = np.linspace(minR,maxR,100)

    for ibin in range(nBins):
        ieta1 = good_idx[0][ibin]
        ieta2 = good_idx[1][ibin]
        ipt1 = good_idx[2][ibin]
        ipt2 = good_idx[3][ibin]        
        
        A1_bin = A[ieta1]
        A2_bin = A[ieta2]
        e1_bin = e[ieta1]
        e2_bin = e[ieta2]
        M1_bin = M[ieta1]
        M2_bin = M[ieta2]

        sigma_bin = sigma[ibin]
        nsig_bin = nsig[ibin]
        n_true_bin = n_true[ibin]

        plt.clf()
        
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.errorbar(mass, dataset[ibin,:], yerr=np.sqrt(dataset[ibin,:]), fmt='.')
        ax1.set_ylabel('number of events')
        ax1.text(0.95, 0.95, 'A1: {:.5f}\n A2: {:.5f}\n e1: {:.4f}\n e2: {:.4f}\n M1: {:.9f}\n M2: {:.9f}\n\
            sigma: {:.3f}\n nsig: {:.0f}\n ntrue: {:.0f}'\
            .format(A1_bin, A2_bin, e1_bin, e2_bin, M1_bin, M2_bin, sigma_bin,nsig_bin, n_true_bin),
        verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=12)
        ax1.set_xlim([minR, maxR])
        
        ax1.plot(mass, pdf[ibin,:])
                
        ax2.errorbar(mass,dataset[ibin,:]/pdf[ibin,:],yerr=np.sqrt(dataset[ibin,:])/pdf[ibin,:], fmt='.')
        ax2.set_xlabel('dimuon mass')
        ax2.set_ylabel('ratio data/pdf')
        
        ax2.set_xlim([minR, maxR])
        ax2.set_ylim([0., 2.5])


        plt.savefig('PLOTS{}MCCorr/plot_{}{}{}{}.png'.format('J' if isJ else 'Z',ieta1,ieta2,ipt1,ipt2))
        plt.close(fig)



