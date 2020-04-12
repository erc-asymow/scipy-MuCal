import jax.numpy as np
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

def defineState(nEtaBins,nPtBins,dataset):

    scale = np.ones((nEtaBins,nEtaBins,nPtBins,nPtBins)) #+ np.random.normal(0, 0.0005, (nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-3.9) 
    nsig = np.log(np.where(np.sum(dataset,axis=2)>0.,np.sum(dataset,axis=2),2.))
        
    x = np.concatenate((scale.flatten(), sigma.flatten(), nsig.flatten()),axis=0)
        
    print x.shape, 'x.shape'
                
    return x.astype('float64')

def defineStatebkg(nEtaBins,nPtBins,dataset):

    scale = np.ones((nEtaBins,nEtaBins,nPtBins,nPtBins)) #+ np.random.normal(0, 0.0005, (nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-3.9) 
    nsig = np.log(np.where(np.sum(dataset,axis=2)>0.,0.9*np.sum(dataset,axis=2),2.))
    slope = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-0.1) 
    nbkg = np.log(np.where(np.sum(dataset,axis=2)>0.,0.1*np.sum(dataset,axis=2),2.))
    
    x = np.concatenate((scale.flatten(), sigma.flatten(), nsig.flatten(),slope.flatten(),nbkg.flatten()),axis=0)
    
    print x.shape, 'x.shape'
                
    return x.astype('float64')

def defineStatePars(nEtaBins,nPtBins,dataset, isJ):

    A = np.ones((nEtaBins)) #+ np.random.normal(0, 0.0005, (nEtaBins))
    e = np.zeros((nEtaBins)) #+ np.random.normal(0, 0.005, (nEtaBins))
    M = np.zeros((nEtaBins)) #+ np.random.normal(0, 0.000005, (nEtaBins))
    sigma = np.full((nEtaBins,nEtaBins,nPtBins,nPtBins),-3.9) 
    nsig = np.log(np.where(np.sum(dataset,axis=2)>0,np.sum(dataset,axis=2),2))

    if isJ:
        x = np.concatenate((A.flatten(), e.flatten(), M.flatten(), sigma.flatten(), nsig.flatten()),axis=0)
    else:
        x = np.concatenate((A.flatten(), M.flatten(), sigma.flatten(), nsig.flatten()),axis=0)

    print x.shape, 'x.shape'
                
    return x.astype('float64')

def kernelpdf(scale, sigma, dataset, datasetGen, isJ):

    #dataset is binned as eta1,eta2,mass,pt2,pt1

    if isJ:
        maxR = np.full((100),3.3)
        minR = np.full((100),2.9)
    else:
        maxR = np.full((100),75.)
        minR = np.full((100),115.)

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

def kernelpdfPars(A, e, M, sigma, dataset, datasetGen, isJ, etas, binCenters1, binCenters2):

    #dataset is binned as eta1,eta2,mass,pt2,pt1

    if isJ:
        maxR = np.full((100),3.3)
        minR = np.full((100),2.9)
    else:
        maxR = np.full((100),75.)
        minR = np.full((100),115.)  

    valsReco = np.linspace(minR[0],maxR[0],100)
    valsGen = valsReco

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

    if isJ:
        maxR = 3.3
        minR = 2.9
    else:
        maxR = 75.
        minR = 115.

    valsReco = np.linspace(minR,maxR,100)

    I = (np.exp(-slope*minR) - np.exp(-slope*maxR))/slope

    massbinwidth = (maxR-minR)/100

    h = np.tensordot(slope,valsReco,axes=0) 
    h_ext = np.swapaxes(np.swapaxes(h,2,4),3,4)

    pdf = np.exp(-h_ext)/I

    return pdf*massbinwidth


def nll(x,nEtaBins,nPtBins,dataset,datasetGen, isJ):

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)
    
    scale = x[:sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins))
    sigma = np.exp(x[sep:2*sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))

    nsig = np.exp(x[2*sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    sigpdf = nsig[:,:,np.newaxis,:,:]*kernelpdf(scale, sigma, dataset, datasetGen, isJ)

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

    nll = nsig+nbkg - np.sum(dataset*np.log(np.where(sigpdf>0.,sigpdf+bkgpdf,1.)), axis =2)
        
    return np.sum(nll)

def nllPars(x,nEtaBins,nPtBins,dataset,datasetGen, isJ, etas, binCenters1, binCenters2):

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)

    A = x[:nEtaBins]

    if isJ:
        e = x[nEtaBins:2*nEtaBins]
        M = x[2*nEtaBins:3*nEtaBins]
        sigma = np.exp(x[3*nEtaBins:3*nEtaBins+sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
        nsig = np.exp(x[3*nEtaBins+sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    else: 
        e = np.zeros((nEtaBins))
        M = x[nEtaBins:2*nEtaBins]
        sigma = np.exp(x[2*nEtaBins:2*nEtaBins+sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
        nsig = np.exp(x[2*nEtaBins+sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))

    
    sigpdf = nsig[:,:,np.newaxis,:,:]*kernelpdfPars(A, e, M, sigma, dataset, datasetGen, isJ, etas, binCenters1, binCenters2)

    nll = nsig - np.sum(dataset*np.log(np.where(sigpdf>0.,sigpdf,1.)), axis =2)
        
    return np.sum(nll)

def plots(x,nEtaBins,nPtBins,dataset,datasetGen,isJ):

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
    n_true = np.sum(dataset,axis=2)
    
    sigpdf = nsig[:,:,np.newaxis,:,:]*kernelpdf(scale, sigma, dataset, datasetGen,isJ)

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

                    plt.savefig('PLOTS{}MC/plot_{}{}{}{}.pdf'.format('J' if isJ else 'Z',ieta1,ieta2,ipt1,ipt2))
                    plt.close(fig)



def plotsbkg(x,nEtaBins,nPtBins,dataset,datasetGen,isJ):

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

                    plt.savefig('PLOTS{}DATA/plot_{}{}{}{}.pdf'.format('J' if isJ else 'Z',ieta1,ieta2,ipt1,ipt2))


def plotsPars(x,nEtaBins,nPtBins,dataset,datasetGen,isJ):

    if isJ:
        maxR = 3.3
        minR = 2.9
    else:
        maxR = 75.
        minR = 115.

    sep = np.power(nEtaBins,2)*np.power(nPtBins,2)

    A = x[:nEtaBins,np.newaxis]

    if isJ:
        e = x[nEtaBins:2*nEtaBins]
        M = x[2*nEtaBins:3*nEtaBins]
        sigma = np.exp(x[3*nEtaBins:3*nEtaBins+sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
        nsig = np.exp(x[3*nEtaBins+sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
    else: 
        e = np.zeros((nEtaBins))
        M = x[nEtaBins:2*nEtaBins]
        sigma = np.exp(x[2*nEtaBins:2*nEtaBins+sep].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))
        nsig = np.exp(x[2*nEtaBins+sep:].reshape((nEtaBins,nEtaBins,nPtBins,nPtBins)))

    n_true = np.sum(dataset,axis=2)    
    
    sigpdf = nsig[:,:,np.newaxis,:,:]*kernelpdfPars(A, e, M, sigma, dataset, datasetGen,isJ)

    pdf = sigpdf

    mass = np.linspace(minR,maxR,100)

    for ieta1 in range(nEtaBins):
        for ieta2 in range(nEtaBins):
            for ipt1 in range(nPtBins):
                for ipt2 in range(nPtBins):

                    if not pdf[ieta1,ieta2,:,ipt1,ipt2].any()>0.: continue

                    A1_bin = A[ieta1,0]
                    A2_bin = A[ieta2,0]
                    e1_bin = e[ieta1]
                    e2_bin = e[ieta2]
                    M1_bin = M[ieta1]
                    M2_bin = M[ieta2]

                    sigma_bin = sigma[ieta1,ieta2,ipt1,ipt2]
                    nsig_bin = nsig[ieta1,ieta2,ipt1,ipt2]
                    n_true_bin = n_true[ieta1,ieta2,ipt1,ipt2]

                    plt.clf()
                  
                    fig, (ax1, ax2) = plt.subplots(nrows=2)
                    ax1.errorbar(mass, dataset[ieta1,ieta2,:,ipt1,ipt2], yerr=np.sqrt(dataset[ieta1,ieta2,:,ipt1,ipt2]), fmt='.')
                    ax1.set_ylabel('number of events')
                    ax1.text(0.95, 0.95, 'A1: {:.5f}\n A2: {:.5f}\n e1: {:.4f}\n e2: {:.4f}\n M1: {:.9f}\n M2: {:.9f}\n\
                        sigma: {:.3f}\n nsig: {:.0f}\n ntrue: {:.0f}'\
                        .format(A1_bin, A2_bin, e1_bin, e2_bin, M1_bin, M2_bin, sigma_bin,nsig_bin, n_true_bin),
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax1.transAxes,
                    color='black', fontsize=12)
                    ax1.set_xlim([minR, maxR])
                    
                    ax1.plot(mass, pdf[ieta1,ieta2,:,ipt1,ipt2])
                            
                    ax2.errorbar(mass,dataset[ieta1,ieta2,:,ipt1,ipt2]/pdf[ieta1,ieta2,:,ipt1,ipt2],yerr=np.sqrt(dataset[ieta1,ieta2,:,ipt1,ipt2])/pdf[ieta1,ieta2,:,ipt1,ipt2], fmt='.')
                    ax2.set_xlabel('dimuon mass')
                    ax2.set_ylabel('ratio data/pdf')
                    
                    ax2.set_xlim([minR, maxR])
                    ax2.set_ylim([0., 2.5])


                    plt.savefig('PLOTS{}MCCorr/plot_{}{}{}{}.pdf'.format('J' if isJ else 'Z',ieta1,ieta2,ipt1,ipt2))
                    plt.close(fig)



