import jax.numpy as np
import jax
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
import functools

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

    #A = np.ones((nEtaBins))
    #e = np.zeros((nEtaBins))
    #M = np.zeros((nEtaBins))
    #a = 0.07e-3*np.ones((nEtaBins))
    #b = 0.02e-3*np.ones((nEtaBins))
    #c = 1.5e-8*np.ones((nEtaBins))
    #d = 370.*np.ones((nEtaBins))
    
    
    A = np.zeros((nEtaBins))
    e = np.zeros((nEtaBins))
    M = np.zeros((nEtaBins))
    a = np.zeros((nEtaBins))
    #a = 0.07e-3*np.ones((nEtaBins))
    b = 0.02e-3*np.ones((nEtaBins))
    c = 1.5e-8*np.ones((nEtaBins))
    d = 370.*np.ones((nEtaBins))
    #nsig = ndata
    #nbkg = 0.01*ndata
    #nsig = np.log(ndata)
    #nbkg = np.sqrt(0.01*ndata)
    nsig = np.zeros_like(ndata)
    nbkg = np.sqrt(0.01)*np.ones_like(ndata)
    #slope = np.zeros_like(nbkg)
    slope = np.zeros_like(nbkg)
    
    #A = np.zeros((nEtaBins))
    #e = np.zeros((nEtaBins))
    #M = np.zeros((nEtaBins))
    #a = np.zeros((nEtaBins))
    ##nsig = 0.*np.ones_like(ndata)
    ##nbkg = 0.1*np.ones_like(ndata)
    #nsig = np.log(ndata)
    #nbkg = np.sqrt(0.01*ndata)

    if isJ:
        #x = np.concatenate((A, e, M, a, nsig),axis=0)
        x = np.concatenate((A, e, M, a, nsig, nbkg, slope),axis=0)
    else:
        x = np.concatenate((A, M, a, nsig),axis=0)

    print(x.shape, 'x.shape')
                
    return x.astype('float64')

def kernelpdf(scale, sigma, datasetGen, masses):

    valsMass = 0.5*(masses[:-1]+masses[1:])
    massWidth = masses[1:]-masses[:-1]
    #massWidth = massWidth[np.newaxis,:]
    massWidth = np.reshape(massWidth, len(scale.shape)*(1,) + (-1,))
    
    valsReco = np.reshape(valsMass, len(scale.shape)*(1,) + (-1,1))
    valsGen = np.reshape(valsMass, len(scale.shape)*(1,) + (1,-1))
    
    #valsReco = valsMass[np.newaxis,:,np.newaxis]
    #valsGen = valsMass[np.newaxis,np.newaxis,:]
    
    #scale_ext = scale[:,np.newaxis,np.newaxis]
    #sigma_ext = valsGen*sigma[:,np.newaxis,np.newaxis]
    
    scale_ext = np.reshape(scale, scale.shape + (1,1))
    sigma_ext = valsGen*np.reshape(sigma, sigma.shape + (1,1))

    h = scale_ext*valsGen

    datasetGen_ext = np.expand_dims(datasetGen,-2)

    #analytic integral
    #xscale = np.sqrt(2.)*sigma_ext

    #maxZ = (masses[-1]-h)/xscale
    #minZ = (masses[0]-h)/xscale
    
    #normarg = 0.5*(erf(maxZ)-erf(minZ))
    #I = datasetGen[:,np.newaxis,:]*normarg
    #I = np.sum(I, axis=-1)


    #contribution from each gen mass bin with correct relative normalization
    pdfarg = datasetGen_ext*np.exp(-np.square(valsReco  -h)/(2 * np.square(sigma_ext)))/sigma_ext/np.sqrt(2.*np.pi)
    #sum over gen mass bins
    pdf = np.sum(pdfarg,axis=-1)
    #numerical integration over reco mass bins
    I = np.sum(massWidth*pdf, axis=-1, keepdims=True)
    
    #print("kernelpdf debug")
    #print(np.any(np.isnan(pdfarg)), np.any(np.isnan(I)))
    pdf = pdf/np.where(pdf>0., I, 1.)

    return pdf

def kernelpdfPars(A, e, M, sigma, datasetGen, masses, etas, binCenters1, binCenters2, good_idx):

    #compute scale from physics parameters
    #etasC = (etas[:-1] + etas[1:]) / 2.

    #sEta = np.sin(2*np.arctan(np.exp(-etasC)))
    s1 = sEta[good_idx[0]]
    s2 = sEta[good_idx[1]]
    
    #c1 = binCenters1
    #c2 = binCenters2
    
    coeffe1 = binCenters1[...,1]
    coeffM1 = binCenters1[...,0]

    coeffe2 = binCenters2[...,1]
    coeffM2 = binCenters2[...,0]

    # select the model parameters from the eta bins corresponding
    # to each kinematic bin
    A1 = A[good_idx[0]]
    e1 = e[good_idx[0]]
    M1 = M[good_idx[0]]

    A2 = A[good_idx[1]]
    e2 = e[good_idx[1]]
    M2 = M[good_idx[1]]

    #term1 = A1-s1*e1*c1+M1/c1
    #term2 = A2-s2*e2*c2-M2/c2
    term1 = A1-e1*coeffe1+M1*coeffM1
    term2 = A2-e2*coeffe2-M2*coeffM2
    combos = term1*term2
    scale = np.sqrt(combos)
    
    return kernelpdf(scale, sigma, datasetGen, masses)

def computeTrackLength(eta):

    L0=108.-4.4 #max track length in cm. tracker radius - first pixel layer

    tantheta = 2/(np.exp(eta)-np.exp(-eta))
    r = 267.*tantheta #267 cm: z position of the outermost disk of the TEC 
    L = np.where(np.absolute(eta) <= 1.4, L0, (np.where(eta > 1.4, np.minimum(r, 108.)-4.4, np.minimum(-r, 108.)-4.4)))

    #print(L)
    return L
    
    #return np.ones((eta.shape[0]))


def exppdf(slope, masses):
    nBinsMass = masses.shape[0]
    
    valsMass = 0.5*(masses[:-1]+masses[1:])
    massWidth = masses[1:]-masses[:-1]
    #massWidth = massWidth[np.newaxis,:]
    massWidth = np.reshape(massWidth, len(slope.shape)*(1,) + (-1,))
    
    
    valsReco = np.reshape(valsMass, len(slope.shape)*(1,) + (-1,))
    
    slope_ext = np.expand_dims(slope,-1)/(masses[-1]-masses[0])
    
    #analytic integral
    #I = (np.exp(-slope_ext*masses[0]) - np.exp(-slope_ext*masses[-1]))/slope_ext
    #print(slope_ext.shape)
    #print(I.shape)
    #I = I[:,np.newaxis]
    
    #pdf = np.exp(-slope_ext*valsReco)
    pdf = np.exp(-slope_ext*(valsReco-masses[0]))
    #numerical integration over reco mass bins
    #I = np.sum(pdf, axis=-1, keepdims=True)
    I = np.sum(massWidth*pdf, axis=-1, keepdims=True)
    pdf = pdf/np.where(pdf>0.,I,1.)

    #print(I)
    #print(Ia)
    #print(slope_ext)
    #print(pdf[0])

    return pdf

def scaleSqFromModelPars(A, e, M, etas, binCenters1, binCenters2, good_idx, linearize=False):
    etasC = (etas[:-1] + etas[1:]) / 2.

    sEta = np.sin(2*np.arctan(np.exp(-etasC)))
    s1 = sEta[good_idx[0]]
    s2 = sEta[good_idx[1]]
    
    coeffe1 = binCenters1[...,1]
    coeffM1 = binCenters1[...,0]

    coeffe2 = binCenters2[...,1]
    coeffM2 = binCenters2[...,0]

    # select the model parameters from the eta bins corresponding
    # to each kinematic bin
    A1 = A[good_idx[0]]
    e1 = e[good_idx[0]]
    M1 = M[good_idx[0]]

    A2 = A[good_idx[1]]
    e2 = e[good_idx[1]]
    M2 = M[good_idx[1]]

    #term1 = A1-s1*e1*k1+M1/k1
    #term2 = A2-s2*e2*k2-M2/k2
    
    term1 = A1-e1*coeffe1+M1*coeffM1
    term2 = A2-e2*coeffe2-M2*coeffM2
    
    if linearize:
        #neglect quadratic term1*term2
        scaleSq = 1.+term1+term2
    else:
        scaleSq = (1.+term1)*(1.+term2)
        
    return scaleSq

def sigmaSqFromModelPars(a,b,c,d, etas, binCenters1, binCenters2, good_idx):
    
    #compute sigma from physics parameters
    etasC = (etas[:-1] + etas[1:]) / 2.
    
    #rough approximation for now
    #p1 = 1./binCenters1
    #p2 = 1./binCenters2
    
    coeffc1 = binCenters1[...,2]
    coeffb1 = binCenters1[...,3]
    
    coeffc2 = binCenters2[...,2]
    coeffb2 = binCenters2[...,3]
    

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

    #res1 = a1*np.power(l1,2) + c1*np.power(l1,4)*np.power(p1,2) + b1*np.power(l1,2)/(1+d1/(np.power(p1,2)*np.power(l1,2)))
    #res2 = a2*np.power(l2,2) + c2*np.power(l2,4)*np.power(p2,2) + b2*np.power(l2,2)/(1+d2/(np.power(p2,2)*np.power(l2,2)))
    
    #res1 = a1 + c1*np.power(p1,2) + b1/(1.+d1/(np.power(p1,2)*np.power(l1,2)))
    #res2 = a2 + c2*np.power(p2,2) + b2/(1.+d2/(np.power(p2,2)*np.power(l2,2)))
    
    res1 = a1 + c1*coeffc1 + b1*coeffb1
    res2 = a2 + c2*coeffc2 + b2*coeffb2

    sigmaSq = 0.25*(res1+res2)
    
    return sigmaSq

def chi2LBins(x, binScaleSq, binSigmaSq, hScaleSqSigmaSq, etas, binCenters1, binCenters2, good_idx):
    # return the gaussian likelihood (ie 0.5*chi2) from the scales and sigmas squared computed from the
    # physics model parameters vs the values and covariance matrix from the binned fit
    
    A,e,M,a,b,c,d = modelParsFromParVector(x)
    
    scaleSqModel = scaleSqFromModelPars(A,e,M,etas, binCenters1, binCenters2, good_idx, linearize=False)
    sigmaSqModel = sigmaSqFromModelPars(a,b,c,d,etas, binCenters1, binCenters2, good_idx)
    
    scaleSqSigmaSqModel = np.stack((scaleSqModel,sigmaSqModel), axis=-1)
    scaleSqSigmaSqBinned = np.stack((binScaleSq,binSigmaSq), axis=-1)
    
    diff = scaleSqSigmaSqModel-scaleSqSigmaSqBinned
    
    #batched column vectors
    diffcol = np.expand_dims(diff,-1)
    
    #batched row vectors
    diffcolT = np.expand_dims(diff,-2)

    #print("chi2 shapes")
    #print(diffcol.shape, diffcolT.shape, hScaleSqSigmaSq.shape)

    #batched matrix multiplication
    lbins = 0.5*np.matmul(diffcolT,np.matmul(hScaleSqSigmaSq, diffcol))
    
    return np.sum(lbins)
    
    #print(chi2bins.shape)
    
    #return chi2 
    
    
def chi2SumBins(x, binScaleSq, binSigmaSq, covScaleSqSigmaSq, etas, binCenters1, binCenters2, good_idx):
    chi2Binspartial = functools.partial(chi2Bins, etas=etas, binCenters1=binCenters1, binCenters2=binCenters2,good_idx=good_idx)
    chi2bins = jax.vmap(chi2Binspartial(x, binScaleSq, binSigmaSq, covScaleSqSigmaSq))
    chi2 = np.sum(chi2bins)
    return chi2
                        

def modelParsFromParVector(x):
    x = x.reshape((-1,5))
    
    A = x[...,0]
    e = x[...,1]
    M = x[...,2]
    a = x[...,3]
    c = x[...,4]
    #b = x[...,5]
    
    #c = np.zeros_like(a)
    b = np.zeros_like(a)
    d = 370.*np.ones_like(a)
    
    return A,e,M,a,b,c,d

def scaleSigmaFromModelParVector(x, etas, binCenters1, binCenters2, good_idx):
    A,e,M,a,b,c,d = modelParsFromParVector(x)
    return scaleSigmaFromPars(A, e, M, a, b, c, d, etas, binCenters1, binCenters2, good_idx)

def scaleFromModelPars(A, e, M, etas, binCenters1, binCenters2, good_idx):
    scaleSq = scaleSqFromModelPars(A, e, M, etas, binCenters1, binCenters2, good_idx, linearize=False)
    return np.sqrt(scaleSq)

def sigmaFromModelPars(a,b,c,d, etas, binCenters1, binCenters2, good_idx):
    
    sigmaSq = sigmaSqFromModelPars(a,b,c,d, etas, binCenters1, binCenters2, good_idx)
    return np.sqrt(sigmaSq)

def scaleSigmaFromPars(A, e, M, a, b, c, d, etas, binCenters1, binCenters2, good_idx):

    scale = scaleFromModelPars(A,e,M,etas, binCenters1, binCenters2, good_idx)
    sigma = sigmaFromModelPars(a,b,c,d, etas, binCenters1, binCenters2, good_idx)
    
    return scale,sigma

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

def splitTransformPars(x, ndata, nEtaBins, nBins, isJ=True):
    #A = x[:nEtaBins]
    A = 0.01*np.tanh(x[:nEtaBins])

    if isJ:
        #e = x[nEtaBins:2*nEtaBins]
        #M = x[2*nEtaBins:3*nEtaBins]
        #a = np.exp(x[3*nEtaBins:4*nEtaBins])
        #nsig = x[4*nEtaBins:4*nEtaBins+nBins]
        #nbkg = x[4*nEtaBins+nBins:]
        
        e = 0.01*np.tanh(x[nEtaBins:2*nEtaBins])
        M = 0.01*np.tanh(x[2*nEtaBins:3*nEtaBins])
        a = 1e-6 + 0.07e-3*np.exp(x[3*nEtaBins:4*nEtaBins])
        #a = 0.07e-3*np.exp(x[3*nEtaBins:4*nEtaBins])
        nsig = ndata*np.exp(x[4*nEtaBins:4*nEtaBins+nBins])
        nbkg = ndata*np.square(x[4*nEtaBins+nBins:4*nEtaBins+2*nBins])
        slope = x[4*nEtaBins+2*nBins:]

    else: 
        e = np.zeros((nEtaBins))
        M = x[nEtaBins:2*nEtaBins]
        a = x[2*nEtaBins:3*nEtaBins]
        nsig = np.exp(x[3*nEtaBins:])

    b = 0.03e-3*np.ones((nEtaBins))
    c = 15.e-9*np.ones((nEtaBins))
    d = 370.*np.ones((nEtaBins))
    
    return A,e,M,a,b,c,d,nsig,nbkg,slope
    

def scaleSqSigmaSqFromBinsPars(x):
    scale, sigma = scaleSigmaFromBinPars(x)
    return np.square(scale), np.square(sigma)

def scaleSigmaFromBinPars(x):
    #flexible on shape of input array as long as last dimension indexes the parameters within a bin
    scale = x[...,0]
    sigma = x[...,1]
    
    #parameter transformation for bounds
    #(since the bounds are reached only asymptotically, the fit will not converge well
    #if any of the parameters actually lie outside this region, these are just designed to protect
    #against pathological phase space during minimization)
    scale = 1. + 1e-2*np.tanh(scale)
    sigma = 5e-3*np.exp(2.*np.tanh(sigma))
    
    return scale, sigma
    
def bkgModelFromBinPars(x):
    #flexible on shape of input array as long as last dimension indexes the parameters within a bin
    fbkg = x[...,2]
    slope = x[...,3]
    
    # Transformation with hard bounds like in Minuit
    # Fit should still behave ok if bound is reached, though uncertainty
    # associated with this parameter will be underestimated
    fbkg = 0.5*(1.+np.sin(fbkg))
    
    return fbkg, slope
    
def nllBinsFromBinPars(x, dataset, datasetGen, masses):
    # For fitting with floating signal and background parameters
    
    scale, sigma = scaleSigmaFromBinPars(x)
    fbkg, slope = bkgModelFromBinPars(x)
    
    return nllBins(scale, sigma, fbkg, slope, dataset, datasetGen, masses)

def nllBinsFromSignalBinPars(x, fbkg, slope, dataset, datasetGen, masses):
    # For fitting with fixed background parametters
    scale, sigma = scaleSigmaFromBinPars(x)
    
    return nllBins(scale, sigma, fbkg, slope, dataset, datasetGen, masses)

def nllBins(scale, sigma, fbkg, slope, dataset, datasetGen, masses):
    sigpdf = kernelpdf(scale,sigma, datasetGen, masses)
    bkgpdf = exppdf(slope, masses)
    
    fbkg_ext = np.expand_dims(fbkg,-1)
    
    pdf = (1.-fbkg_ext)*sigpdf + fbkg_ext*bkgpdf
    
    nll = np.sum(-dataset*np.log(np.where(dataset>0., pdf, 1.)),axis=-1)
    
    #constraint on slope to keep fit well behaved when fbkg->0
    slopesigma = 5.
    nll += 0.5*np.square(slope)/slopesigma/slopesigma
    return nll

def nllParsSigma(x,nEtaBins,nPtBins,dataset,datasetGen, masses,isJ, etas, binCenters1, binCenters2, good_idx):
    
    ndata = np.sum(dataset,axis=-1)
    nBins = dataset.shape[0]

    #slope = np.zeros_like(nbkg)
    
    #nsig = ndata*nsig
    #nbkg = ndata*nbkg

    A,e,M,a,b,c,d,nsig,nbkg,slope = splitTransformPars(x, ndata, nEtaBins, nBins, isJ)
    
    scale,sigma = scaleSigmaFromPars(A, e, M, a, b, c, d, etas, binCenters1, binCenters2, good_idx)
    
    sigpdf = nsig[:,np.newaxis]*kernelpdf(scale, sigma, datasetGen, masses)
    bkgpdf = nbkg[:,np.newaxis]*exppdf(slope,masses)

    #TODO revisit this protection
    #nll = nsig - np.sum(dataset*np.log(np.where(dataset>0.,sigpdf,1.)), axis = -1)
    
    pdf = sigpdf + bkgpdf
    #print(np.min(pdf),np.min(sigpdf),np.min(bkgpdf))
    nll = nsig + nbkg - np.sum(dataset*np.log(np.where(dataset>0.,pdf,1.)), axis = -1)
    #nll = nsig + nbkg - np.sum(dataset*np.log(np.where(pdf>0.,pdf,1.)), axis = -1)
    
    nll = np.sum(nll)
    
    #add loose constraint on background slope to avoid issues when nbkg becomes very small or 0
    nll = nll + 0.5*np.sum(np.square(slope))/5./5.

    return nll

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
        
    maxR = masses[-1]
    minR = masses[0]

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


def plotsBkg(scale,sigma,fbkg,slope,dataset,datasetGen,masses,isJ,etas, binCenters1, binCenters2, good_idx):


    nBins = dataset.shape[0]
    ndata = np.sum(dataset,axis=-1)    
    minR = masses[0]
    maxR = masses[-1]
    massWidth = masses[1:]-masses[:-1]
    massWidth = massWidth[np.newaxis,:]
    
    masseslow = masses[:-1]
    
    nsig = (1.-fbkg)*ndata
    nbkg = fbkg*ndata
    
    sigpdf = nsig[:,np.newaxis]*massWidth*kernelpdf(scale, sigma, datasetGen, masses)
    bkgpdf = nbkg[:,np.newaxis]*massWidth*exppdf(slope,masses)

    pdf = sigpdf+bkgpdf


    #for ibin in range(nBins):
    for ibin in range(0,nBins,10):
        ieta1 = good_idx[0][ibin]
        ieta2 = good_idx[1][ibin]
        ipt1 = good_idx[2][ibin]
        ipt2 = good_idx[3][ibin]        
        
        scale_bin = scale[ibin]
        sigma_bin = sigma[ibin]
        fbkg_bin = fbkg[ibin]
        slope_bin = slope[ibin]
        n_true_bin = ndata[ibin]

        plt.clf()
        
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.errorbar(masseslow, dataset[ibin,:], yerr=np.sqrt(dataset[ibin,:]), fmt='.')
        ax1.set_ylabel('number of events')
        ax1.text(0.95, 0.95, 'scale: {:.5f}\n sigma: {:.3f}\n fbkg: {:.3f}\n slope: {:.3f}\n ntrue: {:.0f}'\
                        .format(scale_bin,sigma_bin,fbkg_bin,slope_bin,n_true_bin),
        verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=10)
        ax1.set_xlim([minR, maxR])
        
        ax1.plot(masseslow, pdf[ibin,:])
        ax1.plot(masseslow, bkgpdf[ibin,:], ls='--')
                
        ax2.errorbar(masseslow,dataset[ibin,:]/pdf[ibin,:],yerr=np.sqrt(dataset[ibin,:])/pdf[ibin,:], fmt='.')
        ax2.set_xlabel('dimuon mass')
        ax2.set_ylabel('ratio data/pdf')
        
        ax2.set_xlim([minR, maxR])
        ax2.set_ylim([0., 2.5])


        plt.savefig('PLOTS{}MCCorr/plot_{}{}{}{}.png'.format('J' if isJ else 'Z',ieta1,ieta2,ipt1,ipt2))
        plt.close(fig)


def plotsParsBkg(x,nEtaBins,nPtBins,dataset,datasetGen,masses,isJ,etas, binCenters1, binCenters2, good_idx):


    nBins = dataset.shape[0]
    ndata = np.sum(dataset,axis=-1)    
    minR = masses[0]
    maxR = masses[-1]
    massWidth = masses[1:]-masses[:-1]
    massWidth = massWidth[np.newaxis,:]
    
    masseslow = masses[:-1]


    A,e,M,a,b,c,d,nsig,nbkg,slope = splitTransformPars(x, ndata, nEtaBins, nBins, isJ)
    
    scale,sigma = scaleSigmaFromPars(A, e, M, a, b, c, d, etas, binCenters1, binCenters2, good_idx)
    
    sigpdf = nsig[:,np.newaxis]*massWidth*kernelpdf(scale, sigma, datasetGen, masses)
    bkgpdf = nbkg[:,np.newaxis]*massWidth*exppdf(slope,masses)

    pdf = sigpdf+bkgpdf


    #for ibin in range(nBins):
    for ibin in range(0,nBins,10):
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
        nbkg_bin = nbkg[ibin]
        n_true_bin = ndata[ibin]

        plt.clf()
        
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.errorbar(masseslow, dataset[ibin,:], yerr=np.sqrt(dataset[ibin,:]), fmt='.')
        ax1.set_ylabel('number of events')
        ax1.text(0.95, 0.95, 'A1: {:.5f}\n A2: {:.5f}\n e1: {:.4f}\n e2: {:.4f}\n M1: {:.9f}\n M2: {:.9f}\n\
            sigma: {:.3f}\n nsig: {:.0f}\n nbkg: {:.0f}\n ntrue: {:.0f}'\
            .format(A1_bin, A2_bin, e1_bin, e2_bin, M1_bin, M2_bin, sigma_bin,nsig_bin, nbkg_bin, n_true_bin),
        verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=10)
        ax1.set_xlim([minR, maxR])
        
        ax1.plot(masseslow, pdf[ibin,:])
        ax1.plot(masseslow, bkgpdf[ibin,:], ls='--')
                
        ax2.errorbar(masseslow,dataset[ibin,:]/pdf[ibin,:],yerr=np.sqrt(dataset[ibin,:])/pdf[ibin,:], fmt='.')
        ax2.set_xlabel('dimuon mass')
        ax2.set_ylabel('ratio data/pdf')
        
        ax2.set_xlim([minR, maxR])
        ax2.set_ylim([0., 2.5])


        plt.savefig('PLOTS{}MCCorr/plot_{}{}{}{}.png'.format('J' if isJ else 'Z',ieta1,ieta2,ipt1,ipt2))
        plt.close(fig)



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




