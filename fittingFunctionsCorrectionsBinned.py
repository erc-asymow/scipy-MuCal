from autograd import grad, hessian
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



def roll1Dto2D(bin2D, nPhiBins):

    return bin2D/nPhiBins, bin2D%nPhiBins

def defineState(nEtaBins,nPhiBins,dataset):

    #input values for the parameters
    A = np.ones((nEtaBins))+ np.random.normal(0, 0.00005, nEtaBins)
    e = np.zeros((nEtaBins))+ np.random.normal(0, 0.0001, nEtaBins)
    M = np.zeros((nEtaBins*nPhiBins)) + np.random.normal(0, 0.0000001, (nEtaBins*nPhiBins))
    
    count = 0
    for idx in dataset:
        if not dataset[idx]["smearedgenMass"].shape[0]<1:
            count+=1
            
            
    scale = np.full(count,1.) + np.random.normal(0, 0.0005, count)
    sigma = np.full(count,0.6) + np.random.normal(0, 0.1, count)
    
    #fsig = np.full(count,10000) #+ np.random.normal(0, 0.05, 1)
    #tau = np.full(count,-1.) + np.random.normal(0, 0.5, 1)
    
    #x = np.concatenate((A,e,M,fsig,tau), axis=None)
   
    x = np.concatenate((A,M,sigma), axis=None)
                
    return x.astype("float64")

def nll(x,nEtaBins,i,j,idx,dataset,datasetGen):

    z = dataset[(i,j)]["smearedgenMass"]
    zGen = datasetGen[(i,j)]["genMass"]
    
    ieta1, _ = roll1Dto2D(i,1)
    ieta2, _ = roll1Dto2D(j,1)
    
    #retrieve parameter value

    A1 = x[ieta1]
    A2 = x[ieta2]
    M1 = x[nEtaBins+i]
    M2 = x[nEtaBins+j]

    c1Counts = np.histogram(dataset[(i,j)]["c1"], bins=100, range=(1./100.,1./20))[0]
    c1Vals = np.linspace(1./100.,1./20,100)

    c2Counts = np.histogram(dataset[(i,j)]["c2"], bins=100, range=(1./100.,1./20))[0]
    c2Vals = np.linspace(1./100.,1./20,100)

    #bin c1 and c2

    term1 = A1+M1/c1Vals
    term2 = A2-M2/c2Vals

    #bin the genMass
    
    genMass = np.histogram(zGen, bins=100, range=(80.,100.))[0]
    vals = np.linspace(80.,100.,100)

    h=np.outer(np.sqrt(term1*term2),vals)
    sigma = x[2*nEtaBins+idx] #only valid if integrating over phi
        
    counts = np.histogram(z, bins=100, range=(80.,100.))[0]
    mass = np.linspace(80.,100.,100)

    mass_ext = mass[:,np.newaxis]
    
    xscale = np.sqrt(2.)*sigma
    maxZ = ((100.-h.astype('float64'))/xscale)
    minZ = ((80.-h.astype('float64'))/xscale)

    I = np.sum(genMass*np.sqrt(np.pi/2.)*sigma*(erf(maxZ)-erf(minZ)),axis=1)/np.sum(genMass)
    
    pdf = z.shape[0]*np.sum(genMass*np.exp(-np.power(mass_ext  -h.astype('float64'), 2.)/(2 * np.power(sigma, 2.)))/I/np.sum(genMass),axis=1)
    nll = np.sum(- poisson.logpmf(counts, pdf ) )
    #print nll
    
    return nll


def nllSimul(x, nEtaBins, datasetJ, datasetZ, datasetJGen, datasetZGen):

    l = np.float64(0.)

    idx2D = 0
   
    for idx in datasetZ:

        #if not idx==(0,0): continue
         
        if datasetZ[idx]["smearedgenMass"].shape[0]<1:
            continue

        i = idx[0]
        j = idx[1]

        
        l+=nll(x,nEtaBins,i,j,idx2D,datasetZ,datasetZGen)
        idx2D+=1

    return l


def plots(x,nEtaBins,dataset,datasetGen):

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    idx2D = 0
    for idx in dataset:

        i = idx[0] #first muon
        j = idx[1] #second muon

        if dataset[idx]["smearedgenMass"].shape[0]<1:
            continue

        z = dataset[idx]["smearedgenMass"]
        zGen = datasetGen[(i,j)]["genMass"]
       
        ieta1, _ = roll1Dto2D(i,1)
        ieta2, _ = roll1Dto2D(j,1)
    
        #retrieve parameter value

        A1 = x[ieta1]
        A2 = x[ieta2]
        M1 = x[nEtaBins+i]
        M2 = x[nEtaBins+j]

        c1Counts = np.histogram(dataset[(i,j)]["c1"], bins=100, range=(1./100.,1./20))[0]
        c1Vals = np.linspace(1./100.,1./20,100)

        c2Counts = np.histogram(dataset[(i,j)]["c2"], bins=100, range=(1./100.,1./20))[0]
        c2Vals = np.linspace(1./100.,1./20,100)

        #bin c1 and c2

        term1 = A1+M1/c1Vals
        term2 = A2-M2/c2Vals

        #bin the genMass
    
        genMass = np.histogram(zGen, bins=100, range=(80.,100.))[0]
        vals = np.linspace(80.,100.,100)

        h=np.outer(np.sqrt(term1*term2),vals)
        sigma = x[2*nEtaBins+idx2D] #only valid if integrating over phi
        
        counts = np.histogram(z, bins=100, range=(80.,100.))[0]
        mass = np.linspace(80.,100.,100)

        mass_ext = mass[:,np.newaxis]
    
        xscale = np.sqrt(2.)*sigma
        maxZ = ((100.-h.astype('float64'))/xscale)
        minZ = ((80.-h.astype('float64'))/xscale)

        I = np.sum(genMass*np.sqrt(np.pi/2.)*sigma*(erf(maxZ)-erf(minZ)),axis=1)/np.sum(genMass)
    
        pdf = z.shape[0]*np.sum(genMass*np.exp(-np.power(mass_ext  -h.astype('float64'), 2.)/(2 * np.power(sigma, 2.)))/I/np.sum(genMass),axis=1)


        plt.clf()
        w = (100.-80.)/100
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.errorbar(mass, counts, yerr=np.sqrt(counts), fmt='.')
        ax1.plot(mass, pdf*w)
        plt.xlim(80, 100)
        
        ax2.errorbar(mass,counts/(pdf*w),yerr=np.sqrt(counts)/(pdf*w), fmt='.')
        
        plt.xlim(80, 100)

        plt.savefig("plotZcorrections_{}{}.pdf".format(i,j))

        idx2D+=1


fileZ = open("calInputZMCsm.pkl", "rb")
datasetZ = pickle.load(fileZ)
fileZgen = open("calInputZMCgen.pkl", "rb")
datasetZgen = pickle.load(fileZgen)

etas = np.arange(-0.8, 1.2, 0.4)
#phis = np.arange(-np.pi, np.pi+2.*np.pi/6.,2.*np.pi/6.)
#etas = np.array((-0.8,-0.4))
phis = np.array((-np.pi,np.pi))


x = defineState(len(etas)-1,len(phis)-1,datasetZ)

print "minimising"

xtol = np.finfo('float64').eps

grad = grad(nllSimul)
hess = hessian(nllSimul)

btol = 1.e-8

#lb = [-np.inf,-np.inf,0.,-np.inf]
#ub = [np.inf,np.inf,1.,np.inf]
#constraints = LinearConstraint( A=np.eye(x.shape[0]), lb=lb, ub=ub,keep_feasible=True )

res = minimize(nllSimul, x, args=(len(etas)-1, datasetZ, datasetZ, datasetZgen, datasetZgen),method = 'trust-constr',jac = grad, hess = hess,options={'verbose':3,'disp':True,'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

print res

plots(res.x,len(etas)-1,datasetZ,datasetZgen)

hessian = hess(x,len(etas)-1, datasetZ, datasetZ, datasetZgen, datasetZgen)
print np.linalg.eigvals(hessian)

print grad(res.x,len(etas)-1, datasetZ, datasetZ, datasetZgen, datasetZgen), "grad"
print hessian, "hessian"
invhess = np.linalg.inv(hessian)

edm = 0.5*np.matmul(np.matmul(grad(res.x,len(etas)-1, datasetZ, datasetZ, datasetZgen, datasetZgen).T,invhess),grad(res.x,len(etas)-1, datasetZ, datasetZ, datasetZgen, datasetZgen))

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

A = ROOT.TH1D("A", "A", len(etas)-1, etas)
M = ROOT.TH2D("M", "M", len(etas)-1, etas,len(phis)-1, phis)

A = array2hist(res.x[:len(etas)-1], A, np.sqrt(np.diag(invhess)[:len(etas)-1]))

err = np.sqrt(np.diag(invhess)[len(etas)-1:])

for i in range(1, len(etas)-1+1):
    for j in range(1, len(phis)-1+1):
        M.SetBinContent(i,j,res.x[len(etas)-1+i+(j-1)*len(etas)-1])
        M.SetBinError(i,j,err[i+(j-1)*len(etas)-1-1])

f = ROOT.TFile("calibrationLineshapeSimul.root", 'recreate')
f.cd()

A.Write()
M.Write()

































