import os
import multiprocessing

ncpu = multiprocessing.cpu_count()

os.environ["OMP_NUM_THREADS"] = str(ncpu)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncpu)
os.environ["MKL_NUM_THREADS"] = str(ncpu)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncpu)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncpu)

import jax
import jax.numpy as np
import numpy as onp
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

from fittingFunctionsBinned import defineStatePars, nllPars, defineState, nll, defineStateParsSigma, nllParsSigma, plots, plotsPars, plotsParsBkg, scaleFromPars, splitTransformPars, nllBins
import argparse
import functools
import time

#slower but lower memory usage calculation of hessian which
#explicitly loops over hessian rows
def hessianlowmem(fun):
    def _hessianlowmem(x, f):
        _, hvp = jax.linearize(jax.grad(f), x)
        hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return functools.partial(_hessianlowmem, f=fun)

#compromise version which vectorizes the calculation, but only partly to save memory
def hessianoptsplit(fun, vsize=4):
    def _hessianopt(x, f):
        _, hvp = jax.linearize(jax.grad(f), x)
        hvp = jax.jit(hvp)
        n = np.prod(x.shape)
        idxs =  np.arange(vsize, n, vsize)
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        splitbasis = np.split(basis,idxs)
        vhvp = jax.vmap(hvp)
        vhvp = jax.jit(vhvp)
        return np.concatenate([vhvp(b) for b in splitbasis]).reshape(x.shape + x.shape)
    return functools.partial(_hessianopt, f=fun)

#optimized version which is faster than the built-in hessian for some reason
# **TODO** follow up with jax authors to understand why
def hessianopt(fun):
    def _hessianopt(x, f):
        _, hvp = jax.linearize(jax.grad(f), x)
        hvp = jax.jit(hvp)
        vhvp = jax.vmap(hvp)
        vhvp = jax.jit(vhvp)
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return vhvp(basis).reshape(x.shape + x.shape)
    return functools.partial(_hessianopt, f=fun)


#def vgrad(fun):
    #g = jax.grad(fun)
    #return jax.vmap(g)

def hvp(fun):
    def _hvp(x, v, f):
        return jax.jvp(jax.grad(f), (x,), (v,))[1]
    return functools.partial(_hvp, f=fun)

class CachingHVP():
    def __init__(self,fun):
        self.grad = jax.jit(jax.grad(fun))
        self.x = None
        self.flin = None
        
    def hvp(self,x,v):
        if self.x is None or not np.equal(x,self.x).all():
            _,flin = jax.linearize(self.grad,x)
            self.flin = jax.jit(flin)
            #self.flin = flin
            self.x = x
        return self.flin(v)

parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-runCalibration', '--runCalibration', default=False, action='store_true', help='Use to fit corrections, omit to fit scale parameter')
parser.add_argument('-fitResolution', '--fitResolution', default=False, action='store_true', help='Use to fit resolution paramter, omit to fit sigma')


args = parser.parse_args()
isJ = args.isJ
runCalibration = args.runCalibration
fitResolution = args.fitResolution

#file = open("calInput{}MC_4etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
file = open("calInput{}DATA_12etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
pkg = pickle.load(file)

dataset = pkg['dataset']
etas = pkg['edges'][0]
pts = pkg['edges'][2]
masses = pkg['edges'][-1]
binCenters1 = pkg['binCenters1']
binCenters2 = pkg['binCenters2']
good_idx = pkg['good_idx']

filegen = open("calInput{}MCgen_12etaBins_5ptBins.pkl".format('J' if isJ else 'Z'), "rb")
datasetgen = pickle.load(filegen)

nEtaBins = len(etas)-1
nPtBins = len(pts)-1
nBins = dataset.shape[0]

print(pts)

if runCalibration:
    if fitResolution:
        x = defineStateParsSigma(nEtaBins,nPtBins, dataset, isJ)
    else:
        x = defineStatePars(nEtaBins,nPtBins, dataset, isJ)
else:
    x = defineState(nEtaBins,nPtBins, dataset)


print("minimising")

xtol = np.finfo('float64').eps
btol = 1.e-8
#btol = 0.1
maxiter = 100000
#maxiter = 2

if runCalibration:
    lb_scale = np.concatenate((-0.01*np.ones(nEtaBins),-0.01*np.ones(nEtaBins), -1e-2*np.ones(nEtaBins)),axis=0)
    ub_scale = np.concatenate((0.01*np.ones(nEtaBins),0.01*np.ones(nEtaBins), 1e-2*np.ones(nEtaBins)),axis=0)

    if fitResolution:
        #lb_sigma = np.concatenate((np.zeros(nEtaBins),np.zeros(nEtaBins),np.zeros(nEtaBins),np.zeros(nEtaBins)), axis =0)
        #ub_sigma = np.concatenate((1.e-3*np.ones(nEtaBins),1.e-3*np.ones(nEtaBins),1.e-4*np.ones(nEtaBins),100000.*np.ones(nEtaBins)), axis =0)
        #lb_sigma = np.zeros(nEtaBins)
        lb_sigma = 1.e-6*np.ones(nEtaBins)
        ub_sigma = 1.e-3*np.ones(nEtaBins)
    else:
        lb_sigma = np.full((nBins),-np.inf)
        ub_sigma = np.full((nBins),np.inf)
else:   
    lb_scale = np.full((nBins),0.95)
    ub_scale = np.full((nBins),1.05)

    lb_sigma = np.full((nBins),-np.inf)
    ub_sigma = np.full((nBins),np.inf)


lb_nsig = np.full((nBins),-np.inf)
ub_nsig = np.full((nBins),np.inf)

lb_nbkg = np.full((nBins),-np.inf)
ub_nbkg = np.full((nBins),np.inf)

#lb = np.concatenate((lb_scale,lb_sigma,lb_nsig),axis=0)
#ub = np.concatenate((ub_scale,ub_sigma,ub_nsig),axis=0)

lb = np.concatenate((lb_scale,lb_sigma,lb_nsig,lb_nbkg),axis=0)
ub = np.concatenate((ub_scale,ub_sigma,ub_nsig,ub_nbkg),axis=0)

bounds = []
for l,u in zip(lb,ub):
    bounds.append((l,u))

#constraints = LinearConstraint( A=np.eye(x.shape[0]), lb=lb, ub=ub,keep_feasible=True )
constraints = []

if runCalibration:
    if fitResolution:
        fnll = nllParsSigma
    else:
        fnll = nllPars
    #convert fnll to single parameter function fnllx(x)
    fnllx = functools.partial(fnll, nEtaBins=nEtaBins, nPtBins=nPtBins, dataset=dataset, datasetGen=datasetgen, isJ=isJ, etas=etas, masses=masses, binCenters1=binCenters1, binCenters2=binCenters2, good_idx=good_idx)
else:
    fnll = nll
    #convert fnll to single parameter function fnllx(x)
    fnllx = functools.partial(fnll, nEtaBins=nEtaBins, nPtBins=nPtBins, dataset=dataset, datasetGen=datasetgen, masses=masses)



#def vgrad(fun):
    #g = jax.grad(fun)
    #return jax.vmap(g)

#tests = np.array(5.)
#print(tests.shape)
#assert(0)

scale = np.ones((nBins,),dtype='float64')
sigma = 6e-3*np.ones((nBins,),dtype='float64')
fbkg = 0.05*np.ones((nBins,),dtype='float64')
slope = 0.02*np.ones((nBins,),dtype='float64')

xscale = np.stack([scale,sigma,fbkg,slope],axis=-1)

nllBinspartial = functools.partial(nllBins, masses=masses)

gbins = jax.grad(nllBinspartial, argnums=(0))
vgbins = jax.vmap(gbins)
vgbins = jax.jit(vgbins,static_argnums=(1,2))

hbins = jax.hessian(nllBinspartial, argnums=0)
vhbins = jax.vmap(hbins)
vhbins = jax.jit(vhbins,static_argnums=(1,2))

ve = jax.vmap(np.linalg.eigh)

#gbins = jax.jit(gbins)

#vgbins = jax.jit(vgbins)




def nllsum(xscale):
    return np.sum(nllBins(xscale, dataset, datasetgen, masses))

gnsum = jax.jit(jax.grad(nllsum))
hnsum = hessianlowmem(nllsum)


vgnllbins = vgbins(xscale,dataset,datasetgen)
#gnllbins = gbins(scale,sigma,fbkg,slope,dataset,datasetgen)
gnllbinssum = gnsum(scale)

vhnllbins = vhbins(xscale,dataset,datasetgen)

e,u = ve(vhnllbins)
print(e.shape)
print(e)
print(np.min(e))

assert(0)

print(vhnllbins.shape)
print(vhnllbins)
#print(len(vhnllbins))
#print(vhnllbins[0])
#print(len(vhnllbins[0]))
#print(vhnllbins[0][0])

def benchmark(fun, number=100):
    #with jax.disable_jit():
    t0 = time.time()
    for i in range(number):
        #xr = xscale + 1e-6*onp.random.standard_normal()
        res = fun()
        print(res.flatten()[0])
    t = time.time() - t0
    print(t,number)
    return t/number

#assert(0)

#print(vgnllbins.shape)
#print(vgnllbins)
#print(gnllbinssum.shape)
#print(gnllbinssum)

resv = benchmark(lambda: vgbins(xscale,dataset,datasetgen), number=10)
ressum = benchmark(lambda: gnsum(xscale), number=10)
resvh = benchmark(lambda: vhbins(xscale,dataset,datasetgen), number=10)
resh = benchmark(lambda: hnsum(xscale), number=1)

#resv = benchmark(vgbins, number=10)
#resvh = benchmark(vhbins, number=10)

#print(resv, resvh)
print(resv,ressum, resvh)


#vgradnll = jax.jit(vgrad(fnllx))

#print(vgradnll(x))

assert(0)


fgradnll = jax.jit(jax.value_and_grad(fnllx))
#fgradnll = jax.value_and_grad(fnllx)

#def fgradwrapper(x):
    

def fgradnlldebug(x):
    #print(x)
    f,grad = fgradnll(x)
    print(f)
    if np.isnan(f) or np.any(np.isnan(grad)):
        print("nan detected")
        print(x)
        print(f)
        print(grad)
        
        fnllx(x)
        assert(0)
        
    return f,grad

hessnll = hessianlowmem(fnllx)
hesspnll = jax.jit(hvp(fnllx))

#FIXME
#this seems to actually work properly on gpu, but tries to allocate too much memory on cpu for some reason
#hessnll = jax.jit(hessian(fnllx))





#res = minimize(fgradnlldebug, x,\
    #method = 'trust-constr',jac = True, hess=SR1(),constraints=constraints,\
    #options={'verbose':3,'disp':True,'maxiter' : maxiter, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

#wrapper to handle printing which otherwise doesn't work properly from bfgs apparently
class NLLHandler():
    def __init__(self, fun, fundebug = None):
        self.fun = fun
        self.fundebug = fundebug
        self.iiter = 0
        self.f = 0.
        self.grad = np.array(0.)
        
    def wrapper(self, x):
        f,grad = self.fun(x)
        if np.isnan(f) or np.any(np.isnan(grad)):
            print("nan detected")
            print(x)
            print(f)
            print(grad)
            
            if self.fundebug is not None:
                self.fundebug(x)
                
            assert(0)    
        self.f = f
        self.grad = grad
        return f,grad
    
    def callback(self,x):
        print(self.iiter, self.f, np.linalg.norm(self.grad))
        self.iiter += 1
        
handler = NLLHandler(fgradnll, fnllx)

hvpcache = CachingHVP(fnllx)

res = minimize(handler.wrapper, x, callback=handler.callback,\
    method = 'trust-krylov',jac = True, hessp=hvpcache.hvp,\
    options={'verbose':3,'disp':False,'maxiter' : maxiter, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

#xi = x
#for i in range(3):
    #res = minimize(handler.wrapper, xi,\
        #method = 'bfgs',jac = True, hess=None,callback=handler.callback,\
        #options={'verbose':9,'disp':9,'maxiter' : maxiter, 'gtol' : 1e-16, 'xtol' : xtol, 'barrier_tol' : btol})
    #xi = res.x


#res = minimize(fgradnlldebug, x,\
    #method = 'L-BFGS-B',jac = True, hess=None,\
    #options={'verbose':9,'disp':9,'maxiter' : maxiter, 'gtol' : 1e-16, 'xtol' : xtol, 'barrier_tol' : btol})


print(res)


fitres = res.x

val,gradient = fgradnll(res.x)
gradfinal = gradient

print(val)
print(gradient)

#print gradient, "gradient"

hessian = hessnll(res.x)
#hessian = np.eye(x.shape[0])

hessfinal = hessian

#print np.linalg.eigvals(hessfinal), "eigenvalues"

invhess = np.linalg.inv(hessfinal)

edm = 0.5*np.matmul(np.matmul(gradfinal.T,invhess),gradfinal)

print(fitres, "+/-", np.sqrt(np.diag(invhess)))
print(edm, "edm")

#errs = np.sqrt(np.diag(invhess))
#erridx = np.where(np.isnan(errs))
#print(erridx)
#binidx = (erridx[0] - nEtaBins-nBins,)
#print(binidx)
#for gid in good_idx:
    #print(gid[binidx])

slope = res.x[4*nEtaBins+2*nBins:]
print(np.min(slope),np.max(slope), np.mean(slope))

diag = np.diag(np.sqrt(np.diag(invhess)))
diag = np.linalg.inv(diag)
corr = np.dot(diag,invhess).dot(diag)

#only for model parameters
corr = corr[:4*nEtaBins,:4*nEtaBins]

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.clf()
plt.pcolor(corr, cmap='jet', vmin=-1, vmax=1)
plt.colorbar()
plt.savefig("corrMC.pdf")

def scaleFromAeM(xAeM, etas, binCenters1, binCenters2, good_idx):

    #TODO move this to common code as well
    A = 0.01*np.tanh(xAeM[:nEtaBins])
    e = 0.01*np.tanh(xAeM[nEtaBins:2*nEtaBins])
    M = 0.01*np.tanh(xAeM[2*nEtaBins:3*nEtaBins])
    
    scale = scaleFromPars(A, e, M, etas, binCenters1, binCenters2, good_idx)

    return scale

if runCalibration:
    #plotsPars(res.x,nEtaBins,nPtBins,dataset,datasetgen,masses,isJ,etas, binCenters1, binCenters2, good_idx)
    
    plotsParsBkg(res.x,nEtaBins,nPtBins,dataset,datasetgen,masses,isJ,etas, binCenters1, binCenters2, good_idx)

    print("computing scales and errors:")
    
    ndata = np.sum(dataset,axis=-1)
    A,e,M,a,b,c,d,nsig,nbkg,slope = splitTransformPars(res.x, ndata, nEtaBins, nBins, isJ)

    hA = ROOT.TH1D("A", "A", nEtaBins, onp.array(etas.tolist()))
    he = ROOT.TH1D("e", "e", nEtaBins, onp.array(etas.tolist()))
    hM = ROOT.TH1D("M", "M", nEtaBins, onp.array(etas.tolist()))

    hA = array2hist(A, hA, np.sqrt(np.diag(invhess)[:nEtaBins]))
    he = array2hist(e, he, np.sqrt(np.diag(invhess)[nEtaBins:2*nEtaBins]))
    hM = array2hist(M, hM, np.sqrt(np.diag(invhess)[2*nEtaBins:3*nEtaBins]))

    hA.GetYaxis().SetTitle('b field correction')
    he.GetYaxis().SetTitle('material correction')
    hM.GetYaxis().SetTitle('alignment correction')

    hA.GetXaxis().SetTitle('#eta')
    he.GetXaxis().SetTitle('#eta')
    hM.GetXaxis().SetTitle('#eta')

    if fitResolution:
        a = res.x[3*nEtaBins:4*nEtaBins]
        ha = ROOT.TH1D("a", "a", nEtaBins, onp.array(etas.tolist()))
        ha = array2hist(a, ha, np.sqrt(np.diag(invhess)[3*nEtaBins:4*nEtaBins]))
        ha.GetYaxis().SetTitle('material correction')
        ha.GetXaxis().SetTitle('#eta')
    
    xAeM = res.x[:3*nEtaBins]
    scale = scaleFromAeM(xAeM, etas, binCenters1, binCenters2, good_idx)
    
    jacobianscale = jax.jit(jax.jacfwd(scaleFromAeM))
    jac = jacobianscale(xAeM,etas, binCenters1, binCenters2, good_idx)
    invhessAeM = invhess[:3*nEtaBins,:3*nEtaBins]
    scale_invhess = np.matmul(np.matmul(jac,invhessAeM),jac.T)
    scale_err = np.sqrt(np.diag(scale_invhess))
    print("computed scales error, now plotting...")
    
    #have to use original numpy to construct the bin edges because for some reason this doesn't work with the arrays returned by jax
    scaleplot = ROOT.TH1D("scale", "scale", nBins, onp.linspace(0, nBins, nBins+1))
    scaleplot = array2hist(scale, scaleplot, scale_err)

    for ibin in range(nBins):
        ieta1 = good_idx[0][ibin]
        ieta2 = good_idx[1][ibin]
        ipt1 = good_idx[2][ibin]
        ipt2 = good_idx[3][ibin]
        scaleplot.GetXaxis().SetBinLabel(ibin+1,'eta1_{}_eta2_{}_pt1_{}_pt2_{}'.format(ieta1,ieta2,ipt1,ipt2))

    scaleplot.GetXaxis().LabelsOption("v")

    f = ROOT.TFile("calibrationMC.root", 'recreate')
    f.cd()

    hA.Write()
    he.Write()
    hM.Write()
    if fitResolution:
        ha.Write()
    scaleplot.Write()

else:
    print("plotting...")
    plots(res.x,nEtaBins,nPtBins,dataset,datasetgen,masses,isJ,good_idx)

    f = ROOT.TFile("scaleMC.root", 'recreate')
    f.cd()

    scaleplot = ROOT.TH1D("scale", "scale", nBins, onp.linspace(0, nBins, nBins+1))
    scaleplot.GetYaxis().SetTitle('scale')

    scaleplot = array2hist(fitres[:nBins], scaleplot, np.sqrt(np.diag(invhess)[:nBins]))

    for ibin in range(nBins):
        ieta1 = good_idx[0][ibin]
        ieta2 = good_idx[1][ibin]
        ipt1 = good_idx[2][ibin]
        ipt2 = good_idx[3][ibin]
        scaleplot.GetXaxis().SetBinLabel(ibin+1,'eta1_{}_eta2_{}_pt1_{}_pt2_{}'.format(ieta1,ieta2,ipt1,ipt2))

    scaleplot.GetXaxis().LabelsOption("v")
    scaleplot.Write()

