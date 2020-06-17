#import numpy as np
import jax
from jax import numpy as np
import numpy as onp

from fittingFunctionsBinned import computeTrackLength

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import ROOT
import root_numpy

#def computeTrackLength(eta):

    #L0=108.-4.4 #max track length in cm. tracker radius - first pixel layer

    #tantheta = 2/(np.exp(eta)-np.exp(-eta))
    #r = 267.*tantheta #267 cm: z position of the outermost disk of the TEC 
    #L = np.where(np.absolute(eta) <= 1.4, L0, (np.where(eta > 1.4, np.minimum(r, 108.)-4.4, np.minimum(-r, 108.)-4.4)))

    ##print(L)
    #return L0/L

imgtype = "png"
#imgtype = "pdf"

#f = np.load("unbinnedfit.npz")

#fname = "unbinnedfitfullagnostic.npz"
fname = "compresmixedtriple/unbinnedfiterrsmedfulldiag.npz"
#fname = "unbinnedfit_gun_jpsi.npz"
#fname = "unbinnedfitfullselected.npz"
#fname = "unbinnedfitfullselectedfixedgausexp.npz"
#fname = "unbinnedfitfullselectedfixedgaus.npz"

with np.load(fname) as f:
    print(f.files)
    #dkplus = f["dkplus"]
    #dkerrplus = f["dkerrplus"]
    #sigmakplus = f["sigmakplus"]
    #dkfineplus = f["dkfineplus"]
    #sigmakfineplus = f["sigmakfineplus"]
    #dkminus = f["dkminus"]
    #dkerrminus = f["dkerrminus"]
    #sigmakminus = f["sigmakminus"]
    #dkfineminus = f["dkfineminus"]
    #sigmakfineminus = f["sigmakfineminus"]
    #etas = f["etas"]
    #ks = f["ks"]
    #ksfine = f["ksfine"]
    
    xbinned = f["xbinned"]
    errsbinned = f["errsbinned"]
    hdsetks = f["hdsetks"]
    #scalesigmamodel = f["scalesigmamodel"]
    scalesigmamodelfine = f["scalesigmamodelfine"]
    #errsmodel = f["errsmodel"]
    errsmodelfine = f["errsmodelfine"]
    etas = f["etas"]
    ks = f["ks"]
    ksfine = f["ksfine"]
    xs = f["xs"]
    xerrs = f["xerrs"]
    covs = f["covs"]
 

etasc = 0.5*(etas[1:] + etas[:-1])
etasl = etas[:-1]
etash = etas[1:]
etasw = etas[1:] - etas[:-1]

l = computeTrackLength(etasc)
sintheta = np.sin(2*np.arctan(np.exp(-etasc)))
costheta = np.cos(2*np.arctan(np.exp(-etasc)))
tantheta = np.tan(2*np.arctan(np.exp(-etasc)))

kc = 0.5*(ks[1:] + ks[:-1])
kl = ks[:-1]
kh = ks[1:]
kw = ks[1:] - ks[:-1]

kcfine = 0.5*(ksfine[1:] + ksfine[:-1])

print(etas)
print(ks)

ds = xs[:,:3]
derrs = xerrs[:,:3]

derrs = 0.5/np.sqrt(ds)*derrs
ds = np.sqrt(ds)

sortidxs = np.argsort(ds, axis=-1)

ds = np.take_along_axis(ds,sortidxs,axis=-1)
derrs = np.take_along_axis(derrs,sortidxs,axis=-1)

xsalt = ds
xerrsalt = derrs
parmsalt = ["d1", "d2", "d3"]




def plotparms(x, errs, labels):
    #safelabels = [label.replace("$","") for label in labels]

    for iparm in range(x.shape[1]):
        val = x[:,iparm]
        err = errs[:,iparm]
        
        plot = plt.figure()
        plt.errorbar(etasc, val, xerr=0.5*etasw, yerr=err,fmt='none')
        plt.title(labels[iparm])
        plt.xlabel("$\eta$")
        #if isres:
            #plt.ylim(bottom=0.)
        plot.savefig(f"plots/parm_{iparm}.{imgtype}")

def makeroot(x, errs, labels):
    f = ROOT.TFile("calibrationMC.root", 'recreate')
    f.cd()
    
    safelabels = [label.replace("$","") for label in labels]
    
    for iparm in range(x.shape[1]):
        print("errs")
        print(errs[:,iparm])
        h = ROOT.TH1D(safelabels[iparm], labels[iparm], etas.shape[0]-1, onp.array(etas))
        h.GetXaxis().SetTitle("#eta")
        root_numpy.array2hist(x[:,iparm],h,errs[:,iparm])
        h.Write()

#plotparms(xs,xerrs, parms)
plotparms(xsalt, xerrsalt, parmsalt)
#plt.show()
makeroot(xsalt, xerrsalt, parmsalt)

#plt.show()

