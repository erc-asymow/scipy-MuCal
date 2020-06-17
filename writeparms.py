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
fname = "unbinnedfitgood.npz"
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

def altparms(parms, eta):
    l = computeTrackLength(eta)
    
    A,e,M,a,b,c,d,W,Y,Z,V,e2,Z2 = parms.T
    
    #asq = a**2
    #bsq = b**2
    #csq = c**2
    #dsq = d**2
    
    a = np.abs(a)
    b = np.abs(b)
    c = np.abs(c)
    d = np.abs(d)
    

    #a = a*l
    #b = b*l
    #c = c*l**2
    #d = d/l
    
    gsq = b**2/c**2 + d**2
    g = np.sqrt(gsq)
    
    a = a*l
    c = c*l**2
    g = g/l
    d = d/l
    
    Y = np.sqrt(1.+Y**2) - 1.
    
    g1 = np.sqrt((1.+W)*Y/(1.+A))
    g2 = np.sqrt(Y)
    
    
    r = a/c
    
    h1 = g1/r
    h2 = g2/r
    h3 = g/r
    h4 = d/r
    
    h12 = h1/h2
    h34 = h3/h4
    
    res = np.stack([A,e,M,W,Y,a,c,g1,g2,g,d,h1,h2,h3,h4,h12,h34], axis=-1)
    
    return res

jacg = jax.jit(jax.vmap(jax.jacfwd(altparms)))
#jacg = jax.jit(jax.jacfwd(altparms))

xsalt = altparms(xs, etasc)
jac = jacg(xs, etasc)
jacT = np.swapaxes(jac,-1,-2)
#print(jac.shape)
#assert(0)
covsalt = np.matmul(jac,np.matmul(covs,jacT))
    
xerrsalt = np.sqrt(np.diagonal(covsalt, offset=0, axis1=-1, axis2=-2))
         
parms = ["A",
         "e",
         "M",
         "$a$",
         "$b$",
         "$c$",
         "$d$",
         "W",
         "Y",
         "Z",
         "V",
         "e2",
         "Z2",
         ]

parmsalt = ["A","e","M","W","Y","a","c","$g_1$","$g_2$","g","d","$h_1$","$h_2$","$h_3$","$h_4$","$h_1/h_2$","$h_3/h_4$"]



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
        h = ROOT.TH1D(safelabels[iparm], labels[iparm], etas.shape[0]-1, onp.array(etas))
        h.GetXaxis().SetTitle("#eta")
        root_numpy.array2hist(x[:,iparm],h,errs[:,iparm])
        h.Write()

#plotparms(xs,xerrs, parms)
plotparms(xsalt, xerrsalt, parmsalt)
makeroot(xsalt, xerrsalt, parmsalt)

#plt.show()

