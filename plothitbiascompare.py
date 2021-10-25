import numpy as np
import matplotlib.pyplot as plt

def computeTrackLength(eta):

    L0=108.-4.4 #max track length in cm. tracker radius - first pixel layer

    tantheta = 2/(np.exp(eta)-np.exp(-eta))
    r = 267.*tantheta #267 cm: z position of the outermost disk of the TEC 
    L = np.where(np.absolute(eta) <= 1.4, L0, (np.where(eta > 1.4, np.minimum(r, 108.)-4.4, np.minimum(-r, 108.)-4.4)))

    #print(L)
    return L0/L

imgtype = "png"
#imgtype = "pdf"

#f = np.load("unbinnedfit.npz")

#fname = "unbinnedfitfullagnostic.npz"
#fname = "unbinnedfitkf.npz"

#fname = "unbinnedfitkforig.npz"
#fname = "unbinnedfitkfrefit.npz"
#fname = "unbinnedfitglobal.npz"
#fname = "unbinnedfitglobalcor.npz"
#fname = "unbinnedfitglobaliter.npz"
#fname = "unbinnedfitkforig.npz"
#fname = "plots_v84_13_15/unbinnedfitglobalitercor.npz"
#fname2 = "plots_v85_13_15/unbinnedfitglobalitercor.npz"

fname = "plots_v84_08/unbinnedfitglobalitercor.npz"
fname2 = "plots_v85_08/unbinnedfitglobalitercor.npz"

#fname = "unbinnedfitglobalcortest.npz"
#fname = "unbinnedfitglobal.npz"
#fname = "scalecheckpoint2/unbinnedfit.npz"
#fname = "scalerescheckpoint/unbinnedfit_2den.npz"
#fname = "scalerescheckpoint/unbinnedfit_3den.npz"
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
    #hdsetks = f["hdsetks"]
    #scalesigmamodel = f["scalesigmamodel"]
    scalesigmamodelfine = f["scalesigmamodelfine"]
    #errsmodel = f["errsmodel"]
    errsmodelfine = f["errsmodelfine"]
    #etas = f["etas"]
    ks = f["ks"]
    subdets = f["subdets"]
    layers = f["layers"]
    stereos = f["stereos"]
    ksfine = f["ksfine"]
    xs = f["xs"]
    xerrs = f["xerrs"]
    covs = f["covs"]
 

kc = 0.5*(ks[1:] + ks[:-1])
kl = ks[:-1]
kh = ks[1:]
kw = ks[1:] - ks[:-1]

kcfine = 0.5*(ksfine[1:] + ksfine[:-1])

#print(etas)
print(ks)
#print(dkplus)


subdetlabels = ["PXB", "PXE", "TIB", "TOB", "TID", "TEC"]

with np.load(fname2) as f:
    xs2 = f["xs"]
    xerrs2 = f["xerrs"]


for iparm in range(xs.shape[1]):
    label = f"parm_{iparm}"
    val = xs[:,iparm]/xs2[:,iparm]
    err = np.sqrt((xerrs[:,iparm]/xs[:,iparm])**2 + (xerrs2[:,iparm]/xs2[:,iparm])**2)*val
    
    isres = iparm in [12,16]
    
    if isres:
        val = np.abs(val)
        #err = np.abs(2.*val)*err
        #val = val**2
    
    plot = plt.figure()
    plt.errorbar(range(xs.shape[0]), val, xerr=0.5, yerr=err,fmt='none')
    plt.title(label)
    plt.xlabel("$\eta$")
    #if isres:
        #plt.ylim(bottom=0.)
    plot.savefig(f"plotscompare/parm_{iparm}.{imgtype}")

plt.show()
