import numpy as np
import matplotlib.pyplot as plt
import ROOT
#import root_numpy

def computeTrackLength(eta):

    L0=108.-4.4 #max track length in cm. tracker radius - first pixel layer

    tantheta = 2/(np.exp(eta)-np.exp(-eta))
    r = 267.*tantheta #267 cm: z position of the outermost disk of the TEC 
    L = np.where(np.absolute(eta) <= 1.4, L0, (np.where(eta > 1.4, np.minimum(r, 108.)-4.4, np.minimum(-r, 108.)-4.4)))

    #print(L)
    return L0/L

plotdir = "plotscaleparms"

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
fname = "unbinnedfitglobalitercorscale.npz"

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
#print(dkplus)

fmc = ROOT.TFile.Open("calmc.root")
fdata = ROOT.TFile.Open("calibrationDATA.root")

histnames = ["A", "e", "M", "a", "c"]

for iparm in range(xs.shape[1]):
    label = f"parm_{iparm}"
    val = xs[:,iparm]
    err = xerrs[:,iparm]
    
    isres = iparm in [12,16]
    
    if isres:
        val = np.abs(val)
        #err = np.abs(2.*val)*err
        #val = val**2
    
    hmc = fmc.Get(histnames[iparm])
    mcvals = np.zeros_like(val)
    mcerrs = np.zeros_like(err)
    
    hdata = fdata.Get(histnames[iparm])
    datavals = np.zeros_like(val)
    dataerrs = np.zeros_like(err)
    
    
    
    for ieta,etac in enumerate(etasc):
        mcvals[ieta] = hmc.GetBinContent(hmc.FindFixBin(etac))
        mcerrs[ieta] = hmc.GetBinError(hmc.FindFixBin(etac))
    
        datavals[ieta] = hdata.GetBinContent(hdata.FindFixBin(etac))
        dataerrs[ieta] = hdata.GetBinError(hdata.FindFixBin(etac))
    
        if (iparm == 1):
            mcvals[ieta] *= sintheta[ieta]
            datavals[ieta] *= sintheta[ieta]
    
    if (iparm<3):
        mcvals *= -1.
        datavals *= -1.
    
    plot = plt.figure()
    plt.errorbar(etasc, val, xerr=0.5*etasw, yerr=err,fmt='none', label = "MC Truth ($\mu$ gun)")
    plt.errorbar(etasc, mcvals, xerr=0.5*etasw, yerr=mcerrs,fmt='none', label= "MC Reco ($J/\psi$)")
    plt.errorbar(etasc, datavals, xerr=0.5*etasw, yerr=dataerrs,fmt='none', label = "Data ($J/\psi$)")
    plt.title(label)
    plt.xlabel("$\eta$")
    plt.legend(loc = "lower center")
    #if isres:
        #plt.ylim(bottom=0.)
    plot.savefig(f"{plotdir}/parm_{iparm}.{imgtype}")



plt.show()
#assert(0)

