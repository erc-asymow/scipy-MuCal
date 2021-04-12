import numpy as np
import matplotlib.pyplot as plt

def computeTrackLength(eta):

    L0=108.-4.4 #max track length in cm. tracker radius - first pixel layer

    tantheta = 2/(np.exp(eta)-np.exp(-eta))
    r = 267.*tantheta #267 cm: z position of the outermost disk of the TEC 
    L = np.where(np.absolute(eta) <= 1.4, L0, (np.where(eta > 1.4, np.minimum(r, 108.)-4.4, np.minimum(-r, 108.)-4.4)))

    #print(L)
    return L0/L

plotdir = "plotscale"

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



for iparm in range(xs.shape[1]):
    label = f"parm_{iparm}"
    val = xs[:,iparm]
    err = xerrs[:,iparm]
    
    isres = iparm in [12,16]
    
    if isres:
        val = np.abs(val)
        #err = np.abs(2.*val)*err
        #val = val**2
    
    plot = plt.figure()
    plt.errorbar(etasc, val, xerr=0.5*etasw, yerr=err,fmt='none')
    plt.title(label)
    plt.xlabel("$\eta$")
    #if isres:
        #plt.ylim(bottom=0.)
    plot.savefig(f"{plotdir}/parm_{iparm}.{imgtype}")



#plt.show()
#assert(0)

for ieta in range(etasc.shape[0]):
    scalebinnedplus = xbinned[ieta,1,:,0]
    scalebinnedminus = xbinned[ieta,0,:,0]
    sigmabinnedplus = xbinned[ieta,1,:,1]
    sigmabinnedminus = xbinned[ieta,0,:,1]
    
    scalebinnedpluserr = errsbinned[ieta,1,:,0]
    scalebinnedminuserr = errsbinned[ieta,0,:,0]
    sigmabinnedpluserr = errsbinned[ieta,1,:,1]
    sigmabinnedminuserr = errsbinned[ieta,0,:,1]    
    
    #ksplus = hdsetks[ieta,1,:]
    #ksminus = hdsetks[ieta,0,:]
    
    ksplus = kc
    ksminus = kc
    
    #scalebinnedplus = xbinned[ieta,0,:,0]
    ##scalebinnedminus = xbinned[ieta,0,:,0]
    #sigmabinnedplus = xbinned[ieta,0,:,1]
    ##sigmabinnedminus = xbinned[ieta,0,:,1]
    
    #scalebinnedpluserr = errsbinned[ieta,0,:,0]
    ##scalebinnedminuserr = errsbinned[ieta,0,:,0]
    #sigmabinnedpluserr = errsbinned[ieta,0,:,1]
    ##sigmabinnedminuserr = errsbinned[ieta,0,:,1]    
    
    #ksplus = hdsetks[ieta,0,:]
    ##ksminus = hdsetks[ieta,0,:]
    
    xerrplus = np.stack( (ksplus-kl,kh-ksplus),axis=0)
    xerrminus = np.stack( (ksminus-kl,kh-ksminus),axis=0)    
    
    xerrpluspt = np.stack( (1./ksplus-1./kl,1./kh-1./ksplus),axis=0)
    xerrminuspt = np.stack( (1./ksminus-1./kl,1./kh-1./ksminus),axis=0)
    
    print("xerrplus", xerrplus.shape)
    
    #print(scalesigmamodel.shape)
    #print(errsmodel.shape)
    
    #scalemodelplus = scalesigmamodel[ieta,1,:,0]
    #scalemodelminus = scalesigmamodel[ieta,0,:,0]
    #sigmamodelplus = scalesigmamodel[ieta,1,:,1]
    #sigmamodelminus = scalesigmamodel[ieta,0,:,1]
    
    #scalemodelpluserr = errsmodel[ieta,1,:,0]
    #scalemodelminuserr = errsmodel[ieta,0,:,0]
    #sigmamodelpluserr = errsmodel[ieta,1,:,1]
    #sigmamodelminuserr = errsmodel[ieta,0,:,1]  
    
    
    scalemodelfineplus = scalesigmamodelfine[ieta,1,:,0]
    scalemodelfineminus = scalesigmamodelfine[ieta,0,:,0]
    sigmamodelfineplus = scalesigmamodelfine[ieta,1,:,1]
    sigmamodelfineminus = scalesigmamodelfine[ieta,0,:,1]
    
    scalemodelfinepluserr = errsmodelfine[ieta,1,:,0]
    scalemodelfineminuserr = errsmodelfine[ieta,0,:,0]
    sigmamodelfinepluserr = errsmodelfine[ieta,1,:,1]
    sigmamodelfineminuserr = errsmodelfine[ieta,0,:,1]    
    
    
    #convert to sigma^2
    sigmabinnedpluserr = 2.*sigmabinnedplus*sigmabinnedpluserr
    sigmabinnedminuserr = 2.*sigmabinnedminus*sigmabinnedminuserr
    sigmabinnedplus = sigmabinnedplus**2
    sigmabinnedminus = sigmabinnedminus**2
    
    sigmamodelfinepluserr = 2.*sigmamodelfineplus*scalemodelfinepluserr
    sigmamodelfineminuserr = 2.*sigmamodelfineminus*scalemodelfineminuserr
    sigmamodelfineplus = sigmamodelfineplus**2
    sigmamodelfineminus = sigmamodelfineminus**2

    #sigmamodelfinepluserr = 2.*sigmamodelfineplus*sigmamodelfinepluserr
    #sigmamodelfineminuserr = 2.*sigmamodelfineminus*sigmamodelfineminuserr
    #sigmamodelfineplus = sigmamodelfineplus**2
    #sigmamodelfineminus = sigmamodelfineminus**2    
    
    
    print("scalemodelfinepluserr.shape", scalemodelfinepluserr.shape)
    
    #print("M.shape, A.shape", M.shape, A.shape)
    ##M = xs[ieta,2]
    #M = M[ieta]
    #A = A[ieta]
    #e = e[ieta]
    #Y = Y[ieta]
    #Z = Z[ieta]
    #W = W[ieta]
    ##scaleplusalt = -M/kcfine - 0.5*sigmamodelfineminus**2
    ##scaleplusalt = -M/kcfine -Z/kcfine-e*kcfine  - Y*kcfine**2  -A  -W/kcfine**2
    ##A + q*M/k + e*k + W/k**2 + Y*k**2 + Z/k + q*V + q*e2*k + q*Z2*k**2
    ##scaleplusalt = +M/kcfine +Y*kcfine**2 +A  +e*kcfine 
    #scaleplusalt = +M/kcfine +Y*kcfine**2 +e*kcfine  + Z/kcfine + A + sigmamodelfineplus
    ##scaleplusalt = Z/kcfine + W/kcfine**2
    
    plot = plt.figure()
    plt.fill_between(kcfine,scalemodelfineplus-scalemodelfinepluserr-1.,scalemodelfineplus+scalemodelfinepluserr-1., alpha=0.5)
    plt.plot(kcfine, scalemodelfineplus-1.)
    plt.fill_between(-kcfine,scalemodelfineminus-scalemodelfineminuserr-1.,scalemodelfineminus+scalemodelfineminuserr-1., alpha=0.5)
    plt.plot(-kcfine, scalemodelfineminus-1.)
    plt.errorbar(ksplus,scalebinnedplus-1., xerr=xerrplus, yerr=scalebinnedpluserr,fmt='none')
    plt.errorbar(-ksminus,scalebinnedminus-1., xerr=-xerrminus, yerr=scalebinnedminuserr,fmt='none')
    plt.xlabel("$q\ k$ (GeV$^{-1}$)")
    plt.ylabel("$\delta k/k$")
    plt.title(f"Scale, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
    
    #p1 = plt.plot(kcfine, scaleplusalt)
    #plt.legend((p1,),('$-M/k - 0.5\sigma^2$',))
    #plt.show()
    plt.savefig(f"{plotdir}/scale_{ieta}.{imgtype}")
    
    plot = plt.figure()
    plt.fill_between(1./kcfine,scalemodelfineplus-scalemodelfinepluserr-1.,scalemodelfineplus+scalemodelfinepluserr-1., alpha=0.5)
    plt.plot(1./kcfine, scalemodelfineplus-1.)
    plt.fill_between(-1./kcfine,scalemodelfineminus-scalemodelfineminuserr-1.,scalemodelfineminus+scalemodelfineminuserr-1., alpha=0.5)
    plt.plot(-1./kcfine, scalemodelfineminus-1.)
    plt.errorbar(1./ksplus,scalebinnedplus-1., xerr=xerrpluspt, yerr=scalebinnedpluserr,fmt='none')
    plt.errorbar(-1./ksminus,scalebinnedminus-1., xerr=-xerrminuspt, yerr=scalebinnedminuserr,fmt='none')
    plt.xlabel("$q\ p_T$ (GeV)")
    plt.ylabel("$\delta k/k$")
    plt.title(f"Scale, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
    #plt.plot(1./kcfine,scaleplusalt)
    plt.savefig(f"{plotdir}/scalept_{ieta}.{imgtype}")    
    #p1 = plt.plot(1./kcfine, scaleplusalt)
    #plt.legend( (p1[0],),('$M/k + 0.5\sigma^2$',))
    #plt.savefig(f"{plotdir}/scalerescompare.pdf")    
    #plt.show()    

    if True:
        
        plot = plt.figure()
        plt.fill_between(kcfine,sigmamodelfineplus-sigmamodelfinepluserr,sigmamodelfineplus+sigmamodelfinepluserr, alpha=0.5)
        plt.plot(kcfine, sigmamodelfineplus)
        plt.fill_between(-kcfine,sigmamodelfineminus-sigmamodelfineminuserr,sigmamodelfineminus+sigmamodelfineminuserr, alpha=0.5)
        plt.plot(-kcfine, sigmamodelfineminus)
        plt.errorbar(ksplus,sigmabinnedplus, xerr=xerrplus, yerr=sigmabinnedpluserr,fmt='none')
        plt.errorbar(-ksminus,sigmabinnedminus, xerr=-xerrminus, yerr=sigmabinnedminuserr,fmt='none')
        plt.xlabel("$q\ k$ (GeV$^{-1}$)")
        plt.ylabel("$\sigma^2_k/k^2$")
        plt.title(f"Resolution, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
        plt.savefig(f"{plotdir}/res_{ieta}.{imgtype}")
        
        plot = plt.figure()
        plt.fill_between(1./kcfine,sigmamodelfineplus-sigmamodelfinepluserr,sigmamodelfineplus+sigmamodelfinepluserr, alpha=0.5)
        plt.plot(1./kcfine, sigmamodelfineplus)
        plt.fill_between(-1./kcfine,sigmamodelfineminus-sigmamodelfineminuserr,sigmamodelfineminus+sigmamodelfineminuserr, alpha=0.5)
        plt.plot(-1./kcfine, sigmamodelfineminus)
        plt.errorbar(1./ksplus,sigmabinnedplus, xerr=xerrpluspt, yerr=sigmabinnedpluserr,fmt='none')
        plt.errorbar(-1./ksminus,sigmabinnedminus, xerr=-xerrminuspt, yerr=sigmabinnedminuserr,fmt='none')
        plt.xlabel("$q\ p_T$ (GeV)")
        plt.ylabel("$\sigma^2_k/k^2$")
        plt.title(f"Resolution, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
        plt.savefig(f"{plotdir}/respt_{ieta}.{imgtype}")
    
        #plt.show()

    #plt.errorbar(kc,dkminus[ieta,:],yerr=dkerrminus[ieta,:], xerr=0.5*kw)
    #plt.plot(kcfine, dkfineminus[ieta,:])

    #plt.figure()
    #plt.errorbar(kc,sigmakplus[ieta,:],yerr=None, xerr=0.5*kw)
    #plt.plot(kcfine, sigmakfineplus[ieta,:])

    #plt.errorbar(kc,sigmakminus[ieta,:],yerr=None, xerr=0.5*kw)
    #plt.plot(kcfine, sigmakfineminus[ieta,:])


    #plt.errorbar(kc,sigmakplus[0,:],yerr=None, xerr=0.5*kw)

#plt.show()
