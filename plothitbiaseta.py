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
fname = "unbinnedfitglobalitercor.npz"

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
    #subdets = f["subdets"]
    #layers = f["layers"]
    #stereos = f["stereos"]
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

    #A = parms[0]
    #e = parms[1]
    #M = parms[2]
    ##W = parms[3]
    #a = parms[3]
    #b = parms[4]
    #c = parms[5]
    #d = parms[6]
    #Y = parms[7]
    #Z = parms[8]
    #V = parms[9]
    #W = parms[10]
    #e2 = parms[11]
    #W2 = parms[12]
    #Z2 = parms[13]
#print("dkfineplus.shape", dkfineplus.shape)
#delta = A - e*sintheta*k + q*M/k + W/k**2 + Y/k + Z*k**2 + q*V + q*e2*k + q*W2/k**2 + q*Z2*k**2
#parms = ["$\delta k/k \sim -k^0$ (A)",
         #"$\delta k/k \sim \sin \\theta\ k$ ($\epsilon$)",
         #"$\delta k/k \sim -q/k$ (M)",
         #"$(\sigma k/k)^2 \sim (L_0/L)^2\ k^0$ (a)",
         #"g",
         #"$(\sigma k/k)^2 \sim (L_0/L)^4\ 1/k^2$ (c)",
         #"d",
         #"$\delta k/k \sim -1/k$",
         #"$\delta k/k \sim -k^2$",
         #"$\delta k/k \sim -q*k^0$",
         #"$\delta k/k \sim -1/k^2$ (W)",
         #"$\delta k/k \sim -q*k$",
         #"$\delta k/k \sim -q/k^2$",
         #"$\delta k/k \sim -q*k^2$",
         #]
         
#parms = ["$\delta k/k \sim -k^0$ (A)",
         #"$\delta k/k \sim \sin \\theta\ k$ ($\epsilon$)",
         #"$\delta k/k \sim -q/k$ (M)",
         #"$\delta k/k \sim -1/k^2$ (W)",
         #"$\delta k/k \sim -1/k$ (Y)",
         #"$\delta k/k \sim -k^2$ (Z)",
         #"$(\sigma k/k)^2 \sim (L_0/L)^2\ k^0$ (a)",
         #"g",
         #"$(\sigma k/k)^2 \sim (L_0/L)^4\ 1/k^2$ (c)",
         #"d",
          ##"$\delta k/k \sim -q*k^0$", #"V",
         #"$\delta k/k \sim -q*k$ (e2)",#"e2",
         ##"$\delta k/k \sim -q/k^2$",#"W2",
         #"$\delta k/k \sim -q*k^2$ (Z2)",#"Z2",
         #"",
         #"",
         #]


    #A = parms[0]
    #e = parms[1]
    #M = parms[2]
    #W = parms[3]
    #Y = parms[4]
    #Z = parms[5]
    #a = parms[6]
    #b = parms[7]
    #c = parms[8]
    #d = parms[9]
    #e2 = parms[10]
    
    #delta = A - e*k + q*M/k + Z*k**2 + q*e2*k + Y/k + W*q
    #return 1.-delta

#parms = ["$\delta k/k \sim -k^0$ (A)",
         #"$\delta k/k \sim  k$ ($\epsilon$)",
         #"$\delta k/k \sim -q/k$ (M)",
         #"$\delta k/k \sim -q k^0 $ (W)",
         #"$\delta k/k \sim -1/k$ (Y)",
         #"$\delta k/k \sim -k^2$ (Z)",
         #"$(\sigma k/k)^2 \sim (L_0/L)^2\ k^0$ (a)",
         #"g",
         #"$(\sigma k/k)^2 \sim (L_0/L)^4\ 1/k^2$ (c)",
         #"d",
          ##"$\delta k/k \sim -q*k^0$", #"V",
         #"$\delta k/k \sim -q*k$ (e2)",#"e2",
         ##"$\delta k/k \sim -q/k^2$",#"W2",
         ##"$\delta k/k \sim -q*k^2$ (Z2)",#"Z2",
         ##"",
         ##"",
         #]
         
#parms = ["A",
         #"e",
         #"M",
         #"$a^2$",
         #"$b^2$",
         #"$c^2$",
         #"$d^2$",
         #"W",
         #"Y",
         #"Z",
         #"V",
         #"e2",
         #"Z2",
         #]

etasc = 0.5*(etas[1:] + etas[:-1])
etasl = etas[:-1]
etash = etas[1:]
etasw = etas[1:] - etas[:-1]

#A,e,M,a,b,c,d,W,Y,Z,V,e2,Z2 = xs.T
#Aerr,eerr,Merr,aerr,berr,cerr,derr,Werr,Yerr,Zerr,Verr,e2err,Z2err = xerrs.T

#Yalt = np.sqrt(1.+Y**2)-1.
##Yerralt = np.exp(Y)*Yerr
#Yerralt = Yerr

#dalt = d**2*l**2
#derralt = 2*np.abs(d)*derr*l**2

#plot = plt.figure()
#plt.errorbar(etasc,dalt, xerr = 0.5*etasw, yerr = derralt, fmt="none")
#plt.errorbar(etasc,Yalt, xerr = 0.5*etasw, yerr = Yerralt, fmt="none")

#plt.show()

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
    plot.savefig(f"plots/parm_{iparm}.{imgtype}")

plt.show()

for ieta in range(xbinned.shape[0]):
    #subdetlabel = subdetlabels[subdets[ieta]]
    #layer = layers[ieta]
    #stereo = stereos[ieta]
    #if stereo>0:
        #stereolabel = "stereo"
    #else:
        #stereolabel = "rphi"
    
    #ksplus = hdsetks[ieta,1,:]
    #ksminus = hdsetks[ieta,0,:]
    
    ksplus = kc
    ksminus = kc
    
    xerrplus = np.stack( (ksplus-kl,kh-ksplus),axis=0)
    xerrminus = np.stack( (ksminus-kl,kh-ksminus),axis=0)    
    
    xerrpluspt = np.stack( (1./ksplus-1./kl,1./kh-1./ksplus),axis=0)
    xerrminuspt = np.stack( (1./ksminus-1./kl,1./kh-1./ksminus),axis=0)
    
    #xerrplus *= 0.
    #xerrminus *= 0.
    
    scalebinnedplus = xbinned[ieta,1,:,0]
    scalebinnedminus = xbinned[ieta,0,:,0]
    sigmabinnedplus = xbinned[ieta,1,:,1]
    sigmabinnedminus = xbinned[ieta,0,:,1]
    
    scalebinnedpluserr = errsbinned[ieta,1,:,0]
    scalebinnedminuserr = errsbinned[ieta,0,:,0]
    sigmabinnedpluserr = errsbinned[ieta,1,:,1]
    sigmabinnedminuserr = errsbinned[ieta,0,:,1]    
    
    
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
    
    #print(ksfine.shape)
    #print(scalemodelfineplus.shape)
    #print(scalemodelfinepluserr.shape)
    
    plot = plt.figure()
    plt.fill_between(kcfine,scalemodelfineplus-scalemodelfinepluserr,scalemodelfineplus+scalemodelfinepluserr, alpha=0.5)
    plt.plot(kcfine, scalemodelfineplus)
    plt.fill_between(-kcfine,scalemodelfineminus-scalemodelfineminuserr,scalemodelfineminus+scalemodelfineminuserr, alpha=0.5)
    plt.plot(-kcfine, scalemodelfineminus)
    plt.errorbar(ksplus,scalebinnedplus, xerr=xerrplus, yerr=scalebinnedpluserr,fmt='none')
    plt.errorbar(-ksminus,scalebinnedminus, xerr=-xerrminus, yerr=scalebinnedminuserr,fmt='none')
    plt.xlabel("$q\ k$ (GeV$^{-1}$)")
    plt.ylabel("$\delta x (cm)$")
    plt.title(f"Bias: ieta = {ieta}")
    
    #p1 = plt.plot(ksfine, scaleplusalt)
    #plt.legend((p1,),('$-M/k - 0.5\sigma^2$',))
    #plt.show()
    plt.savefig(f"plots/scale_{ieta}.{imgtype}")
    
    plot = plt.figure()
    plt.fill_between(1./kcfine,scalemodelfineplus-scalemodelfinepluserr,scalemodelfineplus+scalemodelfinepluserr, alpha=0.5)
    plt.plot(1./kcfine, scalemodelfineplus)
    plt.fill_between(-1./kcfine,scalemodelfineminus-scalemodelfineminuserr,scalemodelfineminus+scalemodelfineminuserr, alpha=0.5)
    plt.plot(-1./kcfine, scalemodelfineminus)
    plt.errorbar(1./ksplus,scalebinnedplus, xerr=xerrpluspt, yerr=scalebinnedpluserr,fmt='none')
    plt.errorbar(-1./ksminus,scalebinnedminus, xerr=-xerrminuspt, yerr=scalebinnedminuserr,fmt='none')
    plt.xlabel("$q\ p_T$ (GeV)")
    plt.ylabel("$\delta x (cm)$")
    plt.title(f"Bias: ieta = {ieta}")
    #plt.title(f"Scale, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
    #plt.plot(1./ksfine,scaleplusalt)
    plt.savefig(f"plots/scalept_{ieta}.{imgtype}")    
    #p1 = plt.plot(1./ksfine, scaleplusalt)
    #plt.legend( (p1[0],),('$M/k + 0.5\sigma^2$',))
    #plt.savefig(f"plots/scalerescompare.pdf")    
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
        plt.ylabel("$\sigma^2_x (cm^2)$")
        plt.title(f"Resolution: ieta = {ieta}")
        #plt.title(f"Resolution, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
        plt.savefig(f"plots/res_{ieta}.{imgtype}")
        
        plot = plt.figure()
        plt.fill_between(1./kcfine,sigmamodelfineplus-sigmamodelfinepluserr,sigmamodelfineplus+sigmamodelfinepluserr, alpha=0.5)
        plt.plot(1./kcfine, sigmamodelfineplus)
        plt.fill_between(-1./kcfine,sigmamodelfineminus-sigmamodelfineminuserr,sigmamodelfineminus+sigmamodelfineminuserr, alpha=0.5)
        plt.plot(-1./kcfine, sigmamodelfineminus)
        plt.errorbar(1./ksplus,sigmabinnedplus, xerr=xerrpluspt, yerr=sigmabinnedpluserr,fmt='none')
        plt.errorbar(-1./ksminus,sigmabinnedminus, xerr=-xerrminuspt, yerr=sigmabinnedminuserr,fmt='none')
        plt.xlabel("$q\ p_T$ (GeV)")
        plt.ylabel("$\sigma^2_x (cm^2)$")
        #plt.title(f"Resolution, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
        plt.title(f"Resolution: ieta = {ieta}")
        plt.savefig(f"plots/respt_{ieta}.{imgtype}")
    
        #plt.show()

    #plt.errorbar(kc,dkminus[ieta,:],yerr=dkerrminus[ieta,:], xerr=0.5*kw)
    #plt.plot(ksfine, dkfineminus[ieta,:])

    #plt.figure()
    #plt.errorbar(kc,sigmakplus[ieta,:],yerr=None, xerr=0.5*kw)
    #plt.plot(ksfine, sigmakfineplus[ieta,:])

    #plt.errorbar(kc,sigmakminus[ieta,:],yerr=None, xerr=0.5*kw)
    #plt.plot(ksfine, sigmakfineminus[ieta,:])


    #plt.errorbar(kc,sigmakplus[0,:],yerr=None, xerr=0.5*kw)

##plt.show()
