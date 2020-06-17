import numpy as np
import matplotlib.pyplot as plt

def computeTrackLength(eta):

    L0=108.-4.4 #max track length in cm. tracker radius - first pixel layer

    tantheta = 2/(np.exp(eta)-np.exp(-eta))
    r = 267.*tantheta #267 cm: z position of the outermost disk of the TEC 
    L = np.where(np.absolute(eta) <= 1.4, L0, (np.where(eta > 1.4, np.minimum(r, 108.)-4.4, np.minimum(-r, 108.)-4.4)))

    #print(L)
    return L0/L

#imgtype = "png"
imgtype = "pdf"

#f = np.load("unbinnedfit.npz")

#fname = "unbinnedfitfullagnostic.npz"
#fname = "unbinnedfiterrs.npz"
#fname = "unbinnedfiterrseig.npz"
#fname = "unbinnedfiterrsTlog_12.npz"
#fname  = "unbinnedfiterrseigdiaga01.npz"
fname = "unbinnedfiterrsmed.npz"
#fname = "comprestriple/unbinnedfiterrsmeddiagfullreg.npz"
#fname = "compresnodiag/unbinnedfiterrsmedfull.npz"
#fname = "compresnodiag/unbinnedfiterrsmed012.npz"
#fname = "compresdiag/unbinnedfiterrsmedfull.npz"
#fname = "compresdiag/unbinnedfiterrsmed0124.npz"
#fname = "compres/unbinnedfiterrsmedfulldiag.npz"
#fname = "unbinnedfiterrseigdiaga00.npz"
#fname = "unbinnedfit.npz"
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

#print(dkplus)


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
         
parms = ["a",
         "c",
         "g1",
         "d1",
         "g2",
         "d2",
         "g3",
         "d3",
         ]


#a,c,g1,d1,g2,d2,g3,d3 = xs.T
#aerr,cerr,g1err,d1err,g2err,d2err,g3err,d3err = xerrs.T


#d1 = xs[:,0]
#d2 = xs[:,1]
#d1err = xerrs[:,0]
#d2err = xerrs[:,1]



#didxs = (1,2,4,5)
#didxs = slice(0,6)
didxs = slice(0,3)
ds = xs[:,didxs]
derrs = xerrs[:,didxs]

derrs = 0.5/np.sqrt(ds)*derrs
ds = np.sqrt(ds)

sortidxs = np.argsort(ds, axis=-1)
#sortidxs = np.argsort(np.abs(ds-30.), axis=-1)
#sortidxs = np.argsort(np.abs(ds-8.), axis=-1)

#ds = np.take_along_axis(ds,sortidxs,axis=-1)
#derrs = np.take_along_axis(derrs,sortidxs,axis=-1)

#print(sortidxs)
#print(ds)
#print(derrs)


#label = f"d_{i}"
label = "d"
plot = plt.figure()
plt.title(label)
plt.xlabel("$\eta$")

for i,(val,err) in enumerate(zip(ds.T,derrs.T)):
    plt.errorbar(etasc, val, xerr=0.5*etasw, yerr=err,fmt='none')

netabins = etasc.shape[0]
nelems = scalesigmamodelfine.shape[-1]


#gidxs = tuple(range(2,5)) + tuple(range(7,10))
gidxs = (0,5)

if True:
    localparms = xs[:,3:]
    localparms = np.reshape(localparms,(netabins,10,nelems))

    localerrs = xerrs[:,3:]
    localerrs = np.reshape(localerrs,(netabins,10,nelems))

    for ielem in range(nelems):

        gs = localparms[:,gidxs,ielem]
        gerrs = localerrs[:,gidxs,ielem]


        label = f"g, elem {ielem}"
        plot = plt.figure()
        plt.title(label)
        plt.xlabel("$\eta$")

        for val,err in zip(gs.T,gerrs.T):
            plt.errorbar(etasc, val, xerr=0.5*etasw, yerr=err,fmt='none')

    
#plt.show()

#d1 = xs[:,1]
#d2 = xs[:,2]
#d1err = xerrs[:,1]
#d2err = xerrs[:,2]

#d1err = 0.5/np.sqrt(d1)*d1err
#d2err = 0.5/np.sqrt(d2)*d2err

#d1 = np.sqrt(d1)
#d2 = np.sqrt(d2)

#doswap = d1>d2

#dlow = np.where(doswap,d2,d1)
#dhigh = np.where(doswap,d1,d2)

#dlowerr = np.where(doswap,d2err,d1err)
#dhigherr = np.where(doswap,d1err,d2err)

##docap = dhigh>500.
##dhigh = np.where(docap,500.,dhigh)
##dhigherr = np.where(docap,0.,dhigherr)

#for val,err,label in zip([dlow,dhigh],[dlowerr,dhigherr],["dlow","dhigh"]):
    #plot = plt.figure()
    #plt.errorbar(etasc, val, xerr=0.5*etasw, yerr=err,fmt='none')
    #plt.title(label)
    #plt.xlabel("$\eta$")
    
plt.show()

if False:
#if True:
    for iparm in range(xs.shape[1]):
        val = xs[:,iparm]
        err = xerrs[:,iparm]
        #label = parms[iparm]
        label = f"parms_{iparm}"
        
        isres = iparm in [0,1,3,5,7]
        
        #isres = iparm>2 and iparm<7
        
        if isres:
            val = abs(val)
            ##err = np.abs(2.*val)*err
            #val = val**2
        
        plot = plt.figure()
        plt.errorbar(etasc, val, xerr=0.5*etasw, yerr=err,fmt='none')
        #plt.title(parms[iparm])
        plt.title(label)
        plt.xlabel("$\eta$")
        #if isres:
            #plt.ylim(bottom=0.)
        plot.savefig(f"plots/parm_{iparm}.{imgtype}")



#plt.show()
#assert(0)

nelems = scalesigmamodelfine.shape[-1]

d2 = xs[:,1]
d3 = xs[:,2]
a = xs[:,3:3+nelems]
g2 = xs[:,3+3*nelems:3+4*nelems]
g3 = xs[:,3+4*nelems:3+5*nelems]
dosub = False

for ieta in range(etasc.shape[0]):
#for ieta in range(4,5):
#for ieta in range(6,7):
#for ieta in range(24):
    for ielem in range(nelems):

        scalebinnedplus = xbinned[ieta,1,:,ielem]
        scalebinnedminus = xbinned[ieta,0,:,ielem]
        
        scalebinnedpluserr = errsbinned[ieta,1,:,ielem]
        scalebinnedminuserr = errsbinned[ieta,0,:,ielem]
        
        
        ksplus = hdsetks[ieta,1,:]
        ksminus = hdsetks[ieta,0,:]
        
        #scalebinnedplus = 0.5*(scalebinnedplus+scalebinnedminus)
        #scalebinnedpluserr = 0.5*np.sqrt(scalebinnedpluserr**2 + scalebinnedminuserr**2)
        
        #ksplus = 0.5*(ksplus+ksminus)
        
        
        xerrplus = np.stack( (ksplus-kl,kh-ksplus),axis=0)
        xerrminus = np.stack( (ksminus-kl,kh-ksminus),axis=0)    
        
        xerrpluspt = np.stack( (1./ksplus-1./kl,1./kh-1./ksplus),axis=0)
        xerrminuspt = np.stack( (1./ksminus-1./kl,1./kh-1./ksminus),axis=0)
        
        print("xerrplus", xerrplus.shape)
        
        
        scalemodelfineplus = scalesigmamodelfine[ieta,1,:,ielem]
        scalemodelfineminus = scalesigmamodelfine[ieta,0,:,ielem]

        
        scalemodelfinepluserr = errsmodelfine[ieta,1,:,ielem]
        scalemodelfineminuserr = errsmodelfine[ieta,0,:,ielem]
                
        
        
        
        #dosub=False
        #if dosub:
            #ia = a[ieta,ielem]
            #ig2 = g2[ieta,ielem]
            #ig3 = g3[ieta,ielem]
            #id2 = d2[ieta]
            #id3 = d3[ieta]
            
            #scalebinnedplus += -ia*ksplus**2
            #scalebinnedminus += -ia*ksminus**2
            
            #scalemodelfineplus += -ia*ksfine**2
            #scalemodelfineminus += -ia*ksfine**2

            #g = ig2
            #d = id2
            #scalebinnedplus += -g/(1.+d*ksplus**2)
            #scalebinnedminus += -g/(1.+d*ksminus**2)
            
            #scalemodelfineplus += -g/(1.+d*ksfine**2)
            #scalemodelfineminus += -g/(1.+d*ksfine**2)

            
            #g = ig3
            #d = id3
            #scalebinnedplus += -g/(1.+d*ksplus**2)
            #scalebinnedminus += -g/(1.+d*ksminus**2)
            
            #scalemodelfineplus += -g/(1.+d*ksfine**2)
            #scalemodelfineminus += -g/(1.+d*ksfine**2)
        
        print("scalemodelfinepluserr.shape", scalemodelfinepluserr.shape)
        
        
        plot = plt.figure()
        plt.fill_between(ksfine,scalemodelfineplus-scalemodelfinepluserr,scalemodelfineplus+scalemodelfinepluserr, alpha=0.5)
        plt.plot(ksfine, scalemodelfineplus)
        plt.fill_between(-ksfine,scalemodelfineminus-scalemodelfineminuserr,scalemodelfineminus+scalemodelfineminuserr, alpha=0.5)
        plt.plot(-ksfine, scalemodelfineminus)
        plt.errorbar(ksplus,scalebinnedplus, xerr=xerrplus, yerr=scalebinnedpluserr,fmt='none')
        plt.errorbar(-ksminus,scalebinnedminus, xerr=-xerrminus, yerr=scalebinnedminuserr,fmt='none')
        plt.xlabel("$q\ k$ (GeV$^{-1}$)")
        plt.ylabel("$\sigma^2_k/k^2$")
        plt.title(f"Covariance Matrix ele{ielem}, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
        
        #p1 = plt.plot(ksfine, scaleplusalt)
        #plt.legend((p1,),('$-M/k - 0.5\sigma^2$',))
        #plt.show()
        plt.savefig(f"plots/scale_{ieta}.{imgtype}")
        
        if False:
            plot = plt.figure()
            plt.fill_between(1./ksfine,scalemodelfineplus-scalemodelfinepluserr,scalemodelfineplus+scalemodelfinepluserr, alpha=0.5)
            plt.plot(1./ksfine, scalemodelfineplus)
            #plt.fill_between(-1./ksfine,scalemodelfineminus-scalemodelfineminuserr,scalemodelfineminus+scalemodelfineminuserr, alpha=0.5)
            #plt.plot(-1./ksfine, scalemodelfineminus)
            plt.errorbar(1./ksplus,scalebinnedplus, xerr=xerrpluspt, yerr=scalebinnedpluserr,fmt='none')
            #plt.errorbar(-1./ksminus,scalebinnedminus, xerr=-xerrminuspt, yerr=scalebinnedminuserr,fmt='none')
            #plt.errorbar(1./ksplus,scalebinnedplus*ksplus**2, xerr=xerrpluspt, yerr=scalebinnedpluserr*ksplus**2,fmt='none')
            #plt.errorbar(-1./ksminus,scalebinnedminus*ksminus**2, xerr=-xerrminuspt, yerr=scalebinnedminuserr*ksplus**2,fmt='none')
            plt.xlabel("$q\ p_T$ (GeV)")
            plt.ylabel("$\sigma^2_k/k^2$")
            plt.title(f"Covariance Matrix ele{ielem}, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
            #plt.plot(1./ksfine,scaleplusalt)
            plt.savefig(f"plots/scalept_{ieta}.{imgtype}")    
            #p1 = plt.plot(1./ksfine, scaleplusalt)
            #plt.legend( (p1[0],),('$M/k + 0.5\sigma^2$',))
            #plt.savefig(f"plots/scalerescompare.pdf")    
            #plt.show()    
            
            #plt.show()



plt.show()
