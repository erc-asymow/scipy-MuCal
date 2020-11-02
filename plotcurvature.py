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

if False:
    A,e,M,a,b,c,d,W,Y,Z,V,e2,Z2 = xs.T
    Aerr,eerr,Merr,aerr,berr,cerr,derr,Werr,Yerr,Zerr,Verr,e2err,Z2err = xerrs.T

    ##e2 = -Z2/4.
    #plot = plt.figure()
    #p1 = plt.errorbar(etasc,e2,xerr=0.5*etasw, yerr=e2err, fmt="none")
    #p2 = plt.errorbar(etasc,-(1./4.)*Z2,xerr=0.5*etasw, yerr=(1./4.)*Z2err, fmt="none")
    #plt.legend((p1,p2), ("e2", "-Z2/4"))
    #plt.savefig("e2Z2compare.pdf")
    ##plt.show()


    ##-1/12*e = A = 1/40*Y
    #plt.figure()
    #escale = -1./12.
    #yscale = 1./40.
    #p1 = plt.errorbar(etasc,A,xerr=0.5*etasw, yerr=Aerr, fmt="none")
    #p2 = plt.errorbar(etasc,escale*e,xerr=0.5*etasw, yerr=np.abs(escale)*eerr, fmt="none")
    #p3 = plt.errorbar(etasc,yscale*Y,xerr=0.5*etasw, yerr=np.abs(yscale)*Yerr, fmt="none")
    #plt.legend((p1,p2,p3),("A","-e/12","Y/40"))
    #plt.savefig("plots/AeYcompare.pdf")

    #plt.figure()
    #escale = -1./12.
    #yscale = 1./40.
    #p1 = plt.errorbar(etasc,e-A/escale,xerr=0.5*etasw, yerr=np.sqrt(eerr**2 + (Aerr/escale)**2), fmt="none")
    #plt.legend((p1,),("e-12*A",))
    #plt.savefig("plots/eresidual.pdf")

    #plt.figure()
    #plt.errorbar(etasc,A/l**2,xerr=0.5*etasw, yerr=Aerr/l**2, fmt="none")

    #plt.figure()
    #plt.errorbar(etasc,-e/l**2,xerr=0.5*etasw, yerr=eerr/l**2, fmt="none")

    #plt.figure()
    #plt.errorbar(etasc,Y/l**2,xerr=0.5*etasw, yerr=Yerr/l**2, fmt="none")

    plt.figure()
    wscale = 200.
    p1 = plt.errorbar(etasc,-Z,xerr=0.5*etasw, yerr=Zerr, fmt="none")
    p2 = plt.errorbar(etasc,wscale*W,xerr=0.5*etasw, yerr=np.abs(wscale)*Werr, fmt="none")
    plt.legend((p1,p2),("-Z","200*W"))
    plt.xlabel("$\eta$")
    plt.savefig("plots/ZWcompare.pdf")

    #plt.figure()

    #plt.show()

    plt.figure()
    plt.errorbar(etasc,a**2*l**2,yerr=np.abs(2.*a)*aerr*l**2, xerr=0.5*etasw, fmt="none")
    plt.title("$a^2$ (With $1/L^2$ dependence)")
    plt.xlabel("$\eta$")
    plt.ylim(bottom=0.)
    plt.savefig("plots/ascaled.pdf")


    plt.figure()
    plt.errorbar(etasc,b**2*l**2,yerr=np.abs(2.*b)*berr*l**2, xerr=0.5*etasw, fmt="none")
    plt.title("$b^2$ (With $1/L^2$ dependence)")
    plt.xlabel("$\eta$")
    plt.ylim(bottom=0.)
    plt.savefig("plots/bscaled.pdf")

    plt.figure()
    plt.errorbar(etasc,c**2*l**4,yerr=np.abs(2.*c)*cerr*l**4, xerr=0.5*etasw, fmt="none")
    plt.title("$c^2$ (With $1/L^4$ dependence)")
    plt.ylim(bottom=0.)
    plt.savefig("plots/cscaled.pdf")

    plt.figure()
    plt.errorbar(etasc,d**2/l**2,yerr=np.abs(2.*d)*derr/l**2, xerr=0.5*etasw, fmt="none")
    plt.title("$d^2$ (With $L^2$ dependence)")
    plt.xlabel("$\eta$")
    plt.ylim(bottom=0.)
    plt.savefig("plots/dscaled.pdf")
    plt.title("$d^2$ (With $L^2$ dependence) (zoomed)")
    plt.ylim(top=1000.)
    plt.savefig("plots/dscaledzoom.pdf")
    
    
    #Yalt = np.sqrt(1.+Y**2) - 1.
    #Yerralt = np.abs(Y)/np.sqrt(1.+Y**2)*Yerr
    Yalt = Y
    Yerralt = Yerr
    plot = plt.figure()
    plt.errorbar(etasc,Yalt,yerr=Yerralt, xerr=0.5*etasw, fmt="none")
    plt.title("Y")
    plt.xlabel("$\eta$")
    plt.savefig("plots/Yscaled.pdf")
    #plt.ylim(bottom=0., top=1e5)
    #plt.title("Y (zoomed)")
    #plt.savefig("plots/Yscaledzoom.pdf")
    #plt.ylim(bottom=0., top=5e3)
    #plt.title("Y (more zoomed)")
    #plt.savefig("plots/Yscaledzoom2.pdf")
    #plt.show()


    plt.figure()
    plt.errorbar(etasc, A,yerr=Aerr, xerr=0.5*etasw, fmt="none")
    plt.title("A (zoomed)")
    plt.xlabel("$\eta$")
    plt.ylim(top=0.001, bottom=-0.001)
    plt.savefig("plots/Azoomed.pdf")    

    #plot = plt.figure()
    #plt.errorbar(etasc,W,xerr=0.5*etasw, yerr=Werr, fmt="none")
    #plt.errorbar(etasc,c**2*l**4,xerr=0.5*etasw, yerr=2.*np.abs(c)*cerr*l**4, fmt="none")

    #plt.show()

    #plt.figure()
    #dscale = 1e-7/np.sqrt(325.)
    #plt.errorbar(etasc,dscale*np.abs(d),yerr=dscale*derr, xerr=0.5*etasw, fmt="none")
    #wscale = -1*sintheta
    #plt.errorbar(etasc,wscale*W,yerr=np.abs(wscale)*Werr, xerr=0.5*etasw, fmt="none")

    #wscale = -1.
    #plt.errorbar(etasc,wscale*W,yerr=np.abs(wscale)*Werr, xerr=0.5*etasw, fmt="none")

    #plt.show()


if False:
    A,e,M,W,Y,Z,a,g,c,d,e2,Z2 = xs.T
    Aerr,eerr,Merr,Werr,Yerr,Zerr,aerr,gerr,cerr,derr,e2err,Z2err = xerrs.T

    #Zmod = (1./64.)*Z
    #Zmoderr = (1./64.)*Zerr

    #Ymod = -16.*Y
    #Ymoderr = -16.*Yerr
    #e2/Z2

    plot = plt.figure()
    plt.errorbar(etasc,e2,xerr=0.5*etasw, yerr=e2err, fmt="none")
    plt.errorbar(etasc,-(1./4.)*Z2,xerr=0.5*etasw, yerr=(1./4.)*Z2err, fmt="none")

    #emod = e*sintheta
    #emoderr = eerr*sintheta
    emod = e
    emoderr = eerr


    plot = plt.figure()
    plt.errorbar(etasc,Z,xerr=0.5*etasw, yerr=Zerr, fmt="none")
    plt.errorbar(etasc,32*A,xerr=0.5*etasw, yerr=32*Aerr, fmt="none")
    plt.errorbar(etasc,2.*emod,xerr=0.5*etasw, yerr=2.*emoderr, fmt="none")
    #plt.errorbar(etasc,200e3*W,xerr=0.5*etasw, yerr=200e3*Werr, fmt="none")
    plt.errorbar(etasc,200e3*W,xerr=0.5*etasw, yerr=200e3*Werr, fmt="none")
    plt.errorbar(etasc,-640*Y,xerr=0.5*etasw, yerr=640*Yerr, fmt="none")
    #plt.errorbar(etasc,200e3*W/sintheta/l**2,xerr=0.5*etasw, yerr=200e3*Werr/sintheta/l**2, fmt="none")
    #plt.errorbar(etasc,-640*Y/sintheta/l**2,xerr=0.5*etasw, yerr=640*Yerr/sintheta/l**2, fmt="none")


    #plot = plt.figure()
    #scale = 1./l**2
    #plt.errorbar(etasc,scale*Z,xerr=0.5*etasw, yerr=scale*Zerr, fmt="none")
    #plt.errorbar(etasc,scale*32*A,xerr=0.5*etasw, yerr=scale*32*Aerr, fmt="none")
    #plt.errorbar(etasc,scale*2.*emod,xerr=0.5*etasw, yerr=scale*2.*emoderr, fmt="none")

    plt.figure()
    
    #wscale = 500./l**2/sintheta
    #yscale = 1./l**4
    yscale = -1.*sintheta
    #wscale = 500./l**2
    ##wscale = 400.
    #wscale = 1./100
    wscale = 256./l**2
    #yscale = -1./l**2
    plt.errorbar(etasc,yscale*Y,xerr=0.5*etasw, yerr=np.abs(yscale)*Yerr, fmt="none")
    plt.errorbar(etasc,wscale*W,xerr=0.5*etasw, yerr=wscale*Werr, fmt="none")
    #plt.plot(etasc,0.0001*l**4)
    #plt.errorbar(etasc,yscale*Y,xerr=0.5*etasw, yerr=np.abs(yscale)*Yerr, fmt="none")

    #scale = 1./l**2
    #plt.errorbar(etasc,scale*A,xerr=0.5*etasw, yerr=scale*Aerr, fmt="none")

    #plt.figure()
    #plt.plot(etasc,l)
    #plt.plot(etasc,l**2)
    #plt.plot(etasc,l**4)

    #plt.figure()
    #plt.plot(etasc,1./l)
    #plt.plot(etasc,1./l**2)
    #plt.plot(etasc,1./l**4)

    #plt.figure()
    #plt.plot(etasc,sintheta)
    #plt.plot(etasc,sintheta**2)

    #plt.figure()
    #plt.plot(etasc,1./sintheta)
    #plt.plot(etasc,1./sintheta**2)

    #plt.figure()
    ##plt.plot(etasc,l**2/sintheta**2)
    #scale = -1./l
    #plt.errorbar(etasc,scale*Y,xerr=0.5*etasw, yerr=np.abs(scale)*Yerr, fmt="none")


    #plt.figure()
    #plt.errorbar(etasc,Z/l**2,xerr=0.5*etasw, yerr=Zerr/l**2, fmt="none")

    #plt.figure()
    #plt.errorbar(etasc,-Y/sintheta/l**2,xerr=0.5*etasw, yerr=Yerr/sintheta/l**2, fmt="none")
    #plt.errorbar(etasc,384*W,xerr=0.5*etasw, yerr=384*Werr, fmt="none")

    #plt.figure()
    #plt.errorbar(etasc,W/sintheta/l**2,xerr=0.5*etasw, yerr=Werr/sintheta/l**2, fmt="none")
    #plt.errorbar(etasc,384*W,xerr=0.5*etasw, yerr=384*Werr, fmt="none")


    #plt.figure()
    #plt.errorbar(etasc,-200.*W/l**2,xerr=0.5*etasw, yerr=-200.*Werr/l**2, fmt="none")


    #plt.figure()

    #plt.errorbar(etasc,A/l**2,xerr=0.5*etasw, yerr=Aerr/l**2, fmt="none")


    #plt.figure()


    #plt.show()


    plot = plt.figure()
    plt.errorbar(etasc,192.*W,xerr=0.5*etasw, yerr=192.*Werr, fmt="none")
    plt.errorbar(etasc,-Y,xerr=0.5*etasw, yerr=Yerr, fmt="none")
    #plt.errorbar(etasc,Z/3.,xerr=0.5*etasw, yerr=Zerr/3., fmt="none")

    #plt.errorbar(etasc,16.*A,xerr=0.5*etasw, yerr=16.*Aerr, fmt="none")
    #plt.errorbar(etasc,emod,xerr=0.5*etasw, yerr=emoderr, fmt="none")

    plt.show()

    Zmod = Z*costheta**2/3.
    Zmoderr = Zerr*costheta**2/3.

    Zmod2 = Z*sintheta**2/3.
    Zmoderr2 = Zerr*sintheta**2/3.



    plot = plt.figure()
    plt.errorbar(etasc,emod,xerr=0.5*etasw, yerr=emoderr, fmt="none")
    plt.errorbar(etasc,Zmod,xerr=0.5*etasw, yerr=Zmoderr, fmt="none")
    plt.errorbar(etasc,Zmod2,xerr=0.5*etasw, yerr=Zmoderr2, fmt="none")


    plt.figure()
    plt.errorbar(etasc,16.*Y**2,xerr=0.5*etasw, yerr=16.*2.*Y*Yerr, fmt="none")
    plt.errorbar(etasc,W,xerr=0.5*etasw, yerr=Werr, fmt="none")

    #amod = a*l**2
    #amoderr = aerr*l**2

    #plt.figure()
    #plt.errorbar(etasc,e,xerr=0.5*etasw, yerr=eerr, fmt="none")
    #plt.errorbar(etasc,amod,xerr=0.5*etasw, yerr=amoderr, fmt="none")

    plt.show()


    plt.figure()
    plt.errorbar(etasc,a*l**2,xerr=0.5*etasw, yerr=aerr*l**2, fmt="none")

    plot = plt.figure()
    plt.errorbar(etasc,W,xerr=0.5*etasw, yerr=Werr, fmt="none")
    plt.errorbar(etasc,c**2,xerr=0.5*etasw, yerr=2.*np.abs(c)*cerr, fmt="none")
    #plt.errorbar(etasc,-384.*W,xerr=0.5*etasw, yerr=384.*Werr, fmt="none")

    #Amod = A/3.
    #Aerrmod = Aerr/3.

    #plt.errorbar(etasc,A/Y,xerr=0.5*etasw, yerr=None, fmt="none")
    #plt.errorbar(etasc,A,xerr=0.5*etasw, yerr=Aerr, fmt="none")
    #plt.errorbar(etasc,-16.*Y,xerr=0.5*etasw, yerr=16.*Yerr, fmt="none")
    #plt.errorbar(etasc,Zmod,xerr=0.5*etasw, yerr=Zmoderr, fmt="none")
    #plt.errorbar(etasc,Ymod,xerr=0.5*etasw, yerr=Ymoderr, fmt="none")

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
    
    ksplus = hdsetks[ieta,1,:]
    ksminus = hdsetks[ieta,0,:]
    
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
    ##scaleplusalt = -M/ksfine - 0.5*sigmamodelfineminus**2
    ##scaleplusalt = -M/ksfine -Z/ksfine-e*ksfine  - Y*ksfine**2  -A  -W/ksfine**2
    ##A + q*M/k + e*k + W/k**2 + Y*k**2 + Z/k + q*V + q*e2*k + q*Z2*k**2
    ##scaleplusalt = +M/ksfine +Y*ksfine**2 +A  +e*ksfine 
    #scaleplusalt = +M/ksfine +Y*ksfine**2 +e*ksfine  + Z/ksfine + A + sigmamodelfineplus
    ##scaleplusalt = Z/ksfine + W/ksfine**2
    
    plot = plt.figure()
    plt.fill_between(ksfine,scalemodelfineplus-scalemodelfinepluserr-1.,scalemodelfineplus+scalemodelfinepluserr-1., alpha=0.5)
    plt.plot(ksfine, scalemodelfineplus-1.)
    plt.fill_between(-ksfine,scalemodelfineminus-scalemodelfineminuserr-1.,scalemodelfineminus+scalemodelfineminuserr-1., alpha=0.5)
    plt.plot(-ksfine, scalemodelfineminus-1.)
    plt.errorbar(ksplus,scalebinnedplus-1., xerr=xerrplus, yerr=scalebinnedpluserr,fmt='none')
    plt.errorbar(-ksminus,scalebinnedminus-1., xerr=-xerrminus, yerr=scalebinnedminuserr,fmt='none')
    plt.xlabel("$q\ k$ (GeV$^{-1}$)")
    plt.ylabel("$\delta k/k$")
    plt.title(f"Scale, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
    
    #p1 = plt.plot(ksfine, scaleplusalt)
    #plt.legend((p1,),('$-M/k - 0.5\sigma^2$',))
    #plt.show()
    plt.savefig(f"plots/scale_{ieta}.{imgtype}")
    
    plot = plt.figure()
    plt.fill_between(1./ksfine,scalemodelfineplus-scalemodelfinepluserr-1.,scalemodelfineplus+scalemodelfinepluserr-1., alpha=0.5)
    plt.plot(1./ksfine, scalemodelfineplus-1.)
    plt.fill_between(-1./ksfine,scalemodelfineminus-scalemodelfineminuserr-1.,scalemodelfineminus+scalemodelfineminuserr-1., alpha=0.5)
    plt.plot(-1./ksfine, scalemodelfineminus-1.)
    plt.errorbar(1./ksplus,scalebinnedplus-1., xerr=xerrpluspt, yerr=scalebinnedpluserr,fmt='none')
    plt.errorbar(-1./ksminus,scalebinnedminus-1., xerr=-xerrminuspt, yerr=scalebinnedminuserr,fmt='none')
    plt.xlabel("$q\ p_T$ (GeV)")
    plt.ylabel("$\delta k/k$")
    plt.title(f"Scale, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
    #plt.plot(1./ksfine,scaleplusalt)
    plt.savefig(f"plots/scalept_{ieta}.{imgtype}")    
    #p1 = plt.plot(1./ksfine, scaleplusalt)
    #plt.legend( (p1[0],),('$M/k + 0.5\sigma^2$',))
    #plt.savefig(f"plots/scalerescompare.pdf")    
    #plt.show()    

    if True:
        
        plot = plt.figure()
        plt.fill_between(ksfine,sigmamodelfineplus-sigmamodelfinepluserr,sigmamodelfineplus+sigmamodelfinepluserr, alpha=0.5)
        plt.plot(ksfine, sigmamodelfineplus)
        plt.fill_between(-ksfine,sigmamodelfineminus-sigmamodelfineminuserr,sigmamodelfineminus+sigmamodelfineminuserr, alpha=0.5)
        plt.plot(-ksfine, sigmamodelfineminus)
        plt.errorbar(ksplus,sigmabinnedplus, xerr=xerrplus, yerr=sigmabinnedpluserr,fmt='none')
        plt.errorbar(-ksminus,sigmabinnedminus, xerr=-xerrminus, yerr=sigmabinnedminuserr,fmt='none')
        plt.xlabel("$q\ k$ (GeV$^{-1}$)")
        plt.ylabel("$\sigma^2_k/k^2$")
        plt.title(f"Resolution, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
        plt.savefig(f"plots/res_{ieta}.{imgtype}")
        
        plot = plt.figure()
        plt.fill_between(1./ksfine,sigmamodelfineplus-sigmamodelfinepluserr,sigmamodelfineplus+sigmamodelfinepluserr, alpha=0.5)
        plt.plot(1./ksfine, sigmamodelfineplus)
        plt.fill_between(-1./ksfine,sigmamodelfineminus-sigmamodelfineminuserr,sigmamodelfineminus+sigmamodelfineminuserr, alpha=0.5)
        plt.plot(-1./ksfine, sigmamodelfineminus)
        plt.errorbar(1./ksplus,sigmabinnedplus, xerr=xerrpluspt, yerr=sigmabinnedpluserr,fmt='none')
        plt.errorbar(-1./ksminus,sigmabinnedminus, xerr=-xerrminuspt, yerr=sigmabinnedminuserr,fmt='none')
        plt.xlabel("$q\ p_T$ (GeV)")
        plt.ylabel("$\sigma^2_k/k^2$")
        plt.title(f"Resolution, ${etasl[ieta]:.1f}\leq \eta < {etash[ieta]:.1f}$")
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

plt.show()
