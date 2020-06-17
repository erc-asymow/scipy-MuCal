import numpy as np
import matplotlib.pyplot as plt

fnames = []
#fnames.append("compres/unbinnedfiterrsmednodiag.npz")
#fnames.append("compres/unbinnedfiterrsmedfulldiag.npz")

#fnames.append("comprestriple/unbinnedfiterrsmeddiagfullreg.npz")
#fnames.append("comprestriple/unbinnedfiterrsmednodiagfull.npz")
#fnames.append("comprestriple/unbinnedfiterrsmednodiagfullreg.npz")

#fnames.append("comprescharged/unbinnedfiterrsmednodiagfull.npz")
#fnames.append("comprescharged/unbinnedfiterrsmeddiagfull.npz")
#fnames.append("comprescharged/unbinnedfiterrsmeddiag0124.npz")

#fnames.append("compreschargedtriple/unbinnedfiterrsmednodiagfull.npz")
fnames.append("compresmixedtriple/unbinnedfiterrsmedfulldiag.npz")
fnames.append("unbinnedfiterrsmed.npz")
#fnames.append("compreschargedtriple/unbinnedfiterrsmeddiag0124.npz")

#fnames.append("compres/unbinnedfiterrsmeddiag0124.npz")
#fnames.append("compres/unbinnedfiterrsmeddiag0124argsort.npz")

#fnames.append("compresnodiag/unbinnedfiterrsmed012.npz")
#fnames.append("compresnodiag/unbinnedfiterrsmed0.npz")
#fnames.append("compresnodiag/unbinnedfiterrsmed1.npz")
#fnames.append("compresnodiag/unbinnedfiterrsmed2.npz")
#fnames.append("compresnodiag/unbinnedfiterrsmed345.npz")
#fnames.append("compresnodiag/unbinnedfiterrsmed3.npz")
#fnames.append("compresnodiag/unbinnedfiterrsmed4.npz")
#fnames.append("compresnodiag/unbinnedfiterrsmed5.npz")
#fnames.append("compresnodiag/unbinnedfiterrsmedfull.npz")

#fnames.append("compresdiag/unbinnedfiterrsmedfull.npz")
#fnames.append("compresdiag/unbinnedfiterrsmed0.npz")
#fnames.append("compresdiag/unbinnedfiterrsmed1.npz")
#fnames.append("compresdiag/unbinnedfiterrsmed2.npz")
#fnames.append("compresdiag/unbinnedfiterrsmed3.npz")
#fnames.append("compresdiag/unbinnedfiterrsmed4.npz")
#fnames.append("compresdiag/unbinnedfiterrsmed5.npz")

#dlows = []
#dhighs = []

#dlowerrs = []
#dhigherrs = []

lds = []
lderrs = []

didxs = (0,1,2)

for fname in fnames:
    with np.load(fname) as f:
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

        ds = xs[:,didxs]
        derrs = xerrs[:,didxs]

        derrs = 0.5/np.sqrt(ds)*derrs
        ds = np.sqrt(ds)

        sortidxs = np.argsort(ds, axis=-1)

        ds = np.take_along_axis(ds,sortidxs,axis=-1)
        derrs = np.take_along_axis(derrs,sortidxs,axis=-1)
        
        lds.append(ds)
        lderrs.append(derrs)



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
        
        #dlows.append(dlow)
        #dlowerrs.append(dlowerr)
        
        #dhighs.append(dhigh)
        #dhigherrs.append(dhigherr)
    
    

plot = plt.figure()
plt.xlabel("$\eta$")
plt.title("d")
for vals,errs in zip(lds,lderrs):
    for val,err in zip(vals.T,errs.T):
        plt.errorbar(etasc, val, xerr=0.5*etasw, yerr=err,fmt='none')
    
#plot = plt.figure()
#plt.xlabel("$\eta$")
#plt.title("dhigh")
#for val,err in zip(dhighs,dhigherrs):
    #plt.errorbar(etasc, val, xerr=0.5*etasw, yerr=err,fmt='none')    
    
plt.show()
