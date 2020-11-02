import sympy
#from sympy.tensor.array import derive_by_array
from sympy.printing.cxxcode import cxxcode

dogen=False

N=5

nhits = N

nparsAlignment = nhits + 3
nparsBfield = nhits - 1
nparsEloss = nhits - 1
nstateparms = 5*nhits
npropparms = 5*(nhits-1)
nhitparms = 2*nhits
nmomparms = 3*(nhits-1)
nposparms = 2*(nhits-1)
nrefparms = 5

dx = sympy.MatrixSymbol("dx",nstateparms,1)

dy0 = sympy.MatrixSymbol("dy0",nhitparms,1)
Hh = sympy.MatrixSymbol("Hh",nhitparms,nstateparms)
dxf0 = sympy.MatrixSymbol("dxf0",npropparms,1)
dxb0 = sympy.MatrixSymbol("dxb0",npropparms,1)
Ef = sympy.MatrixSymbol("Ef",npropparms,npropparms)
Eb = sympy.MatrixSymbol("Eb",npropparms,npropparms)
Huf = sympy.MatrixSymbol("Huf",npropparms,nstateparms)
Hub = sympy.MatrixSymbol("Hub",npropparms,nstateparms)
Hf = sympy.MatrixSymbol("Hf",npropparms,npropparms)
Hb = sympy.MatrixSymbol("Hb",npropparms,npropparms)
Ff = sympy.MatrixSymbol("Ff",npropparms,nstateparms)
Fb = sympy.MatrixSymbol("Fb",npropparms,nstateparms)
dFf = sympy.MatrixSymbol("dFf",npropparms,nparsBfield)
dFb = sympy.MatrixSymbol("dFb",npropparms,nparsBfield)
dEf = sympy.MatrixSymbol("dEf",npropparms,nparsEloss)
dEb = sympy.MatrixSymbol("dEb",npropparms,nparsEloss)

A = sympy.MatrixSymbol("A",nhitparms,nparsAlignment)
dbeta = sympy.MatrixSymbol("dbeta",nparsBfield,1)
dxi = sympy.MatrixSymbol("dxi",nparsEloss,1)
dalpha = sympy.MatrixSymbol("dalpha",nparsAlignment,1)

Vinv = sympy.MatrixSymbol("Vinv",nhitparms,nhitparms)
Qfinv = sympy.MatrixSymbol("Qfinv",npropparms,npropparms)
Qbinv = sympy.MatrixSymbol("Qbinv",npropparms,npropparms)

Fref = sympy.MatrixSymbol("Fref",nrefparms,nstateparms)

dh = dy0 - Hh*dx - A*dalpha
df = dxf0 + (Huf - Ef*Hf*Ff)*dx - Ef*Hf*dFf*dbeta - dEf*dxi
db = dxb0 + (Hub - Eb*Hb*Fb)*dx - Eb*Hb*dFb*dbeta - dEb*dxi

#chisq = dh.T*Vinv*dh
chisq = dh.T*Vinv*dh + df.T*Qfinv*df + db.T*Qbinv*db
#chisq = dh.T*Vinvsym*dh + df.T*Qfinvsym*df + db.T*Qbinvsym*db


symsubs = [(Vinv.T,Vinv), (Qfinv.T,Qfinv), (Qbinv.T,Qbinv)]
constvars = [dx, dy0, dxf0, dxb0, dalpha, dbeta, dxi]
constsubs = [(var,sympy.ZeroMatrix(*var.shape)) for var in [dalpha,dbeta,dxi]]

#constsubs = [(var, sympy.ZeroMatrix(*var.shape)) for var in constvars]
#print(constsubs)
#constsubs = [(dx,dx*0),(dy0,dy0*0),(dxf0,dxf0*0),(dxb0,dxb0*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)]
#constsubs = [(dx,dx*0),(dy0,dy0*0),(dxf0,dxf0*0),(dxb0,dxb0*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)]



results = []
labels = []


dchisqdx = sympy.diff(chisq,dx)

chisqmod = 1*chisq.subs([(var,sympy.ZeroMatrix(*var.shape)) for var in constvars if var not in [dx]])
d2chisqdx2 = sympy.diff(chisqmod, dx,2)

results.append(d2chisqdx2)
labels.append("d2chisqdx2")

C = sympy.MatrixSymbol("C",nstateparms,nstateparms)
dxprofiled = -C*dchisqdx

dxref = Fref*dxprofiled
results.append(dxref)
labels.append("dxref")

partdiffvars = [dalpha, dbeta, dxi]
for ivar in partdiffvars:
    dxprofiledmod = 1*dxprofiled.subs([(var,sympy.ZeroMatrix(*var.shape)) for var in constvars if var not in [ivar]])
    res = sympy.diff(dxprofiledmod,ivar)
    label = f"dxd{ivar}"
    results.append(res)
    labels.append(label)

chisq = chisq.subs(dx,dxprofiled)

for var in partdiffvars:
    res = sympy.diff(chisq,var)
    label = f"dchisqd{var}"
    results.append(res)
    labels.append(label)
  
for i in range(len(partdiffvars)):
    ivar = partdiffvars[i]
    chisqmod = 1*chisq.subs([(var,sympy.ZeroMatrix(*var.shape)) for var in constvars if var not in [ivar]])
    res = sympy.diff(chisqmod, ivar, 2)
    label = f"d2chisqd{ivar}2"
    results.append(res)
    labels.append(label)
  
for i in range(len(partdiffvars)):
    ivar = partdiffvars[i]
    for j in range(i+1, len(partdiffvars)):
        jvar = partdiffvars[j]
        chisqmod = 1*chisq.subs([(var,sympy.ZeroMatrix(*var.shape)) for var in constvars if var not in [ivar,jvar]])
        res = sympy.diff(chisqmod, ivar, jvar)
        label = f"d2chisqd{ivar}d{jvar}"
        results.append(res)
        labels.append(label)


#print(Cfull)
results2 = []
for i,(label,result) in enumerate(zip(labels,results)):
  #result = result.subs([(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0),(Vinv+Vinv.T, 2*Vinv), (Qinv+Qinv.T, 2*Qinv)])
  result = result.subs([*constsubs, *symsubs])
  #result = result.subs(C,Cplac)
  #result = sympy.Identity(result.shape[0])*result
  result = 1*result
  results2.append(result)
  cxxres = cxxcode(result,standard='C++11').replace(".T",".transpose()")
  #print(f"auto const& res_{i} = {result};")
  print(f"const MatrixXd {label} = {cxxres};")
  print("")
  
#assert(0)

#substitutions, results3 = sympy.cse(results2)
##loop through output and translate to C++ code
#for sub in substitutions:
  ##print(sub[1])
  #print(f"auto const& {sub[0]} = {cxxcode(sub[1],standard='C++11')};")
  #print("")
#for label,res in zip(labels,results3):
  #print(f"auto const& {label} = {cxxcode(res,standard='C++11')};")
  #print("")

##soln = sympy.
#dx = sympy.solve(dchisqdx, dx, implicit=True)
#print(dx)





#dchisqdx = dchisqdx.expand()
#dchisqdx = dchisqdx.subs(Vinvsym, Vinv).simplify()
#dchisqdx = dchisqdx.subs(Vinv.T, Vinv).simplify()
#dchisqdx = sympy.expand(dchisqdx).simplify()

#print(dchisqdx)
