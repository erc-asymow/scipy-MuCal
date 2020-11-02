import sympy
#from sympy.tensor.array import derive_by_array
from sympy.printing.cxxcode import cxxcode

dogen=False

N=5

nhits = N

nparsAlignment = 2*nhits
nparsBfield = nhits - 1
nparsEloss = nhits - 1
nstateparms = 5*nhits
npropparms = 5*(nhits-1)
nhitparms = 2*nhits
nmomparms = 3*(nhits-1)
nposparms = 2*(nhits-1)
nrefparms = 5

dy0 = sympy.MatrixSymbol("dy0",nhitparms,1)
dx = sympy.MatrixSymbol("dx",nstateparms,1)
dx0pos = sympy.MatrixSymbol("dx0pos",nposparms,1)
dx0mom = sympy.MatrixSymbol("dx0mom",nmomparms,1)
E = sympy.MatrixSymbol("E",nmomparms,nmomparms)
#H = sympy.MatrixSymbol("H",nstateparms,nstateparms)
#Hprop = sympy.MatrixSymbol("Hprop",nstateparms,nstateparms)
Hh = sympy.MatrixSymbol("Hh",nhitparms,nstateparms)
Hpos = sympy.MatrixSymbol("Hpos",nposparms,nstateparms)
Hmom = sympy.MatrixSymbol("Hmom",nmomparms,nstateparms)
Hproppos = sympy.MatrixSymbol("Hproppos",nposparms,npropparms)
Hpropmom = sympy.MatrixSymbol("Hpropmom",nmomparms,npropparms)
F = sympy.MatrixSymbol("F",npropparms,nstateparms)
dF = sympy.MatrixSymbol("dF",npropparms,nparsBfield)
dE = sympy.MatrixSymbol("dE",nmomparms,nparsEloss)

A = sympy.MatrixSymbol("A",nhitparms,nparsAlignment)
dbeta = sympy.MatrixSymbol("dbeta",nparsBfield,1)
dxi = sympy.MatrixSymbol("dxi",nparsEloss,1)
dalpha = sympy.MatrixSymbol("dalpha",nparsAlignment,1)
Vinv = sympy.MatrixSymbol("Vinv",nhitparms,nhitparms)
Vinvsym = (Vinv+Vinv.T)/2

Qinv = sympy.MatrixSymbol("Qinv",nmomparms,nmomparms)
Qinvsym = (Qinv+Qinv.T)/2

Qinvpos = sympy.MatrixSymbol("Qinvpos",nposparms,nposparms)
Qinvpossym = (Qinvpos+Qinvpos.T)/2

lam = sympy.MatrixSymbol("lam", nposparms, 1)
# P = sympy.MatrixSymbol("P", nposparms, nstateparms)

Fref = sympy.MatrixSymbol("Fref",nrefparms,nstateparms)
dxgen0 = sympy.MatrixSymbol("dxgen0",nrefparms,1)
lamgen = sympy.MatrixSymbol("lamgen",nrefparms,1)
Cinvgen = sympy.MatrixSymbol("Cinvgen",nrefparms,nrefparms)


dh = dy0 - Hh*dx - A*dalpha
dmom = dx0mom + (Hmom-E*Hpropmom*F)*dx - E*Hpropmom*dF*dbeta - dE*dxi
dpos = dx0pos + (Hpos-Hproppos*F)*dx - Hproppos*dF*dbeta
dgen = dxgen0 + Fref*dx

chisq = dh.T*Vinvsym*dh + dmom.T*Qinvsym*dmom + dpos.T*Qinvpossym*dpos
#chisq = dh.T*Vinvsym*dh + dmom.T*Qinvsym*dmom + lam.T*dpos
if dogen:
  chisq += lamgen.T*dgen

#workaround because solve doesn't work
#dchisqdx = sympy.diff(chisq,dx)
#rhs = -dchisqdx.subs(dx,dx*0)
#lhs = dchisqdx.subs([(dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0),(dx,sympy.Identity(5*N))])
#lhs = sympy.Identity(5*N)*lhs
#Cfull = lhs.inverse()

results = []
labels = []

gx = sympy.diff(chisq,dx)
gx = gx.subs([(dx, dx*0), (lam, lam*0), (lamgen, lamgen*0)])
results.append(gx)
labels.append("gx")

glam = sympy.diff(chisq,lam)
glam = glam.subs([(dx, dx*0), (lam, lam*0), (lamgen, lamgen*0)])
results.append(glam)
labels.append("glam")

glamgen = sympy.diff(chisq,lamgen)
glamgen = glamgen.subs([(dx, dx*0), (lam, lam*0), (lamgen, lamgen*0)])
results.append(glamgen)
labels.append("glamgen")

#g = sympy.BlockMatrix([[gx],[glam]])

#print(g)
#print(g.shape)

#Kinv= sympy.MatrixSymbol("Kinv",5*N + 2*(N-1), 5*N + 2*(N-1))

#xlam = -Kinv.inverse()*g

#print(xlam)

#drv = sympy.diff(xlam,dxi)
#print(drv)

#assert(0)

#chisqmod = chisq
#chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(lam,lam*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
d2chisqdx2 = sympy.diff(chisqmod,dx,2)
results.append(d2chisqdx2)
labels.append("d2chisqdx2")

#chisqmod = chisq
chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
d2chisqdxdlam = sympy.diff(chisqmod,dx,lam)
results.append(d2chisqdxdlam)
labels.append("d2chisqdxdlam")

chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
d2chisqdxdlamgen = sympy.diff(chisqmod,dx,lamgen)
results.append(d2chisqdxdlamgen)
labels.append("d2chisqdxdlamgen")

#this one is zero
#chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
#d2chisqdlamdlamgen = sympy.diff(chisqmod,lam,lamgen)
#results.append(d2chisqdlamdlamgen)
#labels.append("d2chisqdlamdlamgen")

#C = Cfull
#C = sympy.MatrixSymbol("C",5*N,5*N)
Kxx = sympy.MatrixSymbol("Kxx",nstateparms,nstateparms)
Klamlam = sympy.MatrixSymbol("Klamlam",nposparms, nposparms)
Klamgenlamgen = sympy.MatrixSymbol("Klamgenlamgen",5, 5)
Kxlam = sympy.MatrixSymbol("Kxlam",nstateparms,nposparms)
Kxlamgen = sympy.MatrixSymbol("Kxlamgen",nstateparms,5)
Klamlamgen = sympy.MatrixSymbol("Klamlamgen",nposparms,5)
C = sympy.MatrixSymbol("C",nstateparms,nstateparms)

#dxprofiled = sympy.Inverse(lhs)*rhs
#dxprofiled = C*rhs

#if dogen:
  #dxprofiled = -Kxx*gx - Kxlam*glam - Kxlamgen*glamgen
  #lamprofiled = -Kxlam.T*gx - Klamlam*glam - Klamlamgen*glamgen
  #lamgenprofiled = -Kxlamgen.T*gx - Klamlamgen.T*glam - Klamgenlamgen*glamgen
  #chisq = chisq.subs([(dx,dxprofiled), (lam,lamprofiled), (lamgen,lamgenprofiled)])
#else:
  #dxprofiled = -Kxx*gx - Kxlam*glam
  #lamprofiled = -Kxlam.T*gx - Klamlam*glam
  #chisq = chisq.subs([(dx,dxprofiled), (lam,lamprofiled)])

dxprofiled = -C*gx
chisq = chisq.subs([(dx,dxprofiled)])

#print(chisq)
#assert(0)
                   
                   

#results.append(Cfull)

#dxprofiledmod = 1*dxprofiled.subs([(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
#dxprofiledmod = dxprofiled
dxdy0 = sympy.diff(dxprofiled,dy0).transpose()
#results.append(dxdy0)
#labels.append("dxdy0")

#dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0pos,dx0pos*0), (dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
dxdx0mom = sympy.diff(dxprofiled,dx0mom).transpose()
#results.append(dxdx0mom)
#labels.append("dxdx0mom")

#dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0mom,dx0mom*0), (dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
dxdx0pos = sympy.diff(dxprofiled,dx0pos).transpose()
#results.append(dxdx0pos)
#labels.append("dxdx0pos")

#dxref = Fref*(dxdy0*dy0 + dxdx0mom*dx0mom + dxdx0pos*dx0pos)
dxref = Fref*dxprofiled
results.append(dxref)
labels.append("dxref")

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dbeta,dbeta*0),(dxi,dxi*0)])
dxdalpha = sympy.diff(dxprofiledmod,dalpha).transpose()
dxrefdalpha = Fref*dxdalpha
results.append(dxrefdalpha)
labels.append("dxrefdalpha")

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0),(dxi,dxi*0)])
dxdbeta = sympy.diff(dxprofiledmod,dbeta).transpose()
dxrefdbeta = Fref*dxdbeta
results.append(dxrefdbeta)
labels.append("dxrefdbeta")

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0),(dbeta,dbeta*0)])
dxdxi = sympy.diff(dxprofiledmod,dxi).transpose()
dxrefdxi = Fref*dxdxi
results.append(dxrefdxi)
labels.append("dxrefdxi")

dchisqdalpha = sympy.diff(chisq,dalpha)
results.append(dchisqdalpha)
labels.append("dchisqdalpha")

dchisqdbeta = sympy.diff(chisq,dbeta)
results.append(dchisqdbeta)
labels.append("dchisqdbeta")

dchisqdxi = sympy.diff(chisq,dxi)
results.append(dchisqdxi)
labels.append("dchisqdxi")

#print(dchisqdalpha)

chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dbeta,dbeta*0),(dxi,dxi*0)])
d2chisqdalpha2 = sympy.diff(chisqmod,dalpha,2)
results.append(d2chisqdalpha2)
labels.append("d2chisqdalpha2")

chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0),(dxi,dxi*0)])
d2chisqdbeta2 = sympy.diff(chisqmod,dbeta,2)
results.append(d2chisqdbeta2)
labels.append("d2chisqdbeta2")

chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0),(dbeta,dbeta*0)])
d2chisqdxi2 = sympy.diff(chisqmod,dxi,2)
results.append(d2chisqdxi2)
labels.append("d2chisqdxi2")

chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dxi,dxi*0)])
d2chisqdalphadbeta = sympy.diff(chisqmod,dalpha,dbeta)
results.append(d2chisqdalphadbeta)
labels.append("d2chisqdalphadbeta")

chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dbeta,dbeta*0)])
d2chisqdalphadxi = sympy.diff(chisqmod,dalpha,dxi)
results.append(d2chisqdalphadxi)
labels.append("d2chisqdalphadxi")

chisqmod = 1*chisq.subs([(dxgen0,dxgen0*0), (dy0,dy0*0),(dx0mom,dx0mom*0), (dx0pos,dx0pos*0),(dalpha,dalpha*0)])
d2chisqdbetadxi = sympy.diff(chisqmod,dbeta,dxi)
results.append(d2chisqdbetadxi)
labels.append("d2chisqdbetadxi")


#print(Cfull)
results2 = []
for i,(label,result) in enumerate(zip(labels,results)):
  #result = result.subs([(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0),(Vinv+Vinv.T, 2*Vinv), (Qinv+Qinv.T, 2*Qinv)])
  result = result.subs([(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0),(Vinv.T, Vinv), (Qinv.T, Qinv), (Kxx.T, Kxx), (Klamlam.T, Klamlam),(Qinvpos.T, Qinvpos)])
  #result = result.subs(C,Cplac)
  #result = sympy.Identity(result.shape[0])*result
  result = 1*result
  results2.append(result)
  cxxres = cxxcode(result,standard='C++11').replace(".T",".transpose()")
  #print(f"auto const& res_{i} = {result};")
  print(f"auto const& {label} = {cxxres};")
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
