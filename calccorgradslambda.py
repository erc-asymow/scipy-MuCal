import sympy
#from sympy.tensor.array import derive_by_array
from sympy.printing.cxxcode import cxxcode


N=5

dy0 = sympy.MatrixSymbol("dy0",5*N,1)
dx = sympy.MatrixSymbol("dx",5*N,1)
dx0 = sympy.MatrixSymbol("dx0",5*N,1)
H = sympy.MatrixSymbol("H",5*N,5*N)
E = sympy.MatrixSymbol("E",5*N,5*N)
Hprop = sympy.MatrixSymbol("Hprop",5*N,5*N)
F = sympy.MatrixSymbol("F",5*N,5*N)
dF = sympy.MatrixSymbol("dF",5*N,N)
dE = sympy.MatrixSymbol("dE",5*N,N)

A = sympy.MatrixSymbol("A",5*N,N)
dbeta = sympy.MatrixSymbol("dbeta",N,1)
dxi = sympy.MatrixSymbol("dxi",N,1)
dalpha = sympy.MatrixSymbol("dalpha",N,1)
Vinv = sympy.MatrixSymbol("Vinv",5*N,5*N)
Vinvsym = (Vinv+Vinv.T)/2

Qinv = sympy.MatrixSymbol("Qinv",5*N,5*N)
Qinvsym = (Qinv+Qinv.T)/2

lam = sympy.MatrixSymbol("lam", 2*(N-1), 1)
P = sympy.MatrixSymbol("P", 2*(N-1), 5*N)

dh = dy0 - H*dx - A*dalpha
dms = dx0 + (H-E*Hprop*F)*dx - E*Hprop*dF*dbeta - dE*dxi

chisq = dh.T*Vinvsym*dh + dms.T*Qinvsym*dms + lam.T*P*dms

#workaround because solve doesn't work
#dchisqdx = sympy.diff(chisq,dx)
#rhs = -dchisqdx.subs(dx,dx*0)
#lhs = dchisqdx.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0),(dx,sympy.Identity(5*N))])
#lhs = sympy.Identity(5*N)*lhs
#Cfull = lhs.inverse()

results = []
labels = []

gx = sympy.diff(chisq,dx)
gx = gx.subs([(dx, dx*0), (lam, lam*0)])
results.append(gx)
labels.append("gx")

glam = sympy.diff(chisq,lam)
glam = glam.subs([(dx, dx*0), (lam, lam*0)])
results.append(glam)
labels.append("glam")

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(lam,lam*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
d2chisqdx2 = sympy.diff(chisqmod,dx,2)
results.append(d2chisqdx2)
labels.append("d2chisqdx2")

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
d2chisqdxdlam = sympy.diff(chisqmod,dx,lam)
results.append(d2chisqdxdlam)
labels.append("d2chisqdxdlam")

#C = Cfull
#C = sympy.MatrixSymbol("C",5*N,5*N)
Kxx = sympy.MatrixSymbol("Kxx",5*N,5*N)
Kxlam = sympy.MatrixSymbol("Kxlam",5*N,2*(N-1))
Klamlam = sympy.MatrixSymbol("Klamlam",2*(N-1),2*(N-1))


#dxprofiled = sympy.Inverse(lhs)*rhs
#dxprofiled = C*rhs
dxprofiled = -Kxx*gx - Kxlam*glam
lamprofiled = -Kxlam.T*gx - Klamlam*glam
chisq = chisq.subs([(dx,dxprofiled), (lam,lamprofiled)])

#print(chisq)
#assert(0)
                   
                   

#results.append(Cfull)

dxprofiledmod = 1*dxprofiled.subs([(dx0,dx0*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
dxdy0 = sympy.diff(dxprofiledmod,dy0).transpose()
results.append(dxdy0)
labels.append("dxdy0")

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
dxdx0 = sympy.diff(dxprofiledmod,dx0).transpose()
results.append(dxdx0)
labels.append("dxdx0")

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0,dx0*0),(dbeta,dbeta*0),(dxi,dxi*0)])
dxdalpha = sympy.diff(dxprofiledmod,dalpha).transpose()
results.append(dxdalpha)
labels.append("dxdalpha")

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dxi,dxi*0)])
dxdbeta = sympy.diff(dxprofiledmod,dbeta).transpose()
results.append(dxdbeta)
labels.append("dxdbeta")

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dbeta,dbeta*0)])
dxdxi = sympy.diff(dxprofiledmod,dxi).transpose()
results.append(dxdxi)
labels.append("dxdxi")

dxref = dxdy0*dy0 + dxdx0*dx0
results.append(dxref)
labels.append("dxref")

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

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dbeta,dbeta*0),(dxi,dxi*0)])
d2chisqdalpha2 = sympy.diff(chisqmod,dalpha,2)
results.append(d2chisqdalpha2)
labels.append("d2chisqdalpha2")

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dxi,dxi*0)])
d2chisqdbeta2 = sympy.diff(chisqmod,dbeta,2)
results.append(d2chisqdbeta2)
labels.append("d2chisqdbeta2")

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dbeta,dbeta*0)])
d2chisqdxi2 = sympy.diff(chisqmod,dxi,2)
results.append(d2chisqdxi2)
labels.append("d2chisqdxi2")

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dxi,dxi*0)])
d2chisqdalphadbeta = sympy.diff(chisqmod,dalpha,dbeta)
results.append(d2chisqdalphadbeta)
labels.append("d2chisqdalphadbeta")

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dbeta,dbeta*0)])
d2chisqdalphadxi = sympy.diff(chisqmod,dalpha,dxi)
results.append(d2chisqdalphadxi)
labels.append("d2chisqdalphadxi")

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0)])
d2chisqdbetadxi = sympy.diff(chisqmod,dbeta,dxi)
results.append(d2chisqdbetadxi)
labels.append("d2chisqdbetadxi")


#print(Cfull)
results2 = []
for i,(label,result) in enumerate(zip(labels,results)):
  #result = result.subs([(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0),(Vinv+Vinv.T, 2*Vinv), (Qinv+Qinv.T, 2*Qinv)])
  result = result.subs([(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0),(Vinv.T, Vinv), (Qinv.T, Qinv), (Kxx.T, Kxx),(Klamlam.T, Klamlam)])
  #result = result.subs(C,Cplac)
  result = sympy.Identity(result.shape[0])*result
  #result = 1*result
  results2.append(result)
  cxxres = cxxcode(result,standard='C++11').replace(".T",".transpose()")
  #print(f"auto const& res_{i} = {result};")
  print(f"auto const& {label} = {cxxres};")
  

substitutions, results3 = sympy.cse(results2)
#loop through output and translate to C++ code
for sub in substitutions:
  #print(sub[1])
  print(f"auto const& {sub[0]} = {cxxcode(sub[1],standard='C++11')};")
for label,res in zip(labels,results3):
  print(f"auto const& {label} = {cxxcode(res,standard='C++11')};")

##soln = sympy.
#dx = sympy.solve(dchisqdx, dx, implicit=True)
#print(dx)





#dchisqdx = dchisqdx.expand()
#dchisqdx = dchisqdx.subs(Vinvsym, Vinv).simplify()
#dchisqdx = dchisqdx.subs(Vinv.T, Vinv).simplify()
#dchisqdx = sympy.expand(dchisqdx).simplify()

#print(dchisqdx)
