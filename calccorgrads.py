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

dh = dy0 - H*dx - A*dalpha
dms = dx0 + (H-E*Hprop*F)*dx - E*Hprop*dF*dbeta - dE*dxi

chisq = dh.T*Vinvsym*dh + dms.T*Qinvsym*dms

#workaround because solve doesn't work
dchisqdx = sympy.diff(chisq,dx)
rhs = -dchisqdx.subs(dx,dx*0)
lhs = dchisqdx.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0),(dx,sympy.Identity(5*N))])
lhs = sympy.Identity(5*N)*lhs
Cfull = lhs.inverse()

#C = Cfull
C = sympy.MatrixSymbol("C",5*N,5*N)


#dxprofiled = sympy.Inverse(lhs)*rhs
dxprofiled = C*rhs
chisq = chisq.subs(dx,dxprofiled)

results = []
results.append(Cfull)

dxprofiledmod = 1*dxprofiled.subs([(dx0,dx0*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
dxdy0 = sympy.diff(dxprofiledmod,dy0).transpose()
results.append(dxdy0)

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0)])
dxdx0 = sympy.diff(dxprofiledmod,dx0).transpose()
results.append(dxdx0)

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0,dx0*0),(dbeta,dbeta*0),(dxi,dxi*0)])
dxdalpha = sympy.diff(dxprofiledmod,dalpha).transpose()
results.append(dxdalpha)

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dxi,dxi*0)])
dxdbeta = sympy.diff(dxprofiledmod,dbeta).transpose()
results.append(dxdbeta)

dxprofiledmod = 1*dxprofiled.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dbeta,dbeta*0)])
dxdxi = sympy.diff(dxprofiledmod,dxi).transpose()
results.append(dxdxi)

dxref = dxdy0*dy0 + dxdx0*dx0
results.append(dxref)

dchisqdalpha = sympy.diff(chisq,dalpha)
results.append(dchisqdalpha)

dchisqdbeta = sympy.diff(chisq,dbeta)
results.append(dchisqdbeta)

dchisqdxi = sympy.diff(chisq,dxi)
results.append(dchisqdxi)

#print(dchisqdalpha)

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dbeta,dbeta*0),(dxi,dxi*0)])
d2chisqdalpha2 = sympy.diff(chisqmod,dalpha,2)
results.append(d2chisqdalpha2)

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dxi,dxi*0)])
d2chisqdbeta2 = sympy.diff(chisqmod,dbeta,2)
results.append(d2chisqdbeta2)

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0),(dbeta,dbeta*0)])
d2chisqdxi2 = sympy.diff(chisqmod,dxi,2)
results.append(d2chisqdxi2)

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dxi,dxi*0)])
d2chisqdalphadbeta = sympy.diff(chisqmod,dalpha,dbeta)
results.append(d2chisqdalphadbeta)

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dbeta,dbeta*0)])
d2chisqdalphadxi = sympy.diff(chisqmod,dalpha,dxi)
results.append(d2chisqdalphadxi)

chisqmod = 1*chisq.subs([(dy0,dy0*0),(dx0,dx0*0),(dalpha,dalpha*0)])
d2chisqdbetadxi = sympy.diff(chisqmod,dbeta,dxi)
results.append(d2chisqdbetadxi)


#print(Cfull)
results2 = []
for i,result in enumerate(results):
  #result = result.subs([(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0),(Vinv+Vinv.T, 2*Vinv), (Qinv+Qinv.T, 2*Qinv)])
  result = result.subs([(dalpha,dalpha*0),(dbeta,dbeta*0),(dxi,dxi*0),(Vinv.T, Vinv), (Qinv.T, Qinv), (C.T, C)])
  #result = result.subs(C,Cplac)
  result = sympy.Identity(result.shape[0])*result
  results2.append(result)
  cxxres = cxxcode(result,standard='C++11').replace(".T",".transpose()")
  #print(f"auto const& res_{i} = {result};")
  print(f"auto const& res_{i} = {cxxres};")
  

#substitutions, results3 = sympy.cse(results2)
##loop through output and translate to C++ code
#for sub in substitutions:
  ##print(sub[1])
  #print(f"const double {sub[0]} = {cxxcode(sub[1],standard='C++11')};")
#for i,res in enumerate(results3):
  #print(f"const double res_{i} = {cxxcode(res,standard='C++11')};")

##soln = sympy.
#dx = sympy.solve(dchisqdx, dx, implicit=True)
#print(dx)





#dchisqdx = dchisqdx.expand()
#dchisqdx = dchisqdx.subs(Vinvsym, Vinv).simplify()
#dchisqdx = dchisqdx.subs(Vinv.T, Vinv).simplify()
#dchisqdx = sympy.expand(dchisqdx).simplify()

#print(dchisqdx)
