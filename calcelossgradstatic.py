import sympy
#from sympy.printing import print_ccode
from sympy.printing.cxx import cxxcode

#q = sympy.symbols("q")
qop = sympy.symbols("qop",nonzero=True)
dxdz = sympy.symbols("dxdz")
dydz = sympy.symbols("dydz")
xi = sympy.symbols("xi")

qopval = sympy.symbols("qopval")
dxdzval = sympy.symbols("dxdzval")
dydzval = sympy.symbols("dydzval")

subsconst = [(qop, qopval), (dxdz, dxdzval), (dydz, dydzval)]
subsrev = [(qopval, qop), (dxdzval, dxdz), (dydzval, dydz)]




q = sympy.sign(qop)
#q = sympy.symbols("q")

m2 = sympy.symbols("m2")

emass = sympy.symbols("emass")
poti = sympy.symbols("poti")
eplasma = sympy.symbols("eplasma")

p2 = 1/qop**2
abspz = 1/(qop/q * sympy.sqrt(1+ dxdz**2 + dydz**2))

##p2 = sympy.symbols("p2")
#abspz = sympy.symbols("abspz")


delta0 = 2*sympy.log(eplasma/poti) - 1
xf = sympy.sqrt(p2)/abspz

im2 = 1/m2
e2 = p2 + m2
e = sympy.sqrt(e2)
beta2 = p2/e2
eta2 = p2*im2
ratio2 = emass**2*im2
emax = 2*emass*eta2/(1+2*emass*e*im2+ratio2)

ximod = xi*xf/beta2

dEdx = ximod*(sympy.log(2*emass*emax/poti**2) - 2*beta2 - delta0)

dP = dEdx/sympy.sqrt(beta2)

deltaP = -dP
#going against momentum, so extra minus sign
#deltaP *= -1

deltaP = sympy.simplify(deltaP)

deltaP = deltaP.subs(subsconst)


#pin = sympy.abs(1/qop)
#q = 1
#qopout = q/(q/qop + deltaP)

pin = q/qop
pout = pin + deltaP
qopout = q/pout

#print(qopout)


#print(deltaP)
#dqopoutdqop = sympy.simplify(sympy.diff(deltaP, qop))
dqopoutdqop = sympy.diff(qopout, qop)
dqopoutddxdz = sympy.diff(qopout, dxdz)
dqopoutddydz = sympy.diff(qopout, dydz)
dqopoutdxi = sympy.diff(qopout, xi)


dqopoutdqop = dqopoutdqop.subs(subsrev)
dqopoutddxdz = dqopoutddxdz.subs(subsrev)
dqopoutddydz = dqopoutddydz.subs(subsrev)
dqopoutdxi = dqopoutdxi.subs(subsrev)


#print("d qopout / d qop:")
#print(cxxcode(dqopoutdqop, standard='C++11'))
#print("d qopout / d dxdz:")
#print(cxxcode(dqopoutddxdz, standard='C++11'))
#print("d qopout / d dydz:")
#print(cxxcode(dqopoutddydz, standard='C++11'))

#print("d qopout / d xi:")
#print(cxxcode(dqopoutdxi, standard='C++11'))

resultspre = [dqopoutdqop,dqopoutddxdz,dqopoutddydz,dqopoutdxi]
for res in resultspre:
  print(res)

#collect gradients and simplify with common term substitution
substitutions, results = sympy.cse(resultspre)
#loop through output and translate to C++ code
for sub in substitutions:
  #print(sub[1])
  print(f"const double {sub[0]} = {cxxcode(sub[1],standard='C++17')};")
for i,res in enumerate(results):
  print(f"const double res_{i} = {cxxcode(res,standard='C++17')};")

