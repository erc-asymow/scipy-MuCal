import sympy
from sympy.printing.cxx import cxxcode
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D

#global cartesian coordinate system
coords = CoordSys3D("coords")

qop0 = sympy.Symbol("qop0", nonzero=True)
lam0 = sympy.Symbol("lam0", real=True)
phi0 = sympy.Symbol("phi0", real=True)

qop1 = sympy.Symbol("qop1", nonzero=True)
lam1 = sympy.Symbol("lam1", real=True)
phi1 = sympy.Symbol("phi1", real=True)

mp = sympy.Symbol("mp", nonzero=True)

p0 = 1/abs(qop0)
T0 = sympy.cos(lam0)*sympy.cos(phi0)*coords.i + sympy.cos(lam0)*sympy.sin(phi0)*coords.j + sympy.sin(lam0)*coords.k
P0 = p0*T0
e0 = sympy.sqrt(p0*p0 + mp*mp)

p1 = 1/abs(qop1)
T1 = sympy.cos(lam1)*sympy.cos(phi1)*coords.i + sympy.cos(lam1)*sympy.sin(phi1)*coords.j + sympy.sin(lam1)*coords.k
P1 = p1*T1
e1 = sympy.sqrt(p1*p1 + mp*mp)

#m = sympy.sqrt(2*mp**2 + 2*e0*e1 -2*P0.dot(P1))

m = sympy.sqrt(2*mp**2 + 2*e0*e1 -2*P0.dot(P1))

minvsq = 1./(2*mp**2 + 2*e0*e1 -2*P0.dot(P1))

#m = m.simplify()

print(m)

#parms = [m]
#parmlabels = ["m"]
parms = [minvsq]
parmlabels = ["minvsq"]

inparms = [qop0, lam0, phi0, qop1, lam1, phi1]
inparmlabels = ["qop0", "lam0", "phi0", "qop1", "lam1", "phi1"]


results = []
labels = []

#results.append(m)
#labels.append("m")

print("final derivatives")
for parm,parmlabel in zip(parms, parmlabels):
    print(parmlabel)
    #parm = parm.simplify()
    for inparm,inparmlabel in zip(inparms,inparmlabels):
        dparmdinparm = 1*sympy.diff(parm, inparm)
        res = dparmdinparm
        #res = res.simplify()
        label = f"d{parmlabel}d{inparmlabel}"
        
        
        results.append(res)
        labels.append(label)
        
substitutions, results2 = sympy.cse(results,symbols = numbered_symbols("xf"))
#substitutions, results2 = sympy.cse(results)
#loop through output and translate to C++ code
for sub in substitutions:
  #print(sub[1])
  cxxsub = cxxcode(sub[1],standard='C++17')
  print(f"const double {sub[0]} = {cxxsub};")
for res,label in zip(results2,labels):
  cxxres = cxxcode(res,standard='C++17')
  print(f"const double {label} = {cxxres};")


print(f"Matrix<double, {len(parms)}, {len(inparms)}> res;")
for i,parmlabel in enumerate(parmlabels):
    for j,inparmlabel in enumerate(inparmlabels):
        print(f"res({i},{j}) = d{parmlabel}d{inparmlabel};")
