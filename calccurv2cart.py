import sympy
from sympy.printing.cxx import cxxcode
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D

#global cartesian coordinate system
coords = CoordSys3D("coords")



# initial momentum direction components as constants
W0x = sympy.Symbol("W0x")
W0y = sympy.Symbol("W0y")
W0z = sympy.Symbol("W0z")
W0 = W0x*coords.i + W0y*coords.j + W0z*coords.k
U0 = coords.k.cross(W0).normalize()
V0 = W0.cross(U0)


qop0 = sympy.Symbol("qop0", nonzero=True)
lam0 = sympy.Symbol("lam0")
phi0 = sympy.Symbol("phi0")
xt0 = sympy.Symbol("xt0")
yt0 = sympy.Symbol("yt0")



#initial position
M0 = xt0*U0 + yt0*V0
#M0 = r0 + x0*J0 + y0*K0
#initial momentum direction
T0 = sympy.cos(lam0)*sympy.cos(phi0)*coords.i + sympy.cos(lam0)*sympy.sin(phi0)*coords.j + sympy.sin(lam0)*coords.k
#T0 = localpzsign*(I0 + dxdz0*J0 + dydz0*K0)/sympy.sqrt(1 + dxdz0*dxdz0 + dydz0*dydz0)


x = M0.dot(coords.i)
y = M0.dot(coords.j)
z = M0.dot(coords.k)

p = 1/abs(qop0)

px = p*sympy.cos(lam0)*sympy.cos(phi0)
py = p*sympy.cos(lam0)*sympy.sin(phi0)
pz = p*sympy.sin(lam0)


parms = [x, y, z, px, py, pz]
parmlabels = ["x", "y", "z", "px", "py", "pz"]

inparms = [qop0, lam0, phi0, xt0, yt0]
inparmlabels = ["qop0", "lam0", "phi0", "xt0", "yt0"]

results = []
labels = []


print("final derivatives")
for parm,parmlabel in zip(parms, parmlabels):
    print(parmlabel)
    #parm = parm.simplify()
    for inparm,inparmlabel,in zip(inparms,inparmlabels):
        print(parmlabel, inparmlabel)
        dparmdinparm = 1*sympy.diff(parm, inparm)
        #dparmdinparm = dparmdinparm.subs(subsconst)
        
        res = dparmdinparm
        
        #res = dparmdinparm + dparmds*dsdinparm
        #res = res.subs(subsconst)
        #res = res.subs(subsrev)
        #res = res.subs([(lam0, lam0val), (phi0, phi0val), (xt0, xt0val), (yt0, yt0val)])
        res = 1*res
        res = res.simplify()
        
        label = f"d{parmlabel}d{inparmlabel}"
        results.append(res)
        labels.append(label)

#for res, label in zip(results, labels):
  #print(f"const double {label} = {cxxcode(res,standard='C++11')};")

#substitutions, results2 = sympy.cse(results,symbols = numbered_symbols("xf"))
substitutions, results2 = sympy.cse(results)
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


                
                
        

