import sympy
from sympy.printing.cxxcode import cxxcode
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D

coords = CoordSys3D("coords")

thetau = sympy.Symbol("thetau")
thetav = sympy.Symbol("thetav")

Wx = sympy.Symbol("Wx")
Wy = sympy.Symbol("Wy")
Wz = sympy.Symbol("Wz")
W = Wx*coords.i + Wy*coords.j + Wz*coords.k

Ux = sympy.Symbol("Ux")
Uy = sympy.Symbol("Uy")

Vx = sympy.Symbol("Vx")
Vy = sympy.Symbol("Vy")

Ix = sympy.Symbol("Ix")
Iy = sympy.Symbol("Iy")
Iz = sympy.Symbol("Iz")

Jx = sympy.Symbol("Jx")
Jy = sympy.Symbol("Jy")
Jz = sympy.Symbol("Jz")

Kx = sympy.Symbol("Kx")
Ky = sympy.Symbol("Ky")
Kz = sympy.Symbol("Kz")

U = Ux*coords.i + Uy*coords.j
V = Vx*coords.i + Vy*coords.j
W = Wx*coords.i + Wy*coords.j + Wz*coords.k

I = Ix*coords.i + Iy*coords.j + Iz*coords.k
J = Jx*coords.i + Jy*coords.j + Jz*coords.k
K = Kx*coords.i + Ky*coords.j + Kz*coords.k

tanthetau = sympy.tan(thetau)
tanthetav = sympy.tan(thetav)

#d = tanthetau*U + tanthetav*V + sympy.sqrt(1 - tanthetau*tanthetau - tanthetav*tanthetav)*W
d = (tanthetau*U + tanthetav*V + W).normalize()

dxdz = d.dot(J).simplify()
dydz = d.dot(K).simplify()

subsinit = [(thetau,0), (thetav,0)]

parms = [dxdz, dydz]
parmlabels = ["dxdz", "dydz"]

inparms = [thetau, thetav]
inparmlabels = ["thetau", "thetav"]

results = []
labels = []

for parm,parmlabel in zip(parms,parmlabels):
    for inparm,inparmlabel in zip(inparms,inparmlabels):
        #dparmdinparm = 1*sympy.diff(parm,inparm)
        #d2parmdinparm2 = 1*sympy.diff(dparmdinparm, inparm)
        
        #res = 1*dparmdinparm.subs(subsinit)
        #label = f"d{parmlabel}d{inparmlabel}"
        #results.append(res)
        #labels.append(label)
        
        d2parmdinparm2 = 1*sympy.diff(parm, inparm, 2)
        res = 1*d2parmdinparm2.subs(subsinit)
        label = f"d2{parmlabel}d{inparmlabel}2"
        results.append(res)
        labels.append(label)
        
#substitutions, results2 = sympy.cse(results,symbols = numbered_symbols("xf"))
substitutions, results2 = sympy.cse(results)
#loop through output and translate to C++ code
for sub in substitutions:
  #print(sub[1])
  cxxsub = cxxcode(sub[1],standard='C++11')
  print(f"const double {sub[0]} = {cxxsub};")
for res,label in zip(results2,labels):
  cxxres = cxxcode(res,standard='C++11')
  print(f"const double {label} = {cxxres};")

#print(dxdz)
