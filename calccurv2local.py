import sympy
from sympy.printing.cxxcode import cxxcode
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D

coords = CoordSys3D("coords")


Ix = sympy.Symbol("Ix")
Iy = sympy.Symbol("Iy")
Iz = sympy.Symbol("Iz")

Jx = sympy.Symbol("Jx")
Jy = sympy.Symbol("Jy")
Jz = sympy.Symbol("Jz")

Kx = sympy.Symbol("Kx")
Ky = sympy.Symbol("Ky")
Kz = sympy.Symbol("Kz")

rx = sympy.Symbol("rx")
ry = sympy.Symbol("ry")
rz = sympy.Symbol("rz")

Ux = sympy.Symbol("Ux")
Uy = sympy.Symbol("Uy")
#Uz = sympy.Symbol("Uz")

Vx = sympy.Symbol("Vx")
Vy = sympy.Symbol("Vy")
#Vz = sympy.Symbol("Vz")

pos0x = sympy.Symbol("pos0x")
pos0y = sympy.Symbol("pos0y")
pos0z = sympy.Symbol("pos0z")

qop = sympy.Symbol("qop")
lam = sympy.Symbol("lam")
phi = sympy.Symbol("phi")
xt = sympy.Symbol("xt")
yt = sympy.Symbol("yt")

s = sympy.Symbol("s")

I = Ix*coords.i + Iy*coords.j + Iz*coords.k
J = Jx*coords.i + Jy*coords.j + Jz*coords.k
K = Kx*coords.i + Ky*coords.j + Kz*coords.k

r = rx*coords.i + ry*coords.j + rz*coords.k

U = Ux*coords.i + Uy*coords.j
V = Vx*coords.i + Vy*coords.j

pos0 = pos0x*coords.i + pos0y*coords.j + pos0z*coords.k

coslam = sympy.cos(lam)
sinlam = sympy.sin(lam)
cosphi = sympy.cos(phi)
sinphi = sympy.sin(phi)

x = pos0 + xt*U + yt*V + coslam*cosphi*s*coords.i + coslam*sinphi*s*coords.j + sinlam*s*coords.k

results = []
labels = []

shat = sympy.solve(I.dot(x-r), s)[0]
shat = shat.simplify()

results.append(shat)
labels.append("shat")

#print(shat)

x = x.subs(s,shat)
x = x.simplify()

v = x.dot(J)
w = x.dot(K)

parms = [v, w]
parmlabels = ["v", "w"]

#inparms = [lam, phi, xt, yt]
#inparmlabels = ["lam", "phi", "xt", "yt"]

inparms = [qop, lam, phi, xt, yt]
inparmlabels = ["qop", "lam", "phi", "xt", "yt"]



nparmsin = len(inparms)


for parm,parmlabel in zip(parms, parmlabels):
    #print(parmlabel)
    for i in range(nparmsin):
        inparmi = inparms[i]
        inparmlabeli = inparmlabels[i]
        #print(parmlabel, inparmlabel)
        dparmdinparmi = 1*sympy.diff(parm, inparmi)
        label = f"d{parmlabel}d{inparmlabeli}"
        
        results.append(dparmdinparmi)
        labels.append(label)
        
        #results.append(dparmdinparm)
        #labels.append(label)
        #print(dparmdinparm)
        for j in range(i, nparmsin):
            inparmj = inparms[j]
            inparmlabelj = inparmlabels[j]
            d2parmdinparmidinparmj = 1*sympy.diff(dparmdinparmi, inparmj)
            label = f"d2{parmlabel}d{inparmlabeli}d{inparmlabelj}"
            results.append(d2parmdinparmidinparmj)
            labels.append(label)
            #print(f"{label} = {d2parmdinparmidinparmj}")
        

#for res, label in zip(results, labels):
  #print(f"const double {label} = {cxxcode(res,standard='C++11')};")

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
  
for parm,parmlabel in zip(parms, parmlabels):
    reslabel = f"d2{parmlabel}dx2"
    print(f"Matrix<double, {len(inparms)}, {len(inparms)}> {reslabel};")
    for i in range(nparmsin):
        inparmi = inparms[i]
        inparmlabeli = inparmlabels[i]
        for j in range(nparmsin):
            inparmj = inparms[j]
            inparmlabelj = inparmlabels[j]
            if j>=i:
                label = f"d2{parmlabel}d{inparmlabeli}d{inparmlabelj}"
            else:
                label = f"d2{parmlabel}d{inparmlabelj}d{inparmlabeli}"
            print(f"{reslabel}({i}, {j}) = {label};")

  
for parm,parmlabel in zip(parms, parmlabels):
    reslabel = f"d{parmlabel}dx"
    print(f"Matrix<double, {len(inparms)}, 1> {reslabel};")
    for i in range(nparmsin):
        inparmi = inparms[i]
        inparmlabeli = inparmlabels[i]
        label = f"d{parmlabel}d{inparmlabeli}"
        print(f"{reslabel}[{i}] = {label};")
