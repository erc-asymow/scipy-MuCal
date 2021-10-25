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

Ix1 = sympy.Symbol("Ix1")
Iy1 = sympy.Symbol("Iy1")
Iz1 = sympy.Symbol("Iz1")

Jx1 = sympy.Symbol("Jx1")
Jy1 = sympy.Symbol("Jy1")
Jz1 = sympy.Symbol("Jz1")

Kx1 = sympy.Symbol("Kx1")
Ky1 = sympy.Symbol("Ky1")
Kz1 = sympy.Symbol("Kz1")

rx1 = sympy.Symbol("rx1")
ry1 = sympy.Symbol("ry1")
rz1 = sympy.Symbol("rz1")

I1 = Ix1*coords.i + Iy1*coords.j + Iz1*coords.k
J1 = Jx1*coords.i + Jy1*coords.j + Jz1*coords.k
K1 = Kx1*coords.i + Ky1*coords.j + Kz1*coords.k

r1 = rx1*coords.i + ry1*coords.j + rz1*coords.k

M0x = sympy.Symbol("M0x")
M0y = sympy.Symbol("M0y")
M0z = sympy.Symbol("M0z")
M0const = M0x*coords.i + M0y*coords.j + M0z*coords.k

qop0 = sympy.Symbol("qop0", nonzero=True)
lam0 = sympy.Symbol("lam0")
phi0 = sympy.Symbol("phi0")
xt0 = sympy.Symbol("xt0")
yt0 = sympy.Symbol("yt0")


## initial momentum direction components as constants
#W0x = sympy.Symbol("W0x")
#W0y = sympy.Symbol("W0y")
#W0z = sympy.Symbol("W0z")
#W0 = W0x*coords.i + W0y*coords.j + W0z*coords.k
#U0 = coords.k.cross(W0).normalize()
#V0 = W0.cross(U0)

B = sympy.Symbol("B")
hx = sympy.Symbol("hx")
hy = sympy.Symbol("hy")
hz = sympy.Symbol("hz")
H = hx*coords.i + hy*coords.j + hz*coords.k


s = sympy.Symbol("s")

#qop0val = sympy.Symbol("qop0val")
#lam0val = sympy.atan(W0z/sympy.sqrt(W0x**2 + W0y**2))
#phi0val = sympy.atan2(W0y, W0x)
#xt0val = M0const.dot(U0)
#yt0val = M0const.dot(V0)
#Bval = sympy.Symbol("Bval")
#sval = sympy.Symbol("sval")

#subsconst = [(qop0, qop0val), (lam0, lam0val), (phi0, phi0val), (xt0, xt0val), (yt0, yt0val), (B, Bval), (s, sval)]
#subsrev = [(qop0val, qop0), (Bval, B), (sval, s)]

#initial position
M0 = xt0*U0 + yt0*V0
#M0 = r0 + x0*J0 + y0*K0
#initial momentum direction
T0 = sympy.cos(lam0)*sympy.cos(phi0)*coords.i + sympy.cos(lam0)*sympy.sin(phi0)*coords.j + sympy.sin(lam0)*coords.k
#T0 = localpzsign*(I0 + dxdz0*J0 + dydz0*K0)/sympy.sqrt(1 + dxdz0*dxdz0 + dydz0*dydz0)

HcrossT0 = H.cross(T0)
N0 = HcrossT0.normalize()
alpha = HcrossT0.magnitude()
gamma = H.dot(T0)

Q = -B*qop0
theta = Q*s
costheta = sympy.cos(theta)
sintheta = sympy.sin(theta)

print("M")
M = M0 + gamma*(theta-sintheta)/Q*H + sintheta/Q*T0 + alpha*(1-costheta)/Q*N0
M = M.simplify()
print("T")
T = 1*sympy.diff(M,s).simplify()

TdotI1 = T.dot(I1)

qop = qop0
dxdz = T.dot(J1)/TdotI1
dydz = T.dot(K1)/TdotI1
x = (M-r1).dot(J1)
y = (M-r1).dot(K1)

#dxdz = dxdz.simplify()
#dydz = dydz.simplify()
#x = x.simplify()
#y = y.simplify()

parms = [qop, dxdz, dydz, x, y]
parmlabels = ["qop", "dxdz", "dydz", "x", "y"]

inparms = [qop0, lam0, phi0, xt0, yt0, B]
inparmlabels = ["qop0", "lam0", "phi0", "xt0", "yt0", "B"]

results = []
labels = []

print("dsdinparms")
dsdinparms = []
for inparm in inparms:
    print(inparm)
    dsdinparm = -1*I1.dot(sympy.diff(M,inparm))/TdotI1
    #dsdinparm = -1*T.dot(sympy.diff(M,inparm))
    #dsdinparm = dsdinparm.subs(subsconst).subs(subsrev)
    dsdinparms.append(dsdinparm)

print("final derivatives")
for parm,parmlabel in zip(parms, parmlabels):
    print(parmlabel)
    #parm = parm.simplify()
    dparmds = 1*sympy.diff(parm, s)
    #dparmds = dparmds.subs(subsconst).subs(subsrev)
    for inparm,inparmlabel,dsdinparm in zip(inparms,inparmlabels,dsdinparms):
        print(parmlabel, inparmlabel)
        dparmdinparm = 1*sympy.diff(parm, inparm)
        #dparmdinparm = dparmdinparm.subs(subsconst).subs(subsrev)
        
        res = dparmdinparm + dparmds*dsdinparm
        #res = res.subs(subsconst)
        #res = res.subs(subsrev)
        #res = res.subs([(lam0, lam0val), (phi0, phi0val), (xt0, xt0val), (yt0, yt0val)])
        res = 1*res
        #res = res.simplify()
        
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


                
                
        

