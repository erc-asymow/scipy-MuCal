import sympy
from sympy.printing.cxx import cxxcode
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D

#global cartesian coordinate system
coords = CoordSys3D("coords")

qop0 = sympy.Symbol("qop0", nonzero=True)
lam0 = sympy.Symbol("lam0")
phi0 = sympy.Symbol("phi0")
x0 = sympy.Symbol("x0")
y0 = sympy.Symbol("y0")
z0 = sympy.Symbol("z0")

# initial momentum direction components as constants
W0x = sympy.Symbol("W0x")
W0y = sympy.Symbol("W0y")
W0z = sympy.Symbol("W0z")

B = sympy.Symbol("B")
hx = sympy.Symbol("hx")
hy = sympy.Symbol("hy")
hz = sympy.Symbol("hz")
H = hx*coords.i + hy*coords.j + hz*coords.k


s = sympy.Symbol("s")

qop0val = sympy.Symbol("qop0val")
lam0val = sympy.atan(W0z/sympy.sqrt(W0x**2 + W0y**2))
phi0val = sympy.atan2(W0y, W0x)
x0val = sympy.Symbol("x0val")
y0val = sympy.Symbol("y0val")
z0val = sympy.Symbol("z0val")
Bval = sympy.Symbol("Bval")
sval = sympy.Symbol("sval")

subsconst = [(qop0, qop0val), (lam0, lam0val), (phi0, phi0val), (x0, x0val), (y0, y0val), (z0, z0val), (B, Bval), (s, sval)]
subsrev = [(qop0val, qop0), (x0val, x0), (y0val, y0), (z0val, z0), (Bval, B), (sval, s)]

#initial position
M0 = x0*coords.i + y0*coords.j + z0*coords.k
#initial momentum direction
T0 = sympy.cos(lam0)*sympy.cos(phi0)*coords.i + sympy.cos(lam0)*sympy.sin(phi0)*coords.j + sympy.sin(lam0)*coords.k

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

# final momentum direction as constant
W = T.subs(subsconst).simplify()
U = coords.k.cross(W).normalize()
V = W.cross(U)

#tt2 = T[0]**2 + T[1]**2
#tt = sympy.sqrt(tt2)
#lam = sympy.atan(T[2]/tt)
#phi = sympy.atan2(T[1], T[0])
tx = T.dot(coords.i)
ty = T.dot(coords.j)
tz = T.dot(coords.k)
tt = sympy.sqrt(tx**2 + ty**2)

qop = qop0
lam = sympy.atan(tz/tt)
phi = sympy.atan2(ty, tx)
xt = M.dot(U)
yt = M.dot(V)


#qop = qop - qop0
#lam = lam - lam0
#phi = phi - phi0
#xt = xt - xt0
#yt = yt - yt0

parms = [qop, lam, phi, xt, yt]
parmlabels = ["qop", "lam", "phi", "xt", "yt"]

inparms = [qop0, lam0, phi0, x0, y0, z0, B]
inparmlabels = ["qop0", "lam0", "phi0", "x0", "y0", "z0", "B"]

results = []
labels = []

print("dsdinparms")
dsdinparms = []
for inparm in inparms:
    print(inparm)
    dsdinparm = -1*T.dot(sympy.diff(M,inparm))
    dsdinparm = dsdinparm.subs(subsconst).subs(subsrev)
    dsdinparms.append(dsdinparm)

print("final derivatives")
for parm,parmlabel in zip(parms, parmlabels):
    print(parmlabel)
    #parm = parm.simplify()
    dparmds = 1*sympy.diff(parm, s)
    dparmds = dparmds.subs(subsconst).subs(subsrev)
    for inparm,inparmlabel,dsdinparm in zip(inparms,inparmlabels,dsdinparms):
        print(parmlabel, inparmlabel)
        dparmdinparm = 1*sympy.diff(parm, inparm)
        dparmdinparm = dparmdinparm.subs(subsconst).subs(subsrev)
        
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


                
                
        

