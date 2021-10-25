import sympy
from sympy.printing.cxxcode import cxxcode
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D

#global cartesian coordinate system
coords = CoordSys3D("coords")

M0x = sympy.Symbol("M0x")
M0y = sympy.Symbol("M0y")
M0z = sympy.Symbol("M0z")
M0const = M0x*coords.i + M0y*coords.j + M0z*coords.k

qop0 = sympy.Symbol("qop0", nonzero=True)
lam0 = sympy.Symbol("lam0")
phi0 = sympy.Symbol("phi0")
xt0 = sympy.Symbol("xt0")
yt0 = sympy.Symbol("yt0")
q = sympy.Symbol("q")
mass = sympy.Symbol("mass")

p0 = q/qop0
#p0 = abs(1/qop0)

E0 = sympy.Symbol("E0")
dEdx = sympy.Symbol("dEdx")
xi = sympy.Symbol("xi")

# initial momentum direction components as constants
W0x = sympy.Symbol("W0x")
W0y = sympy.Symbol("W0y")
W0z = sympy.Symbol("W0z")
W0 = W0x*coords.i + W0y*coords.j + W0z*coords.k
U0 = coords.k.cross(W0).normalize()
V0 = W0.cross(U0)

#qop0val = sympy.Symbol("qop0val")
#lam0val = sympy.atan(W0z/sympy.sqrt(W0x**2 + W0y**2))
#phi0val = sympy.atan2(W0y, W0x)
#xt0val = M0const.dot(U0)
#yt0val = M0const.dot(V0)
#Bval = sympy.Symbol("Bval")
#sval = sympy.Symbol("sval")

qop0val = sympy.Symbol("qop0val")
lam0val = sympy.atan(W0z/sympy.sqrt(W0x**2 + W0y**2))
phi0val = sympy.atan2(W0y, W0x)
xt0val = M0const.dot(U0)
yt0val = M0const.dot(V0)
Bzval = sympy.Symbol("Bzval")
sval = sympy.Symbol("sval")


#B = sympy.Symbol("B")
#hx = sympy.Symbol("hx")
#hy = sympy.Symbol("hy")
#hz = sympy.Symbol("hz")
#H = hx*coords.i + hy*coords.j + hz*coords.k

Bx = sympy.Symbol("Bx")
By = sympy.Symbol("By")
Bz = sympy.Symbol("Bz")
Bv0 = Bx*coords.i + By*coords.j + Bz*coords.k

dBdxt0x = sympy.Symbol("dBdxt0x")
dBdxt0y = sympy.Symbol("dBdxt0y")
dBdxt0z = sympy.Symbol("dBdxt0z")
dBdxt0 = dBdxt0x*coords.i + dBdxt0y*coords.j + dBdxt0z*coords.k

dBdyt0x = sympy.Symbol("dBdyt0x")
dBdyt0y = sympy.Symbol("dBdyt0y")
dBdyt0z = sympy.Symbol("dBdyt0z")
dBdyt0 = dBdyt0x*coords.i + dBdyt0y*coords.j + dBdyt0z*coords.k

Bv = Bv0 + dBdxt0*(xt0 - xt0val) + dBdyt0*(yt0 - yt0val)


H = Bv.normalize()
B = Bv.magnitude()

s = sympy.Symbol("s")



#subsconst = [(qop0, qop0val), (lam0, lam0val), (phi0, phi0val), (xt0, xt0val), (yt0, yt0val), (B, Bval), (s, sval)]
#subsrev = [(qop0val, qop0), (Bval, B), (sval, s), (xi,0)]

subsconst = [(qop0, qop0val), (lam0, lam0val), (phi0, phi0val), (xt0, xt0val), (yt0, yt0val), (Bz, Bzval), (s, sval), (xi, 0)]
subsrev = [(qop0val, qop0), (Bzval, Bz), (sval, s), (xi,0)]

#initial position
M0 = xt0*U0 + yt0*V0
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
#M = M.simplify()
print("T")
#T = 1*sympy.diff(M,s).simplify()
T = 1*sympy.diff(M,s)

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

#qop = qop0
#qop = qop0 - E0*qop0val*qop0val*qop0val*(1+xi)*dEdx*s
qop = q/sympy.sqrt(p0*p0 + (1+xi)*(1+xi)*dEdx*dEdx*s*s + 2*sympy.sqrt(p0*p0+mass*mass)*(1+xi)*dEdx*s)
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

inparms = [qop0, lam0, phi0, xt0, yt0, Bz, xi]
inparmlabels = ["qop0", "lam0", "phi0", "xt0", "yt0", "Bz", "xi"]

print("sanity check")
Wx = W.dot(coords.i)
Wy = W.dot(coords.j)
Wz = W.dot(coords.k)
for inparm in inparms:
    dWx = 1*sympy.diff(Wx, inparm)
    dWy = 1*sympy.diff(Wy, inparm)
    dWz = 1*sympy.diff(Wz, inparm)
    print(dWx,dWy,dWz)

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


print(f"Eigen::Matrix<double, {len(parms)}, {len(inparms)}> res;")
for i,parmlabel in enumerate(parmlabels):
    for j,inparmlabel in enumerate(inparmlabels):
        print(f"res({i},{j}) = d{parmlabel}d{inparmlabel};")


                
                
        

