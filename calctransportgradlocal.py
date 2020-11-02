import sympy
from sympy.printing.cxxcode import cxxcode
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D

coords = CoordSys3D("coords")

#M0x = sympy.Symbol("M0x")
#M0y = sympy.Symbol("M0y")
#M0z = sympy.Symbol("M0z")
#M0const = M0x*coords.i + M0y*coords.j + M0z*coords.k

# initial momentum direction components as constants
W0x = sympy.Symbol("W0x")
W0y = sympy.Symbol("W0y")
W0z = sympy.Symbol("W0z")
W0 = W0x*coords.i + W0y*coords.j + W0z*coords.k
U0 = coords.k.cross(W0).normalize()
V0 = W0.cross(U0)

U0 = coords.i
V0 = coords.j
W0 = coords.k



qop0 = sympy.Symbol("qop0", nonzero=True)
dxdz0 = sympy.Symbol("dxdz0")
dydz0 = sympy.Symbol("dydz0")
xt0 = sympy.Symbol("xt0")
yt0 = sympy.Symbol("yt0")
B = sympy.Symbol("B")
s = sympy.Symbol("s")

qop0_const = sympy.Symbol("qop0_const", nonzero=True)
dxdz0_const = sympy.Symbol("dxdz0_const")
dydz0_const = sympy.Symbol("dydz0_const")
xt0_const = sympy.Symbol("xt0_const")
yt0_const = sympy.Symbol("yt0_const")
B_const = sympy.Symbol("B_const")
s_const = sympy.Symbol("s_const")

#subconst = [(qop0, qop0_const), (dxdz0, dxdz0_const), (dydz0, dydz0_const), (xt0, xt0_const), (yt0, yt0_const), (B, B_const), (s, s_const)]

subconst = [(qop0, qop0_const), (dxdz0, 0), (dydz0, 0), (xt0, xt0_const), (yt0, yt0_const), (B, B_const), (s, s_const)]

subsres = [(qop0_const, qop0), (xt0_const, xt0), (yt0_const, yt0), (B_const, B), (s_const, s)]

#subsres = [(sub[1],sub[0]) for sub in subconst]

subnull = [(dxdz0,0), (dydz0,0)]


M0 = xt0*U0 + yt0*V0
T0 = (dxdz0*U0 + dydz0*V0 + W0).normalize().simplify()

hx = sympy.Symbol("hx")
hy = sympy.Symbol("hy")
hz = sympy.Symbol("hz")
H = hx*coords.i + hy*coords.j + hz*coords.k

print("bfield quantites")

HcrossT0 = H.cross(T0)
N0 = HcrossT0.normalize()
alpha = HcrossT0.magnitude()
gamma = H.dot(T0)
#alpha = sympy.Symbol("alpha")
#gamma = sympy.Symbol("gamma")

#alpha = sympy.sympify("0")
#gamma = sympy.sympify("1")

s = sympy.Symbol("s")

Q = -B*qop0
theta = Q*s
costheta = sympy.cos(theta)
sintheta = sympy.sin(theta)

#print(M0)
#print(H)
#print(T0)
#print(N0)

M = M0 + gamma*(theta-sintheta)/Q*H + sintheta/Q*T0 + alpha*(1-costheta)/Q*N0
#M = M0 + (theta-sintheta)/Q*H + sintheta/Q*T0
print("computing T")
#T = 1*sympy.diff(M,s)
T = 1*sympy.diff(M,s).simplify()

print("basis")

W = T.subs(subconst).simplify()
U = coords.k.cross(W).normalize()
V = W.cross(U)

#dU = (U - U0).dot(U0)
#print(dU)
#dU = dU.simplify()
#print(dU)
##print(U)
##print(V)
#assert(0)

print("parms")

qop = qop0
dxdz = T.dot(U)/T.dot(W)
dydz = T.dot(V)/T.dot(W)
xt = M.dot(U)
yt = M.dot(V)

qop = qop-qop0
dxdz = dxdz - dxdz0
dydz = dydz - dydz0
xt = xt - xt0
yt = yt - yt0

#print("parms expand")

#dxdx = 1*dxdz.expand()
#dydz = 1*dydz.expand()
#xt = 1*xt.expand()
#yt = 1*yt.expand()

#xt = 1*xt.expand()
#print(xt)
#assert(0)
#xt = xt.expand()


parms = [qop, dxdz, dydz, xt, yt]
parmlabels = ["qop", "dxdz", "dydz", "xt", "yt"]

inparms = [qop0, dxdz0, dydz0, xt0, yt0, B]
inparmlabels = ["qop0", "dxdz0", "dydz0", "xt0", "yt0", "B"]



print("computing gradients")


#print(sympy.diff(lam, s))
#dlamdqop = 1*sympy.diff(lam,qop) + 1*sympy.diff(lam, s)*dsdqop[0]

#dlamdqop = 1*sympy.diff(lam,qop)

results = []
labels = []

for i,(parm,parmlabel) in enumerate(zip(parms, parmlabels)):
    for j,(inparm,inparmlabel) in enumerate(zip(inparms,inparmlabels)):
        #if i==j:
            #parm = parm - inparm
        print(parmlabel, inparmlabel)
        #parm = parm.simplify()
        dparmdinparm = 1*sympy.diff(parm, inparm).subs(subnull).subs(subsres).simplify()
        dparmds = 1*sympy.diff(parm, s).subs(subnull).subs(subsres).simplify()
        dsdinparm = -1*T.dot(sympy.diff(M,inparm)).subs(subnull).subs(subsres).simplify()
        
        res = dparmdinparm + dparmds*dsdinparm
        #res = res.subs(subsres)
        #res = res.subs(subnull)
        #res = res.subs([(lam0, lam0val), (phi0, phi0val), (xt0, xt0val), (yt0, yt0val)])
        res = 1*res
        #res = res.expand()
        res = res.simplify()
        
        label = f"d{parmlabel}d{inparmlabel}"
        results.append(res)
        labels.append(label)
 
 

#dxtdxt0alt = U.dot(

#for parm,parmlabel in zip(parms, parmlabels):
    #res = 1*sympy.diff(parm, qop) + 1*sympy.diff(parm, s)*dsdqop[0]
    #label = f"d{parmlabel}dqop"
    #results.append(res)
    #labels.append(label)

#results.append(dMdB)
#labels.append("dMdB")

#results.append(dPdB)
#labels.append("dPdB")

#results.append(dMdqop)
#labels.append("dMdqop")

#results.append(dPdqop)
#labels.append("dPdqop")

#results.append(dlamdqop)
#labels.append("dlamdqop")

#print(dMdB)
#print(dPdB)


#for res, label in zip(results, labels):
  #print(f"const double {label} = {cxxcode(res,standard='C++11')};")

substitutions, results2 = sympy.cse(results)
#loop through output and translate to C++ code
for sub in substitutions:
  #print(sub[1])
  cxxsub = cxxcode(sub[1],standard='C++11')
  print(f"auto const {sub[0]} = {cxxsub};")
for res,label in zip(results2,labels):
  cxxres = cxxcode(res,standard='C++11')
  print(f"auto const {label} = {cxxres};")





                
                
        
