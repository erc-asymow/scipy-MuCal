import sympy
from sympy.printing.cxxcode import cxxcode
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D

qop0 = sympy.Symbol("qop0", nonzero=True)
lam0 = sympy.Symbol("lam0")
phi0 = sympy.Symbol("phi0")
xt0 = sympy.Symbol("xt0")
yt0 = sympy.Symbol("yt0")
q = sympy.Symbol("q")
mass = sympy.Symbol("mass")

eloss = sympy.Symbol("eloss")


p0 = q/qop0
#p0 = abs(1/qop0)

dEdx = sympy.Symbol("dEdx")
de = sympy.Symbol("de")


#subsconst = [(qop0, qop0val), (lam0, lam0val), (phi0, phi0val), (xt0, xt0val), (yt0, yt0val), (B, Bval), (s, sval)]
#subsrev = [(qop0val, qop0), (Bval, B), (sval, s), (xi,0)]

subsconst = [(de, 0)]


E0 = sympy.sqrt(p0*p0 + mass*mass)
E = E0 + de*eloss

qop = q/sympy.sqrt(E*E - mass*mass)
lam = lam0
phi = phi0
xt = xt0
yt = yt0

parms = [qop, lam, phi, xt, yt]
parmlabels = ["qop", "lam", "phi", "xt", "yt"]

inparms = [de]
inparmlabels = ["de"]

results = []
labels = []

print("final derivatives")
for parm,parmlabel in zip(parms, parmlabels):
    print(parmlabel)
    #parm = parm.simplify()
    #dparmds = 1*sympy.diff(parm, s)
    #dparmds = dparmds.subs(subsconst).subs(subsrev)
    for inparm,inparmlabel in zip(inparms,inparmlabels):
        print(parmlabel, inparmlabel)
        dparmdinparm = 1*sympy.diff(parm, inparm)
        dparmdinparm = dparmdinparm.subs(subsconst)
        #dparmdinparm = dparmdinparm.subs(subsconst).subs(subsrev)
        
        #res = dparmdinparm + dparmds*dsdinparm
        #res = res.subs(subsconst)
        #res = res.subs(subsrev)
        #res = res.subs([(lam0, lam0val), (phi0, phi0val), (xt0, xt0val), (yt0, yt0val)])
        res = 1*dparmdinparm
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


                
                
        

