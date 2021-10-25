import sympy
from sympy.printing.cxx import cxxcode
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D


mu = sympy.Symbol("mu")
cosalpha = sympy.Symbol("cosalpha")
p0 = sympy.Symbol("p0")
p1 = sympy.Symbol("p1")

pt0 = sympy.Symbol("pt0")
pt1 = sympy.Symbol("pt1")

q0 = sympy.Symbol("q0")
q1 = sympy.Symbol("q1")


A0 = sympy.Symbol("A0")
e0 = sympy.Symbol("e0")
M0 = sympy.Symbol("M0")

A1 = sympy.Symbol("A1")
e1 = sympy.Symbol("e1")
M1 = sympy.Symbol("M1")

m0 = sympy.Symbol("m0")

sigm = sympy.Symbol("sigm")

scale0 = 1. + A0 - e0/pt0 + q0*M0*pt0
scale1 = 1. + A1 - e1/pt1 + q1*M1*pt1

#m = sympy.sqrt(2*mu*mu + 2*sympy.sqrt(scale0*scale0*p0*p0 + mu*mu)*sympy.sqrt(scale1*scale1*p1*p1+mu*mu) - 2*scale0*scale1*p0*p1*cosalpha)

msq = (2*mu*mu + 2*sympy.sqrt(p0*p0/scale0/scale0 + mu*mu)*sympy.sqrt(p1*p1/scale1/scale1 + mu*mu) - 2*p0*p1*cosalpha/scale0/scale1)

invmsq = 1./msq

m = sympy.sqrt(msq)


#siginvmsq = (2/m**3)*sigm


dm = invmsq - 1/m0**2
chisq = dm*dm/sigm/sigm
#chisq = dm*dm

subsconst = [(A0, 0), (e0, 0), (M0, 0), (A1, 0), (e1, 0), (M1, 0)]



parms = [chisq]
parmlabels = ["chisq"]

inparms = [A0, e0, M0, A1, e1, M1]
inparmlabels = ["A0", "e0", "M0", "A1", "e1", "M1"]


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
        res = res.subs(subsconst)
        #res = res.simplify()
        label = f"d{parmlabel}d{inparmlabel}"
        
        results.append(res)
        labels.append(label)
        
        for jinparm,jinparmlabel in zip(inparms,inparmlabels):
            res2 = 1*sympy.diff(dparmdinparm,jinparm)
            res2 = res2.subs(subsconst)
            label = f"d2{parmlabel}d{inparmlabel}d{jinparmlabel}"
            
            results.append(res2)
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


print(f"Eigen::Matrix<double, {len(parms)}, {len(inparms)}> g;")
for i,parmlabel in enumerate(parmlabels):
    for j,inparmlabel in enumerate(inparmlabels):
        print(f"g({i},{j}) = d{parmlabel}d{inparmlabel};")

print(f"Eigen::Matrix<double, {len(inparms)}, {len(inparms)}> h;")
for i,parmlabel in enumerate(parmlabels):
    for j,inparmlabel in enumerate(inparmlabels):
        for k,kinparmlabel in enumerate(inparmlabels):
            print(f"h({j},{k}) = d2{parmlabel}d{inparmlabel}d{kinparmlabel};")

