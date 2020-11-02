import sympy
from sympy.printing.cxxcode import cxxcode
from sympy.utilities.iterables import numbered_symbols

#X = sympy.MatrixSymbol("X",3,1)
#P = sympy.MatrixSymbol("P",3,1)
x = sympy.Symbol("x")
y = sympy.Symbol("y")
z = sympy.Symbol("z")
px = sympy.Symbol("px")
py = sympy.Symbol("py")
pz = sympy.Symbol("pz")
q = sympy.Symbol("q", nonzero=True)

pt2 = px**2 + py**2
p = sympy.sqrt(pt2 + pz**2)
pt = sympy.sqrt(pt2)

qop = q/p
lam = sympy.atan(pz/pt)
phi = sympy.atan2(py, px)
xt = (-py*x + px*y)/pt
yt = (-x*px*pz - y*pz*py + z*pt2)/(p*pt)

parms = [qop, lam, phi, xt, yt]
parmlabels = ["qop", "lam", "phi", "xt", "yt"]

#inparms = [X, P]
#inparmlabels = ["X", "P"]

inparms = [x,y,z, px,py,pz]
inparmlabels = ["x", "y", "z", "px", "py", "pz"]

results = []
labels = []

for parm,parmlabel in zip(parms, parmlabels):
    for inparm, inparmlabel in zip(inparms, inparmlabels):
        res = sympy.diff(parm, inparm)
        label = f"d{parmlabel}d{inparmlabel}"
        results.append(res)
        labels.append(label)
        
substitutions, results2 = sympy.cse(results)
#loop through output and translate to C++ code
for sub in substitutions:
  #print(sub[1])
  cxxsub = cxxcode(sub[1],standard='C++11').replace(".T",".transpose()")
  print(f"const double {sub[0]} = {cxxsub};")
for res,label in zip(results2,labels):
  cxxres = cxxcode(res,standard='C++11').replace(".T",".transpose()")
  print(f"const double {label} = {cxxres};")
  
  
for iparm, parmlabel in enumerate(parmlabels):
    for iinparm, inparmlabel in enumerate(inparmlabels):
        label = f"d{parmlabel}d{inparmlabel}"
        assign = f"J({iparm}, {iinparm}) = {label};"
        print(assign)


