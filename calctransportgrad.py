import sympy
from sympy.printing.cxxcode import cxxcode
from sympy.utilities.iterables import numbered_symbols


M0 = sympy.MatrixSymbol("M0",3,1)
h = sympy.MatrixSymbol("h",3,1)
T0 = sympy.MatrixSymbol("T0",3,1)
N0 = sympy.MatrixSymbol("N0",3,1)
U = sympy.MatrixSymbol("U",3,1)
V = sympy.MatrixSymbol("V",3,1)

gamma = sympy.Symbol("gamma")
alpha = sympy.Symbol("alpha")
s = sympy.Symbol("s")
qop = sympy.Symbol("qop", nonzero=True)
#qop = sympy.Symbol("qop")
#q = sympy.sign(qop)
p = sympy.Abs(1/qop)

#q = sympy.Symbol("q")
#p = sympy.Symbol("p")

magb = sympy.Symbol("magb")
#magb = sympy.Symbol("magb", nonzero=True)

#Q = -magb*q/p
Q = -magb*qop
theta = Q*s
costheta = sympy.cos(theta)
sintheta = sympy.sin(theta)

M = M0 + gamma*(theta-sintheta)/Q*h + sintheta/Q*T0 + alpha*(1-costheta)/Q*N0
#T = gamma*(1-costheta)*h + costheta*T0 + alpha*sintheta*N0
T = 1*sympy.diff(M,s).simplify()
P = p*T

pt2 = P[0]**2 + P[1]**2
pt = sympy.sqrt(pt2)
lam = sympy.atan(P[2]/pt)
phi = sympy.atan2(P[1], P[0])
xt = (M.T*U)[0]
yt = (M.T*V)[0]
#xt = (-P[1]*M[0] + P[0]*M[1])/pt
#yt = (-M[0]*P[0]*P[2] - M[1]*P[2]*P[1] + M[2]*pt2)/(p*pt)

#print(T)
#print(Talt)

#assert(0)

partialdMdB = 1*sympy.diff(M,magb).simplify()
partialdPdB = 1*sympy.diff(P,magb).simplify()

partialdMdqop = 1*sympy.diff(M,qop).simplify()
partialdPdqop = 1*sympy.diff(P,qop).simplify()

dsdB = -T.T*partialdMdB
dsdqop = -T.T*partialdMdqop

#dMdB = partialdMdB + sympy.diff(M,magb)*dsdB
#dPdB = partialdMdB + sympy.diff(P,magb)*dsdB

dMdB = partialdMdB + sympy.diff(M,s).simplify()*dsdB
dPdB = partialdPdB + sympy.diff(P,s).simplify()*dsdB

dMdqop = partialdMdqop + sympy.diff(M,s).simplify()*dsdqop
dPdqop = partialdPdqop + sympy.diff(P,s).simplify()*dsdqop

parms = [lam, phi, xt, yt]
parmlabels = ["lam", "phi", "xt", "yt"]



#print(sympy.diff(lam, s))
#dlamdqop = 1*sympy.diff(lam,qop) + 1*sympy.diff(lam, s)*dsdqop[0]

#dlamdqop = 1*sympy.diff(lam,qop)

results = []
labels = []

for parm,parmlabel in zip(parms, parmlabels):
    res = 1*sympy.diff(parm, magb) + 1*sympy.diff(parm, s)*dsdB[0]
    label = f"d{parmlabel}dB"
    results.append(res)
    labels.append(label)
    
for parm,parmlabel in zip(parms, parmlabels):
    res = 1*sympy.diff(parm, qop) + 1*sympy.diff(parm, s)*dsdqop[0]
    label = f"d{parmlabel}dqop"
    results.append(res)
    labels.append(label)

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

print(dMdB)
print(dPdB)


for i,res in enumerate(results):
  print(f"auto const& res_{i} = {cxxcode(res,standard='C++11')};")

substitutions, results2 = sympy.cse(results,symbols = numbered_symbols("xf"))
#loop through output and translate to C++ code
for sub in substitutions:
  #print(sub[1])
  cxxsub = cxxcode(sub[1],standard='C++11').replace(".T",".transpose()")
  print(f"auto const {sub[0]} = {cxxsub};")
for res,label in zip(results2,labels):
  cxxres = cxxcode(res,standard='C++11').replace(".T",".transpose()")
  print(f"auto const {label} = {cxxres};")





                
                
        
