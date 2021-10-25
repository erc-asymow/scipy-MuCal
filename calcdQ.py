import sympy
from sympy.printing.cxxcode import cxxcode
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D

qop = sympy.Symbol("qop")
mass = sympy.Symbol("mass")
q = sympy.Symbol("q")
Qms0 = sympy.Symbol("Qms0")
Qmsconst = sympy.Symbol("Qmsconst")
Qion0 = sympy.Symbol("Qion0")
Qionconst = sympy.Symbol("Qionconst")

p = q/qop

p2 = p*p
m2 = mass*mass
e2 = p2 + m2

e = sympy.sqrt(e2)

beta = p/e

beta2 = p2/e2

gamma = e/mass

fms = 1/(p2*beta2)
Qms = fms*Qmsconst

subsms = [(Qmsconst, Qms0/fms)]

dQms = 1*sympy.diff(Qms, qop)
dQms = dQms.subs(subsms)

dQms = dQms.simplify()

print(dQms)


eta = beta*gamma
etasq = eta*eta
eMass = sympy.Symbol("eMass")
massRatio = eMass/mass
F1 = 2*eMass*etasq
F2 = 1 + 2*massRatio*gamma + massRatio*massRatio

Emax = F1/F2

#fion = Emax
#fion = Emax*(1-beta2/2)/beta2*e2/(p2*p2*p2)
#fion = (1.-beta2/2.)*e2/(p2*p2*p2)/beta2
fion = (1./beta2/beta2)*e2/(p2*p2*p2)

Qion = Qionconst*fion

subsion = [(Qionconst, Qion0/fion)]

dQion = 1*sympy.diff(Qion, qop)
dQion = dQion.subs(subsion)
dQion = dQion.simplify()

print(dQion)

labels = ["dQms", "dQion"]
results = [dQms, dQion]


substitutions, results2 = sympy.cse(results)
#loop through output and translate to C++ code
for sub in substitutions:
  #print(sub[1])
  cxxsub = cxxcode(sub[1],standard='C++17')
  print(f"const double {sub[0]} = {cxxsub};")
for res,label in zip(results2,labels):
  cxxres = cxxcode(res,standard='C++17')
  print(f"const Matrix<double, 5, 5> {label} = {cxxres};")
