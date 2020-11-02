import sympy
#from sympy.printing import print_ccode
from sympy.printing.cxxcode import cxxcode

#q = sympy.symbols("q")
qop = sympy.symbols("qop",nonzero=True)
dxdz = sympy.symbols("dxdz")
dydz = sympy.symbols("dydz")
xi = sympy.symbols("xi")
radLen = sympy.symbols("radLen")
q = sympy.sign(qop)
signpz = sympy.symbols("signpz")
logfact = sympy.symbols("logfact") #0.038
#q = sympy.symbols("q")

m2 = sympy.symbols("m2")

emass = sympy.symbols("emass")
poti = sympy.symbols("poti")
eplasma = sympy.symbols("eplasma")
amscon = sympy.symbols("amscon")

p = sympy.Abs(1/qop)
p2 = 1/qop**2
abspz = 1/(qop/q * sympy.sqrt(1+ dxdz**2 + dydz**2))

##p2 = sympy.symbols("p2")
#abspz = sympy.symbols("abspz")


delta0 = 2*sympy.log(eplasma/poti) - 1
xf = p/abspz

radLenmod = radLen*xf

im2 = 1/m2
e2 = p2 + m2
e = sympy.sqrt(e2)
beta2 = p2/e2
eta2 = p2*im2
ratio2 = emass**2*im2
emax = 2*emass*eta2/(1+2*emass*e*im2+ratio2)

ximod = xi*xf/beta2

dEdx2 = ximod * emax * (1 - beta2/2)
sigp2 = dEdx2 / (beta2 * p2 * p2)
elos = sigp2

fact = 1 + logfact*sympy.log(radLenmod)
fact = fact**2

a = fact/(beta2*p2)
sigt2 = amscon*radLenmod*a

dz = abspz*signpz/p
dx = abspz*signpz*dxdz/p
dy = abspz*signpz*dydz/p

isl2 = 1./(dx**2 + dy**2)
cl2 = dz**2
cf2 = dx**2*isl2
sf2 = dy**2*isl2
den = 1./cl2**2

msxx = den*sigt2*(sf2*cl2 + cf2)
msxy = den*sigt2*dx*dy
msyy = den*sigt2*(cf2*cl2 + sf2)

covelems = [elos, msxx, msxy, msyy]
covelemlabels = ["elos", "msxx", "msxy", "msyy"]

localparms = [qop, dxdz, dydz, xi, radLen]
localparmlabels = ["qop", "dxdz", "dydz", "xi", "radLen"]

resultspre = []
labels = []

#for covelem, covelemlabel in zip(covelems, covelemlabels):
    #resultspre.append(covelem)
    #labels.append(covelemlabel)

for covelem, covelemlabel in zip(covelems, covelemlabels):
    for localparm, localparmlabel in zip(localparms, localparmlabels):
        res = sympy.diff(covelem, localparm)
        label = f"d{covelemlabel}d{localparmlabel}"
        resultspre.append(res)
        labels.append(label)


for covelem, covelemlabel in zip(covelems, covelemlabels):
    res = sympy.diff(covelem, qop, 2)
    label = f"d2{covelemlabel}dqop2"
    resultspre.append(res)
    labels.append(label)



#dmsxxdqop = sympy.diff(msxx, qop)
#dmsxxddxdz = sympy.diff(msxx, dxdz)
#dmsxxddydz = sympy.diff(msxx, dydz)

#dmsxxdqop = sympy.diff(msxx, qop)
#dmsxxddxdz = sympy.diff(msxx, dxdz)
#dmsxxddydz = sympy.diff(msxx, dydz)

#resultspre.append(dmsxxdqop)


for res in resultspre:
  print(res)

#collect gradients and simplify with common term substitution
substitutions, results = sympy.cse(resultspre)
#loop through output and translate to C++ code
for sub in substitutions:
  #print(sub[1])
  print(f"const double {sub[0]} = {cxxcode(sub[1],standard='C++11')};")
for res,label in zip(results, labels):
  print(f"const double {label} = {cxxcode(res,standard='C++11')};")

