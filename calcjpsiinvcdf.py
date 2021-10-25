import sympy
from sympy.printing import cxxcode


pt_0 = sympy.Symbol("pt_0")
pt_1 = sympy.Symbol("pt_1")
pt_2 = sympy.Symbol("pt_2")
pt = sympy.Symbol("pt")
gamma = sympy.Symbol("gamma", positive=True)
A = sympy.Symbol("A")
B = sympy.Symbol("B")
y = sympy.Symbol("y")

integral_0 = sympy.integrate(B*pt_1**-gamma, (pt, pt_0, pt_2))
integral_1 = sympy.integrate(B*pt_1**-gamma, (pt, pt_0, pt_1)) + sympy.integrate(B*pt**-gamma, (pt, pt_1, pt_2))

invcdf_0 = sympy.solve(integral_0 - y, pt_2)
invcdf_0 = sympy.solve(integral_1 - y, pt_2)

print(invcdf_0)
