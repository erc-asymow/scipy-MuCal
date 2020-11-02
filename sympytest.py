import sympy
from sympy.abc import i, k, m, n, x
from sympy.printing.cxxcode import cxxcode

Q = sympy.MatrixSymbol("Q", 3,3)
Qinv = sympy.MatrixSymbol("Qinv", 3,3)
dQ = sympy.MatrixSymbol("dQ", 3,3)
epsilon = sympy.Symbol("epsilon")

#Qinv = Q.inverse()

#t = (Q+epsilon*dQ).inverse()

t = Q - epsilon*Qinv*dQ*Qinv

gt = 1*sympy.diff(t,epsilon)

print(t)
print(gt)



assert(0)

#N=5

#dy0s = []
#for i in range(N):
  #dy0 = sympy.MatrixSymbol(f"dy0_{i}",2,1)
  #dy0s.append(dy0)
  
#dy0sum = sympy.sum(dy0s, N)
  
##dy0sum = dy0s[0]
##for i in range(1,N):
  ##dy0sum += dy0s[i]
  
#print(dy0sum)

test = sympy.Sum(k, (k, 1, m))

print(test)

res = cxxcode(test,standard='C++11')

print(res)

