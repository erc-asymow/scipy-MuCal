import sympy
from sympy.matrices import Matrix, ones, zeros, hadamard_product, BlockMatrix, block_collapse
from sympy import init_printing
init_printing() 

N = 3

sigma = sympy.symbols("sigma")
a = sympy.symbols("a")
L = sympy.symbols("L")
k = sympy.symbols("k")

##test = (3 + 4*k**2)/(1+5000*k**2)/(1+k**2)/(1+2*k**2)
#test = (3 + 4*k**2)/(1+2*k**2)

#test1 = sympy.apart(test)/k
#test2 = sympy.apart(test/k)

#print(test1)
#print(test2)

#assert(0)

def xinit(i,j):
    return i*L/(N-1)

#def xinit(i,j):
    #return (i+1)*L/N
    
#def xinit(i,j):
    #return L - i*L/N

x = Matrix(N,1,xinit)
    
A0 = ones(N,1)
A1 = x
A2 = x.multiply_elementwise(x)

#print(A0)
#print(A1)
#print(A2)

A = BlockMatrix([A0,A1,A2])
A = A.as_explicit()

M = sympy.symbols("M")
dA = sympy.symbols("dA")
dA2 = sympy.symbols("dA2")
epsilon = sympy.symbols("epsilon")
k_g = sympy.symbols("k_g")
q = sympy.symbols("q")
R = sympy.symbols("R")

#M1 = sympy.symbols("M1")
#M2 = sympy.symbols("M2")

xbs = L/100

alpha = sympy.symbols("alpha")
phi = sympy.symbols("phi")

def ytrueinit(i,j):
    
    #res = k*x[i,0]*(x[i,0]-x[N-1,0])
    
    #res = -k*xbs**2 + k*x[i,0]**2
    res =  phi*x[i,0] + k*x[i,0]**2
    
    #res += dA2*k*x[i,0]**2
    
    #if i==1:
        #res += dA*k
    
    if i==1:
        #res += dA*q*k + M - epsilon*q*k**2
        res += dA*q*k + M - epsilon*q*k**2 + q*R
        pass
    
    #if i>2:
        #res += dA*k + M - epsilon*k**2
        #res += dA*k
        #res += epsilon*k**2*(x[i,0]-x[2,0])**2
        #res += M
    #else:
        #res += -dA*k/(N-1)
    #res += -M/N
        #res += epsilon*k**2 + dA*k
        #res += epsilon*k**2
    #if i==4:
        #res += dA2*k
        #res += epsilon*k**2/(1.+epsilon*k)
        
    #if i==4:
        #res += M2
        
    return res
    
#def dytrueinit(i,j):
    
    

ymeas = Matrix(N,1,ytrueinit)

def Vinit(i,j):
    res = 0
    if i==j:
        res += sigma**2
        
    for l in range(min(i,j)):
        res += (x[i,0] - x[l,0])*(x[j,0] - x[l,0])*a**2*k**2/(N-1)
        
    return res
        
V = Matrix(N,N,Vinit)

#V = V.subs(a,0)
#V = V.subs(k,sympy.sympify("1/20"))

#print(V)

#assert(0)


#A = A.subs(L,1)
#V = V.subs(L,1)
#V = V.subs(sigma,1)
#V = V.subs(a,1)

A = A.simplify()
V = sympy.simplify(V)

#Ainv = A.pinv()
#Ainv = sympy.simplify(Ainv)

#ATinv = A.transpose().pinv()
#ATinv = sympy.simplify(ATinv)

#varalt = Ainv*V*Ainv.transpose()

#print(varalt)


print(A)
print(V)

#det = V.cholesky(hermitian=False)
##det = V.det()
#det = sympy.factor(det)
#print("determinant")
#print(det)

#Vinv = V.inv()
#print("Vinv")
#print(Vinv)


print("invert")
Vinv = V.inv(method = "ADJ")
#Vinv = V.inv()
print("simplify")
Vinv = Vinv.cancel()

#print(Vinv)
#assert(0)



#print(Vinv)

#print("multiply")
#B = Vinv*A

##print(B)

#scale = 2*B[N-1,2]
#scale = sympy.simplify(scale)
#print(scale)
#assert(0)


varinv = A.transpose()*Vinv*A
print("simplify")
varinv = varinv.cancel()


det = sympy.factor(varinv.det())
print("determinant")
print(det)

print("invert")
var = varinv.inv(method="ADJ")

#var = varalt


#xt = var*A.transpose()*Vinv*ymeas

xtpre = var*A.transpose()*Vinv
xtpre = sympy.simplify(xtpre)

print("xtpre k weights:")
print(xtpre[2,:])

xt = xtpre*ymeas

scale = xt
#scale = xt[2]/k - 1
#scale = xt[1]/k
#scale = xt[0]
scale = sympy.simplify(scale)


yres = ymeas - A*scale
yres = sympy.simplify(yres)
print("yres:")
print(yres)

#bsres = scale[0] + scale[1]*xbs + scale[2]*xbs**2
#bsres = sympy.simplify(bsres)
#print("bsres")
#print(bsres)

dypre = sympy.eye(N) - A*xtpre
#print(dypre)
dypre = sympy.simplify(dypre)
#print(dypre)

#singular = dypre.transpose().nullspace()
#singular = dypre.transpose()
#singular = dypre.singular_values()

#print("singular")
#print(singular)

dy = dypre.pinv_solve(yres,arbitrary_matrix=zeros(N,1))
#dy = dypre.pinv()*yres
#dy = yres*dypre.pinv()
dy = sympy.simplify(dy)
print("dy")
print(dy)


#dyavg = 0
#dyxavg = 0

xbar = 0
x2bar = 0
x3bar = 0

for i in range(N):
    #dyavg += dy[i,0]/5
    #dyxavg += dy[i,0]*x[i,0]/5
    xbar += x[i,0]/N
    x2bar += x[i,0]**2/N
    x3bar += x[i,0]**3/N

exp0 = 0
exp1 = 0
exp2 = 0
for i in range(N):
    exp0 += dy[i,0]/N
    exp1 += dy[i,0]*x[i,0]/N
    exp2 += dy[i,0]/x[i,0]/N
    #exp1 += (dy[i,0]*x[i,0]-dy[i,0])/N

exp0 = exp0.simplify()
exp1 = exp1.simplify()

print("exp0")
print(exp0)
print("exp1")
print(exp1)
print("exp2")
print(exp2)

#adjust residuals subject to constraints
#1) zero average dxy wrt beamspot
#2) no global translation
#3) no global rotation
#a0 = -dy[0]
#a2 = (dy[0]*(xbar**2-x2bar) - exp1*xbar)/(x3bar*xbar-x2bar**2)
#a2 = (dy[0]*xbar**2 -dy[0]*x2bar - exp1*xbar + exp0*x2bar)/(x3bar*xbar - x2bar**2)
#a1 = (dy[0] - exp0 - a2*x2bar)/xbar

#a0 = -dy[0]
#a1 = (-exp0*x3bar - a0*x3bar + exp1*x2bar + a0*xbar*x2bar)/(xbar*x3bar - x2bar**2)
#a2 = (-exp1 - a0*xbar - a1*x2bar)/x3bar
 
#a0 = -dy[0]
##a1 = (exp1*x2bar + a0*xbar*x2bar - exp0*x3bar - a0*x3bar)/(xbar*x2bar - x2bar**2)
#a2 = (-exp0 - a0 - a1*xbar)/(x2bar)
 
 
#a0 = symby.symbols("a0")

#a0 = -dy[0]
#a0 = a0.subs(k,0)
a0 = sympy.symbols("a0")
a1 = sympy.symbols("a1")
a2 = sympy.symbols("a2")

#c0 = exp0 + a0

c0 = 0
c1 = 0
c2 = 0

for i in range(N):
    dyalti = dy[i,0] + a0 + a1*x[i,0] + a2*x[i,0]**2
    
    if i==0:
        c0 = dyalti
    #elif i==1:
        #c1 = dyalti
    #elif i==2:
        #c2 = dyalti
    
    c1 += dyalti
    #if i>0:
        #c1 += dyalti/x[i,0]**2
        #c1 += dyalti/x[i,0]
    c2 += dyalti*x[i,0]
    #c2 += dyalti*x[i,0]**2
    #c2 += dyalti/x[i,0]
    #c1 += exp0 + a0 + a1*xbar + a1*x2bar
    #c2 += exp1 + a0*xbar + a1*x2bar + a2*x3bar

c1 = c1.subs(k,0)
c2 = c2.subs(k,0)

#c1 += -M
#c2 += -M*x[4,0]

#a0s = a0
#sres = sympy.solve((c1,c2),(a1,a2))
sres = sympy.solve((c0,c1,c2),(a0,a1,a2))

print("sres")
print(sres)

a0 = sres[a0]
a1 = sres[a1]
a2 = sres[a2]

#a0 = -dy[0]
#a1 = 0
#a2 = 0



#print("a1")
#print(a1)
#print("a2")
#print(a2)

def dyaltinit(i,j):
    return dy[i,0] + a0 + a1*x[i,0] + a2*x[i,0]**2
    #return dy[i,0] - dy[0,0]
    #return dy[i,0] - dyavg - dyxavg
    

dyalt = Matrix(N,1,dyaltinit)

print("dyalt")
print(dyalt)


exp0alt = 0
exp1alt = 0
exp2alt = 0
for i in range(N):
    exp0alt += dyalt[i,0]/N
    exp1alt += dyalt[i,0]*x[i,0]/N
    if i>0:
        exp2alt += dyalt[i,0]/x[i,0]/N

exp0alt = exp0alt.simplify()
exp1alt = exp1alt.simplify()
exp2alt = exp2alt.simplify()
    


print("exp0alt")
print(exp0alt)
print("exp1alt")
print(exp1alt)
print("exp2alt")
print(exp2alt)

print("scale:")
print(scale)

print("factored:")
scale = sympy.factor(scale)
print(scale)


scale = scale.subs(L,sympy.sympify("8/10"))
scale = scale.subs(sigma,sympy.sympify("50/1000000"))
scale = scale.subs(a,sympy.sympify("4/1000"))
#scale = scale.subs(a,0)
scale = scale.subs(epsilon,1)
#scale = scale.subs(M,1)
scale = scale.subs(M,1)
#scale = scale.subs(M,1e-6)
scale = scale.subs(dA,1)
scale = scale.subs(dA2,1)
scale = scale.subs(phi,0)


print("partial fraction decomposition:")
scale = sympy.apart(scale)
print(scale)

scalealt = xtpre*(ymeas-dy)
scalealt = sympy.simplify(scalealt)

print("scalealt:")
print(scalealt)

#yalt2 = ymeas-dy
#for i in range(N):
    #yalt2[i] += -scalealt[0]

#scalealt2 = xtpre*(ymeas-dy-scalealt[0])
scalealt2 = xtpre*(ymeas-dyalt)
scalealt2 = sympy.simplify(scalealt2)
print("scalealt2")
print(scalealt2)

print("simplify")
#sigmak2 = var[0,0]
#sigmak2 = var[1,1]
#sigmak2 = 2*var[1,2]/k
#sigmak2 = 4*var[2,2]/k**2
sigmak2 = 4*var
#sigmak2 = varinv
#sigmak2 = var.det()
#sigmak2 = var[1,1] + var[2,2] - 2*var[1,2]
#sigmak2 = 4*var
#sigmak2 = var.diagonalize(sort=True)[1]
#sigmak2 = varinv.adjugate()
#sigmak2 = varinv
#sigmak2 = 4*(var[0,2]/sympy.sqrt(var[0,0]*var[2,2]))
sigmak2 = sympy.cancel(sigmak2)

print("sigmak2:")
print(sigmak2)

print("factored:")
sigmak2f = sympy.factor(sigmak2)
print(sigmak2f)

#sigmak2 = sigmak2.subs(L,8/10)
#sigmak2 = sigmak2.subs(sigma,10/1000000)
#sigmak2 = sigmak2.subs(a,4/1000)



sigmak2 = sigmak2.subs(L,sympy.sympify("8/10"))
sigmak2 = sigmak2.subs(sigma,sympy.sympify("50/1000000"))
sigmak2 = sigmak2.subs(a,sympy.sympify("4/1000"))

#sigmak2 = sigmak2.diagonalize()[1]

#sigmak2 = sympy.log(sigmak2)

sigmak2 = sympy.apart(sigmak2)

print("partial fraction decomposition:")
print(sigmak2)

assert(0)

#B = V.LUsolve(A)
#B = sympy.cancel(B)
#varinv = A.transpose()*B
#varinv = sympy.cancel(varinv)




selvector = zeros(3,1)
selvector[2,0] = 1

print("second solve")
varv = varinv.LUsolve(selvector)

sigmak2 = 4*varv[2,0]/k**2


print("simplify")


#sigmak2 = sigmak2.subs(L,1)
#sigmak2 = sigmak2.subs(sigma,1)
#sigmak2 = sigmak2.subs(a,1)
#sigmak2 = sympy.simplify(sigmak2)
sigmak2 = sympy.cancel(sigmak2)

print(sigmak2)

#sigmak2 = sympy.simplify(sigmak2)
#sigmak2 = sympy.cancel(sympy.expand(sigmak2))
#sigmak2 = sympy.factor(sigmak2)
#print(sigmak2)


#VL,VD = V.LDLdecomposition(hermitian=False)
#print(VL)
#print(VD)


#print("simplify")
#VL = sympy.simplify(VL)
#VD = sympy.simplify(VD)

#print(VL)
#print(VD)

#P,D = V.diagonalize()

#print(P)
#print(D)

#chol = V.cholesky(hermitian=False)
#print(chol)

#print("simplify")
#chol = sympy.cancel(chol)
#print(chol)







assert(0)

print("first invert")
Vinv = V.inv()
print("simplify")
#Vinv = sympy.simplify(Vinv)
Vinv = sympy.cancel(Vinv)
print("multiply")
varinv = A.transpose()*Vinv*A
print("simplify")
#varinv = sympy.simplify(varinv)
varinv = sympy.cancel(varinv)
#print("varinv", varinv.shape)


#print("decompose")
#LV = V.cholesky(hermitian=False)
#print("simplify")
#LV = sympy.simplify(LV)

#print("triangular solve")
#LA = LV.lower_triangular_solve(A)
#print("simplify")
#LA = sympy.simplify(LA)
#print("multiply")
#varinv = LA.transpose()*LA
#print("simplify")
#varinv = sympy.simplify(varinv)

selvector = zeros(3,1)
selvector[2,0] = 1

print("second solve")
varv = varinv.cholesky_solve(selvector)



assert(0)

#print(varinv)
#print("second invert")
#var = varinv.inv()
#assert(0)
    
#varinv = A.transpose()*V.inv()*A
#varinv = varinv.simplify()
#print(varinv)
#var = varinv.inv()

#print("first solve")
##varinv = A.transpose()*V.cholesky_solve(A)
##varinv = A.transpose()*V.LUsolve(A)
#varinv = A.transpose()*sympy.simplify(V.LUsolve(A))
##varinv = A.transpose()*sympy.simplify(V.cholesky_solve(A))


##varinv = (A.transpose()*V.cholesky_solve(A)).simplify()

#print("first simplify")
#varinv = sympy.simplify(varinv)
##varinv = sympy.factor(sympy.expand(varinv))



#varv = varinv.LUsolve(selvector)
#varv = varinv.LDLsolve(selvector)
#var = varinv.inv(method="LU")



#varinv  = A.transpose()*V.cholesky_solve(A)
#LV = V.cholesky(hermitian=False)
#LVinvA = LV.lower_triangular_solve(A)
#LVinvA = sympy.simplify(LVinvA)
#print("doing pinv")
#LVinvApinv = LVinvA.pinv()
#var = 4.*LVinvApinv.transpose()*LVinvApinv
#print(LVinvA)
#print(varinv)
#var = varinv.inv(method="LU")

#var = 4*var


#sigmak2 = 4*var[2,2]/k**2

sigmak2 = 4*varv[2,0]/k**2

print("simplify")

sigmak2 = sympy.simplify(sigmak2)
print(sigmak2)

#sigmak2 = sympy.expand(sigmak2)
sigmak2 = sigmak2.subs(L,1)
sigmak2 = sigmak2.subs(sigma,1)
sigmak2 = sigmak2.subs(a,1)
sigmak2 = sympy.simplify(sigmak2)
sigmak2 = sympy.apart(sigmak2)
#sigmak2 = sympy.factor(sigmak2)
print(sigmak2)
