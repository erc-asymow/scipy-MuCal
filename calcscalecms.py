import sympy
from sympy.matrices import Matrix, ones, zeros, hadamard_product, BlockMatrix, block_collapse
from sympy import init_printing
init_printing() 

#N = 5

R = [4.1437478, 7.0870599, 7.5626616, 11.746180, 28.888109, 28.979446, 33.737079, 33.829258, 35.048137, 35.139869, 39.898342, 39.989845,
52.630050, 57.872253, 61.893707, 62.010196, 67.135734, 67.253265,
73.687667, 73.806480, 74.882408, 81.997367, 89.673179, 96.831947]


RPhiErr = [0.0005205, 0.0006515, 0.0004069, 0.0047973, 0.0055708, 0.0055551,
0.0054941, 0.0043147, 0.392515, 0.0041837, 0.429451, 0.0042747,
0.465744, 0.0038275, 0.0038912, 0.003795, 0.233091, 0.0038282,
0.265093, 0.0029776, 0.270927, 0.0030374, 0.31596, 0.0023857]


matval = [ 4.753e-6 , 4.06e-6 , 4.06e-6 , 0.00001872, 0.000028, 0.000028,
0.00001061, 0.00001061, 5.467e-6, 5.467e-6 , 8.882e-6 , 8.882e-6 ,
0.00001315, 9.93e-6, 4.358e-6 , 4.358e-6 , 4.691e-6, 4.691e-6,
4.88e-6 , 4.88e-6 , 9.76e-6 , 8.704e-6 , 8.388e-6, 8.37e-6]

N = len(R)



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

#def xinit(i,j):
    #return i*L/(N-1)

def xinit(i,j):
    #return (i+1)*L/N
    return R[i]
    
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

#M1 = sympy.symbols("M1")
#M2 = sympy.symbols("M2")

def yinit(i,j):
    
    res = k*x[i,0]*(x[i,0]-x[N-1,0])
    
    if i==3:
        #res += dA*k + M - epsilon*k**2
        res += dA*k
        res += M
        #res += epsilon*k**2 + dA*k
        #res += epsilon*k**2
    if i==4:
        res += dA2*k
        #res += epsilon*k**2/(1.+epsilon*k)
        
    #if i==4:
        #res += M2
        
    return res
    

ymeas = Matrix(N,1,yinit)

def Vinit(i,j):
    res = 0
    if i==j:
        res += RPhiErr[i]**2
        
    for l in range(min(i,j)):
        res += (x[i,0] - x[l,0])*(x[j,0] - x[l,0])*matval[l]*k**2
        
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
#Vinv = V.inv(method = "ADJ")
Vinv = V.inv(method = "LU")
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


print("partial fraction decomposition:")
scale = sympy.apart(scale)
print(scale)

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
