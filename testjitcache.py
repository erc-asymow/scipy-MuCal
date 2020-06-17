import jax
import jax.numpy as np
import timeit

def fun(x):
    #return np.log(np.sum(np.exp(x)))
    res = np.zeros(())
    for i in range(10000):
        res += np.sum(np.exp(x))
    return res

def testperf(f1,f2,x1,x2):
    res1 = f1(x1)
    res2 = f2(x2)
    
    res1.block_until_ready()
    res2.block_until_ready()

n1 = int(1.0e4)
n2 = int(1.01e4)

a1 = np.linspace(0.,10.,n1)
a2 = np.linspace(0.,10.,n2)

g = jax.grad(fun)

jf1 = jax.jit(g)
jf2 = jax.jit(g)

#jf1 = g
#jf2 = g

jf1(a1).block_until_ready()
jf2(a1).block_until_ready()
#jf1(a2).block_until_ready()
#jf2(a2).block_until_ready()

t1 = timeit.timeit(lambda: testperf(jf1,jf2,a1,a2), number=1000)

t2 = timeit.timeit(lambda: testperf(jf1,jf1,a1,a2), number=1000)
t3 = timeit.timeit(lambda: testperf(jf1,jf1,a1,a1), number=1000)


print(t1,t2,t3)


