import jax
import jax.numpy as np
import numpy as onp
from jax import grad, hessian, jacobian, config
from jax.scipy.special import erf
config.update('jax_enable_x64', True)

from obsminimization import flatten, unflatten, flatten_hessian


def fun(parms):
    val = np.zeros(shape = (), dtype=np.float64)
    for parm in parms:
        val += np.sum(np.square(parm))
    return val


parms = []
parms.append(2.*np.ones(shape=(4,3),dtype=np.float64))
parms.append(2.*np.ones(shape=(7,5),dtype=np.float64))

#parms.append(2.*np.ones(shape=(4,),dtype=np.float64))
#parms.append(2.*np.ones(shape=(7,),dtype=np.float64))

val = fun(parms)

grad = jax.grad(fun)(parms)
hess = jax.hessian(fun)(parms)

gradflat, split_idxs, shapes, x_tree = flatten(grad)

hessflat = flatten_hessian(hess,shapes,val.shape)

print(hessflat)
print(hessflat.shape)

#print(val.shape)

##print(len(grad))
##print(len(hess))

#print("grad shapes")
#for g in grad:
    #print(g.shape)

#print("hess shapes")
#for hi in hess:
    #for hj in hi:
        #print(hj.shape)
        
#print(np.block(hess))

#print(grad.shape)
#print(hess.shape)
             
        
