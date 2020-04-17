import os
import multiprocessing

ncpu = multiprocessing.cpu_count()

os.environ["OMP_NUM_THREADS"] = str(ncpu)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncpu)
os.environ["MKL_NUM_THREADS"] = str(ncpu)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncpu)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncpu)

import jax
import jax.numpy as np
import numpy as onp
from jax import grad, hessian, jacobian, config
from jax.scipy.special import erf

config.update('jax_enable_x64', True)


#parallel minimization using orthogonal basis subspaces
#f should be a scalar valued function, x is the parameter vector to be minimized
#x and any additional arguments should have an outer batch dimension over which fits will
#be parallelized, which could correspond to e.g. independent categories/bins, or sets of data
def pmin(f, x, args = []):
    
    xtol = np.sqrt(np.finfo('float64').eps)
    maxiter = int(100e3)
    
    trust_radius = np.ones(shape=x.shape[:-1], dtype=x.dtype)
    
    def fiter(x, trust_radius, args):
        return piter(f, x, trust_radius, args)
        
    vfiter = jax.vmap(fiter)
    
    #jit compile function where arguments not varying between iterations treated as static
    static_argnums = (2,)
    vfiter = jax.jit(vfiter, static_argnums=static_argnums)
    
    for i in range(maxiter):
        x,trust_radius,val,gradmag = vfiter(x,trust_radius,args)
        #x,trust_radius, actual_reduction, predicted_reduction, rho, val, gradmag = vfiter(x,trust_radius,args)
        #maxidx = np.argmax(trust_radius)
        #print(i, val, trust_radius, gradmag, np.max(trust_radius), np.max(gradmag), actual_reduction[maxidx],predicted_reduction[maxidx])
        print(i, np.sum(val), np.max(trust_radius), np.max(gradmag))
        if np.all(trust_radius<xtol):
            break

def piter(f, x, trust_radius, args):

        fg = jax.value_and_grad(f)
        g = jax.grad(f)
        h = jax.hessian(f)
        
        #compute function value
        
        #compute function value and gradient
        val,grad = fg(x,*args)
        gradmag = np.linalg.norm(grad,axis=-1)
        gradcol = np.expand_dims(grad, axis=-1)
        
        #compute hessian and eigen-decomposition
        hess = h(x,*args)
        e,u = np.linalg.eigh(hess)
        
        ut = u.T
        
        #convert gradient to eigen-basis
        a = np.matmul(ut,gradcol)
        a = np.squeeze(a, axis=-1)
        
        lam = e
        
        #TODO deal with null gradient components and repeated eigenvectors
        lambar = lam
        abarsq = np.square(a)
        
        e0 = lam[...,0]
        
        def phif(s):        
            pmagsq = np.sum(abarsq/np.square(lambar+s),axis=-1)
            pmag = np.sqrt(pmagsq)
            phipartial = np.reciprocal(pmag)
            singular = np.any(np.equal(-s,lambar),axis=-1)
            phipartial = np.where(singular, 0., phipartial)
            phi = phipartial - np.reciprocal(trust_radius)
            return phi
        
        def phiphiprime(s):
            phi = phif(s)
            pmagsq = np.sum(abarsq/np.square(lambar+s),axis=-1)        
            phiprime = np.power(pmagsq,-1.5)*np.sum(abarsq/np.power(lambar+s,3),axis=-1)
            return (phi, phiprime)
        
        #check if unconstrained solution is valid
        sigma0 = np.maximum(-e0, 0.)
        phisigma0 = phif(sigma0)
        usesolu = np.logical_and(e0>0., phisigma0>=0.)
        
        sigma = np.max(np.abs(a)/trust_radius - lam, axis=-1)
        sigma = np.maximum(sigma, 0.)
        sigma = np.where(usesolu, 0., sigma)
        phi, phiprime = phiphiprime(sigma)
        
        
        #TODO, add handling of additional cases here (singular and "hard" cases)
        #iteratively solve for sigma, enforcing unconstrained solution sigma=0 where appropriate
        unconverged = np.ones(shape=sigma.shape, dtype=np.bool_)
        j = 0
        maxiter=int(100e3)

        
        #This can't work with vmap+jit because of the dynamic condition, so we use the jax while_loop below
        #j = 0
        #while np.logical_and(np.any(unconverged), j<maxiter):
            #sigma = sigma - phi/phiprime
            #sigma = np.where(usesolu, 0., sigma)
            #phiout, phiprimeout = phiphiprime(sigma)
            #unconverged = np.logical_and( (phiout > phi) , (phiout < 0.) )
            #phi,phiprime = (phiout, phiprimeout)
            #j = j +1

        def cond(vals):
            sigma, phi, phiprime, unconverged, j = vals
            return np.logical_and(np.any(unconverged) , j<maxiter)

        def body(vals):
            sigma, phi, phiprime, unconverged, j = vals
            sigma = sigma - phi/phiprime
            sigma = np.where(usesolu, 0., sigma)
            phiout, phiprimeout = phiphiprime(sigma)
            unconverged = np.logical_and( (phiout > phi) , (phiout < 0.) )
            phi,phiprime = (phiout, phiprimeout)
            j = j +1
            return (sigma, phi, phiprime, unconverged, j)
        
        sigma = jax.lax.while_loop(cond, body, (sigma,phi,phiprime,unconverged,j))[0]
        
            
        #compute solution from eigenvalues and eigenvectors
        coeffs = -a/(lam+sigma)
        coeffscol = np.expand_dims(coeffs,axis=-1)
        
        p = np.matmul(u, coeffscol)
        p = np.squeeze(p, axis=-1)
        
        #compute predicted reduction in loss function from eigenvalues and eigenvectors
        predicted_reduction = -np.sum(a*coeffs + 0.5*lam*np.square(coeffs), axis=-1)
        
        #compute actual reduction in loss
        x_new = x + p
        val_new = f(x_new, *args)
        #actual_reduction = -(val_new - val)
        actual_reduction = val-val_new
        
        #update trust radius and output parameters, following Nocedal and Wright 2nd ed. Algorithm 4.1
        eta = 0.15
        trust_radius_max = 1e3
        rho = actual_reduction/np.where(np.equal(actual_reduction,0.), 1., predicted_reduction)
        at_boundary = np.logical_not(usesolu)
        trust_radius_out = np.where(rho<0.25, 0.25*trust_radius, np.where(np.logical_and(rho>0.75,at_boundary),np.minimum(2.*trust_radius, trust_radius_max),trust_radius))
        
        x_out = np.where(rho>eta, x_new, x)
        
        return x_out, trust_radius_out, val, gradmag
        #return x_out, trust_radius_out, actual_reduction, predicted_reduction, rho, val, gradmag
