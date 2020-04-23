import os
import multiprocessing
import functools

#ncpu = multiprocessing.cpu_count()

#os.environ["OMP_NUM_THREADS"] = str(ncpu)
#os.environ["OPENBLAS_NUM_THREADS"] = str(ncpu)
#os.environ["MKL_NUM_THREADS"] = str(ncpu)
#os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncpu)
#os.environ["NUMEXPR_NUM_THREADS"] = str(ncpu)

import jax
import jax.numpy as np
import numpy as onp
from jax import grad, hessian, jacobian, config
from jax.scipy.special import erf

config.update('jax_enable_x64', True)

def hessianlowmem(fun):
    def _hessianlowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        _, hvp = jax.linearize(jax.grad(funp), x)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return jax.lax.map(hvp,basis)
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _hessianlowmem

def jaclowmem(fun):
    def _jaclowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        _, jvp = jax.linearize(funp, x)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return jax.lax.map(jvp,basis)
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _jaclowmem



def hessianvlowmem(fun):
    def _hessianlowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        #_, hvp = jax.linearize(jax.grad(funp), x)
        def gdotv(x,v):
            return np.dot(jax.grad(funp)(x),v)
        def hvp(v):
            return jax.grad(gdotv)(x,v)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return jax.lax.map(hvp,basis)
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _hessianlowmem


def hessianvlowmem2(fun):
    def _hessianlowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        #_, hvp = jax.linearize(jax.grad(funp), x)
        def jvp(v):
            return jax.jvp(jax.grad(funp),(x,),(v,))[1]
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        jac = jax.lax.map(jvp,basis)
        hess = np.swapaxes(jac,0,1)
        return hess
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _hessianlowmem


def hessianvlowmem3(fun):
    def _hessianlowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        #_, hvp = jax.linearize(jax.grad(funp), x)
        def jvp(v):
            return jax.vjp(jax.grad(funp),v)[1](x)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        jac = jax.lax.map(jvp,basis)[0]
        #print(jac.shape)
        #hess = np.swapaxes(jac,0,1)
        return jac
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _hessianlowmem

#def vmap(f, chunksize=128):
    #vf = jax.vmap(f)
    

#parallel minimization using orthogonal basis subspaces based on arXiv:1506.07222,
#ported from tensorflow implementation in combinetf, but skipping all of the SR1 update
#parts since the number of parameters in each fit is small and hessian+eigen-decomposition
#can be explicitly computed

#f should be a scalar valued function, x is the parameter vector to be minimized
#x and any additional arguments should have an outer batch dimension over which fits will
#be parallelized, which could correspond to e.g. independent categories/bins, or sets of data

#if doParallel is False this can be used as a standard minimizer, where x is just the parameter vector as usual

def beval(f, batch_size=128, accumulator=lambda x: np.add(x,axis=0)):
    def _beval(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]
        idxstart = np.arange(0,length,batch_size)
        idxend = idxstart + batch_size
        #idxend =  np.arange(batch_size, length, batch_size)
        #if idxend.shape[0]<idxstart.shape[0]:
            #idxend = np.concatenate((idxend,np.array([length])))
        
        out_flat_batches = []
        for istart,iend in zip(idxstart,idxend):
            args_batch = []
            for arg in args:
                args_batch.append(arg[istart:iend])
            out_flat_batch, out_tree = jax.tree_flatten(f(*args_batch))
            out_flat_batches.append(out_flat_batch)

        out_flat_batches = [list(x) for x in zip(*out_flat_batches)]
        out_flat = []
        for out in out_flat_batches:
            out_flat.append(accumulator(out))
        out = jax.tree_util.tree_unflatten(out_tree, out_flat)
        return out
    return _beval

def vgrad(f, batch_size=128):
    #vg = jax.vmap(jax.jacfwd(f, holomorphic=True))
    vg = jax.vmap(jax.grad(f))
    vg = jax.jit(vg)
        
    def _vgrad(x, *args):
        n = x.shape[0]
        idxstart = np.arange(0,n,batch_size)
        idxend =  np.arange(batch_size, n, batch_size)
        if idxend.shape[0]<idxstart.shape[0]:
            idxend = np.concatenate((idxend,np.array([n])))

        outs = []
        for istart,iend in zip(idxstart,idxend):
            argsplit = []
            for arg in args:
                argsplit.append(arg[istart:iend])
            outs.append(vg(x[istart:iend],*argsplit))
            
        return np.concatenate(outs,axis=0)
    return _vgrad

def hgrad(f, batch_size=128):
    #vg = jax.vmap(jax.jacrev(jax.grad(f,holomorphic=True),holomorphic=True))
    vg = jax.vmap(jax.jacfwd((jax.jacrev(f))))
    #vg = jax.vmap(jax.hessian(f, holomorphic=True))
    vg = jax.jit(vg)

    
    def _vgrad(x, *args):
        n = x.shape[0]
        idxstart = np.arange(0,n,batch_size)
        idxend =  np.arange(batch_size, n, batch_size)
        if idxend.shape[0]<idxstart.shape[0]:
            idxend = np.concatenate((idxend,np.array([n])))

        outs = []
        for istart,iend in zip(idxstart,idxend):
            print("iter")
            argsplit = []
            for arg in args:
                argsplit.append(arg[istart:iend])
            outs.append(vg(x[istart:iend],*argsplit))
            
            
        return np.concatenate(outs,axis=0)
    return _vgrad
        

#def hgrad(f, batch_size=128):
    ##vg = jax.vmap(jax.jacrev(jax.grad(f,holomorphic=True),holomorphic=True))
    #vg = jax.vmap(jax.jacfwd((jax.jacrev(f))))
    ##vg = jax.vmap(jax.hessian(f, holomorphic=True))
    ##vg = jax.jit(vg)
    
    #vgs = []
    #for i in range(32):
        #vgs.append(jax.jit(vg, device=jax.devices()[i]))

    
    #def _vgrad(x, *args):
        #n = x.shape[0]
        #idxstart = np.arange(0,n,batch_size)
        #idxend =  np.arange(batch_size, n, batch_size)
        #if idxend.shape[0]<idxstart.shape[0]:
            #idxend = np.concatenate((idxend,np.array([n])))

        ##outs = []
        ##for istart,iend in zip(idxstart,idxend):
            ##print("iter")
            ##argsplit = []
            ##for arg in args:
                ##argsplit.append(arg[istart:iend])
            ##outs.append(vg(x[istart:iend],*argsplit))
            
        #lx = []
        #largs = []
        #for istart,iend in zip(idxstart,idxend):
            #print("iter")
            #argsplit = []
            #for arg in args:
                #argsplit.append(arg[istart:iend])
            #largs.append(argsplit)
            #lx.append(x[istart:iend])
            ##outs.append(vg(x[istart:iend],*argsplit))
        
        #outs = []
        #for i,(ix,iargs) in enumerate(zip(lx,largs)):
            #print("iter2")
            #ivg = vgs[i%32]
            #outs.append(ivg(ix,*iargs))
            
        #return np.concatenate(outs,axis=0)
    #return _vgrad
        
        
def hpgrad(f):
    vg = jax.vmap(jax.jacfwd((jax.jacrev(f))))
    vg = jax.jit(jax.pmap(vg))
    def _vgrad(x,*args):
        ncpu = 2
        n = x.shape[0]
        npadded = np.ceil(n/ncpu).astype(np.int32)*ncpu
        npad = npadded-n
        print(n, npadded, npad)
        xpad = np.pad(x,[(0,npad),(0,0)])
        xpad = np.reshape(xpad,(ncpu,-1,*x.shape[1:]))
        argspad = []
        for arg in args:
            print(arg.shape)
            argpad = np.pad(arg,[(0,npad),(0,0)])
            print(argpad.shape)
            argpad = np.reshape(argpad,(ncpu,-1,*arg.shape[1:]))
            argspad.append(argpad)
        res = jax.pmap(vg(xpad,*argspad))
        res = res = np.reshape(res,(-1,*res.shape[2:]))
        res = res[:n]
        return res
    return _vgrad
            
        
        

def pmin(f, x, args = [], doParallel=True):
    
    tol = np.sqrt(np.finfo('float64').eps)
    maxiter = int(100e3)
    
    trust_radius = np.ones(shape=x.shape[:-1], dtype=x.dtype)
    
    f = jax.jit(f)
    
    batch_size_grad = int(2048)
    batch_size = int(512)
    if doParallel:
        #g = jax.grad(lambda x,*args: np.sum(f(x,*args),axis=0))
        #g = jax.jit(g)
        
        fg = jax.jit(jax.vmap(jax.value_and_grad(f)))
        fg = beval(fg, accumulator=lambda x: np.concatenate(x,axis=0), batch_size=512)
        
        h  = jax.jit(jax.vmap(jax.hessian(f)))
        h = beval(h, accumulator=lambda x: np.concatenate(x,axis=0), batch_size=512)
        
        #hi = jax.jacrev(lambda x,*args: np.sum(g(x,*args),axis=0))
        #h = lambda x,*args: np.swapaxes(hi(x,*args),0,1)
        #h = jax.jit(h)
        
        #g = jax.jit(jax.vmap(jax.grad(f)))
        #h = jax.vmap(jax.hessian(f))
        #h = jax.jit(h,backend="cpu")
        
        #h = jax.jit(jax.vmap(jax.hessian(f)))
        #h = jax.
        #g = vgrad(f,batch_size=batch_size_grad)
        #h = hgrad(f,batch_size=batch_size)
        #h = hpgrad(f)
        #h = jax.jit(jax.hessian(f))
    else:
        fg = jax.jit(jax.value_and_grad(f))
        h = jax.jit(jax.hessian(f))
    
    
    
    print("starting fit")
    
    #fiter = jax.jit(piter, static_argnums=(0,4))
    
    
    val, grad = fg(x,*args)
    for i in range(maxiter):
        x, val, grad, trust_radius, edm, e0 = piter(fg,h,x,val,grad,trust_radius,args)
        gradmag = np.linalg.norm(grad,axis=-1)
        print("iter", i, np.sum(val), np.max(trust_radius), np.max(gradmag), np.sum(edm), np.max(edm), np.min(e0))
        if np.all(np.logical_and(e0>0, edm<tol)):
            break        
    return x

def piter(fg,h,x,val,grad,trust_radius, args):

        

        #fg = jax.value_and_grad(f)
        #h = jax.hessian(f)
        ##h = hessianlowmem(f)
        ##h = hessianvlowmem(f)
        
        #g = jax.jacrev(f)
        #h = jax.jacrev(g)
        #h = jaclowmem(g)
        #h = jax.hessian(f)
        
        #if doParallel:
            #fg = jax.vmap(fg)
            #g = jax.vmap(g)
            #h = jax.vmap(h)
            
        #assert(0)
        #val,grad = fg(x,*args)
        #hess = h(x,*args)
        

        
        #fgi = jax.value_and_grad(f)
        #hi = jax.hessian(f)
        
        ##fg = jax.vmap(fg)
        
        #def fgpack(xv):
            #return fgi(xv[0],*xv[1:])
        
        #def hpack(xv):
            #return hi(xv[0],*xv[1:])
        
        #val = f(x,*args)
        #grad = g(x,*args)
        #val,grad = fg(x,*args)
        #val,grad = jax.lax.map(fgpack,(x,*args))
        #hess = jax.lax.map(hpack,(x,*args))
        
        
        #print("evaluating")
        ##val = f(x,*args)
        ##print("done evaluating f")
        ##grad = g(x,*args)
        #val,grad = fg(x,*args)
        #print("done evaluating fg")
        hess = h(x,*args)
        #hess = np.eye(x.shape[-1],dtype=x.dtype)
        #hess = np.expand_dims(hess,axis=0)
        print("done evaluating")
        
        print("eigendecomposition")
        e,u = eigh(hess)
        #e.block_until_ready()
        print("eigendecomposition done")
        e0 = e[...,0]
        print("next")

        print("tr_solve")
        p, at_boundary, predicted_reduction, edm = tr_solve(grad,e,u,trust_radius)
        print("tr_solve done")

        #compute actual reduction in loss
        x_new = x + p
        val_new, grad_new = fg(x_new, *args)
        actual_reduction = val - val_new
        
        #update trust radius and output parameters, following Nocedal and Wright 2nd ed. Algorithm 4.1
        eta = 0.15
        trust_radius_max = 1e3
        rho = actual_reduction/np.where(np.equal(actual_reduction,0.), 1., predicted_reduction)
        trust_radius_out = np.where(rho<0.25, 0.25*trust_radius, np.where(np.logical_and(rho>0.75,at_boundary),np.minimum(2.*trust_radius, trust_radius_max),trust_radius))
        
        acceptsol = rho>eta
        #compute hessian only if needed
        #hess_new = jax.lax.cond(np.any(acceptsol), None, lambda _: h(x_new,*args), None, lambda _: hess)
        x_out = np.where(acceptsol[...,np.newaxis], x_new, x)
        val_out = np.where(acceptsol, val_new, val)
        grad_out = np.where(acceptsol[...,np.newaxis], grad_new, grad)
        #hess_out = np.where(acceptsol[...,np.newaxis,np.newaxis], hess_new, hess)
        
        return x_out, val_out, grad_out, trust_radius_out, edm, e0

eigh = jax.jit(np.linalg.eigh)

@jax.jit
def tr_solve(grad, e, u, trust_radius):

        #compute function value and gradient
        #val,grad = fg(x,*args)
        gradcol = np.expand_dims(grad, axis=-1)
        
        #compute hessian and eigen-decomposition
        #hess = h(x,*args)
        #hess = np.eye(x.shape[0],dtype=x.dtype)
        #e,u = np.linalg.eigh(hess)
        
        ut = np.swapaxes(u,-1,-2)
        
        #convert gradient to eigen-basis
        a = np.matmul(ut,gradcol)
        a = np.squeeze(a, axis=-1)
        
        lam = e
        e0 = lam[...,0]
        
        #compute estimated distance to minimum for unconstrained solution (only valid if e0>0)
        coeffs0 = -a/lam
        edm = -np.sum(a*coeffs0 + 0.5*lam*np.square(coeffs0), axis=-1)
        
        #TODO deal with null gradient components and repeated eigenvectors
        lambar = lam
        abarsq = np.square(a)

        
        def phif(s):
            s = np.expand_dims(s,-1)
            pmagsq = np.sum(abarsq/np.square(lambar+s),axis=-1)
            pmag = np.sqrt(pmagsq)
            phipartial = np.reciprocal(pmag)
            singular = np.any(np.equal(-s,lambar),axis=-1)
            phipartial = np.where(singular, 0., phipartial)
            phi = phipartial - np.reciprocal(trust_radius)
            return phi
        
        def phiphiprime(s):
            phi = phif(s)
            s = np.expand_dims(s,-1)
            pmagsq = np.sum(abarsq/np.square(lambar+s),axis=-1)        
            phiprime = np.power(pmagsq,-1.5)*np.sum(abarsq/np.power(lambar+s,3),axis=-1)
            return (phi, phiprime)
        
        #check if unconstrained solution is valid
        sigma0 = np.maximum(-e0, 0.)
        phisigma0 = phif(sigma0)
        usesolu = np.logical_and(e0>0., phisigma0>=0.)
        
        sigma = np.max(np.abs(a)/trust_radius[...,np.newaxis] - lam, axis=-1)
        sigma = np.maximum(sigma, 0.)
        sigma = np.where(usesolu, 0., sigma)
        phi, phiprime = phiphiprime(sigma)
        
        
        #TODO, add handling of additional cases here (singular and "hard" cases)
        
        #iteratively solve for sigma, enforcing unconstrained solution sigma=0 where appropriate
        unconverged = np.ones(shape=sigma.shape, dtype=np.bool_)
        j = 0
        maxiter=200

        
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
        coeffs = -a/(lam+sigma[...,np.newaxis])
        coeffscol = np.expand_dims(coeffs,axis=-1)
        
        p = np.matmul(u, coeffscol)
        p = np.squeeze(p, axis=-1)
        
        #compute predicted reduction in loss function from eigenvalues and eigenvectors
        predicted_reduction = -np.sum(a*coeffs + 0.5*lam*np.square(coeffs), axis=-1)
        
        at_boundary = np.logical_not(usesolu)
        
        return p, at_boundary, predicted_reduction, edm
        

        #return x_out, trust_radius_out, actual_reduction, predicted_reduction, rho, val, gradmag
