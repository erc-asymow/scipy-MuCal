#import numpy as np
import jax
import jax.numpy as np
from jax import grad, hessian, jacobian, config
from jax.scipy.special import erf
config.update('jax_enable_x64', True)

etas = np.arange(-0.8, 1.2, 0.4)
#etas = np.array((-0.8,0.8))
ptsJ = np.array((4.31800938,5.2385931,6.77045488,930.92053223),dtype='float64') #quantiles
ptsJC = np.array((0.1703978,0.21041214,0.26139158),dtype='float64') #bin centers in curvature
ptsZ = np.array((20.,30,40,50,60,70,100),dtype='float64')
#pts = np.array((3.,20.))
