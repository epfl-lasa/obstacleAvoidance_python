'''

Library of different dynamical systems

@author Lukas Huber
@date 2018-02-15
'''

import numpy as np


def linearAttractor(x, x0 = [0,0]):
    # change initial value for n dimensions

    #M = x.shape[1]
    M= 1
    X0 = np.kron( np.ones((1,M)), x0 )
    #import pdb; pdb.set_trace() ## DEBUG ##
    
    xd = -(x-x0)
    
    return xd
