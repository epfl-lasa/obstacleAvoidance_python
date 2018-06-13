'''

Library of different dynamical systems

@author Lukas Huber
@date 2018-02-15
'''

import numpy as np


def linearAttractor(x, x0='default'):
    # change initial value for n dimensions

    dim = x.shape[0]

    if x0=='default':
        x0 = dim*[0]
    
    #M = x.shape[1]
    M= 1
    X0 = np.kron( np.ones((1,M)), x0 )

    
    xd = -(x-x0)
    
    return xd
