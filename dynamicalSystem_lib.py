'''

Library of different dynamical systems

@author Lukas Huber
@date 2018-02-15
'''

import numpy as np


def linearAttractor(x, x0='default'):
    # change initial value for n dimensions

    dim = x.shape[0]

    if type(x0)==str and x0=='default':
        x0 = dim*[0]
    
    #M = x.shape[1]
    M= 1
    X0 = np.kron( np.ones((1,M)), x0 )

    xd = -(x-x0)
    
    return xd


def linearAttractor_const(x, x0 = 'default', velConst=0.3, distSlow=0.01):
    # change initial value for n dimensions
    # TODO -- constant velocity // maximum velocity
    
    dx = x0-x
    dx_mag = np.sqrt(np.sum(dx**2))
    
    dx = min(velConst, 1/dx_mag*velConst)*dx

    return dx


def nonlinear_wavy_DS(x, x0=[0,0]):
    xd = np.zeros((np.array(x).shape))
    if len(xd.shape)>1:
        xd[0,:] = - x[1,:] * np.cos(x[0,:]) - x[0,:]
        xd[1,:] = - x[1,:]
    else:
        xd[0] = - x[1] * np.cos(x[0]) - x[0]
        xd[1] = - x[1]
    return xd

def nonlinear_stable_DS(x, x0=[0,0], pp=3 ):
    xd = np.zeros((np.array(x).shape))
    if len(xd.shape)>1:
        xd[0,:] = - x[1,:]
        xd[1,:] = - np.copysign(x[1,:]**pp, x[1,:])
    else:
        xd[0] = - x[0]
        xd[1] = - np.copysign(np.abs(x[1])**pp, x[1])
    return xd



def constVelocity(dx, x, x0=[0,0], velConst = 0.2, distSlow=0.01):
    dx_mag = np.sqrt(np.sum(np.array(dx)**2))
    
    if dx_mag: # nonzero value
        dx = min(1, 1/dx_mag)*velConst*dx

    return dx
