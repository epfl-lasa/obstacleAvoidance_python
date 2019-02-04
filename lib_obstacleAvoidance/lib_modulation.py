
'''
Obstacle Avoidance Library with different options

@author Lukas Huber
@date 2018-02-15

'''

import numpy as np
import numpy.linalg as LA

import sys 
lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)
    
from lib_obstacleAvoidance import *
from dynamicalSystem_lib import *

# Only import needed function
from linear_modulations import *
from nonlinear_modulations import *

import warnings

def getGammmaValue_ellipsoid(ob, x_t):
    return np.sum( (x_t/np.tile(ob.a, (x_t.shape[1],1)).T) **(2*np.tile(ob.p, (x_t.shape[1],1) ).T ), axis=0)

def findRadius(ob, direction, a = [], repetition = 6, steps = 10):
    if not len(a):
        a = [np.min(ob.a), np.max(ob.a)]
        
    # repetition
    for ii in range(repetition):
        if a[0] == a[1]:
            return a[0]
        
        magnitudeDir = np.linspace(a[0], a[1], num=steps)
        Gamma = getGammmaValue_ellipsoid(ob, np.tile(direction, (steps,1)).T*np.tile(magnitudeDir, (np.array(ob.x0).shape[0],1)) )

        if np.sum(Gamma==1):
            return magnitudeDir[np.where(Gamma==1)]
        posBoundary = np.where(Gamma<1)[0][-1]

        a[0] = magnitudeDir[posBoundary]
        posBoundary +=1
        while Gamma[posBoundary]<=1:
            posBoundary+=1

        a[1] = magnitudeDir[posBoundary]
    return (a[0]+a[1])/2.0



def findRadius(ob, direction, a = [], repetition = 6, steps = 10):
    if not len(a):
        a = [np.min(ob.a), np.max(ob.a)]
        
    # repetition
    for ii in range(repetition):
        if a[0] == a[1]:
            return a[0]
        
        magnitudeDir = np.linspace(a[0], a[1], num=steps)
        Gamma = getGammmaValue_ellipsoid(ob, np.tile(direction, (steps,1)).T*np.tile(magnitudeDir, (np.array(ob.x0).shape[0],1)) )
        if np.sum(Gamma==1):
            return magnitudeDir[np.where(Gamma==1)]
        posBoundary = np.where(Gamma<1)[0][-1]

        a[0] = magnitudeDir[posBoundary]
        posBoundary +=1
        while Gamma[posBoundary]<=1:
            posBoundary+=1

        a[1] = magnitudeDir[posBoundary]
    return (a[0]+a[1])/2.0



def findBoundaryPoint(ob, direction):
    # Numerical search -- TODO analytic
    dirNorm = LA.norm(direction,2)
    if dirNorm:
        direction = direction/dirNorm
    else:
        print('No feasible direction is given')
        return ob.x0

    a = [np.min(x0.a), np.max(x0.a)]

    
    return (a[0]+a[1])/2.0*direction + x0



def compute_basis_matrix(d,x_t,obs, R):
    # For an arbitrary shape, the next two lines are used to find the shape segment
    th = np.arctan2(x_t[1],x_t[0])
    # if isfield(obs,.Tpartition.T):
    #     # TODO check
    #     ind = np.find(th>=(obs.partition(:,1)) & th<=(obs.partition(:,2)),1)
    # else:
    #     ind = 1
    
    ind = 1 # No partinioned obstacle
    #for ind in range(partitions)
    if hasattr(obs, 'sf'):
        a = np.array(obs.sf)*np.array(obs.a)
    elif hasattr(obs, 'sf_a'):
        #a = obs.a[:,ind] + obs.sf_a
        a = np.tile(obs.a, 2) + np.array(obs.sf_a)
    else:
        #a = obs.a[:,ind]
        a = np.array(obs.a)

    #p = obs.p[:,ind]
    p = np.array(obs.p)

    Gamma = np.sum((x_t/a)**(2*p))

    # TODO check calculation
    nv = (2*p/a*(x_t/a)**(2*p - 1)) #normal vector of the tangential hyper-plane

    E = np.zeros((d, d))
    
    if hasattr(obs,'center_dyn'):  # automatic adaptation of center 
        #R= compute_R(d, obs.th_r)
        E[:,0] = - (x_t - R.T @ (np.array(obs.center_dyn) - np.array(obs.x0)) )

        #E(:,1) = - (x_t - (obs.x_center*obs.a))
        #elif 'x_center' in obs: # For relative center
    #    E[:,0] = - (x_t - (obs.x_center*obs.a))
    else:
        E[:,0] = - x_t


    # Make diagonal to circle to improve behavior
    nv_hat = -x_t
    
    #generating E, for a 2D model it simply is: E = [dx [-dx(2)dx(1)]]
    E[0,1] = nv[1]
    E[1,1] = -nv[0]


    #     E[:,+1] = [0-nv(3)nv(2)]
    # if d==2:
    #     E[0,1] = nv[1]
    #     E[1,1] = -nv[0]
        
    #     E[:,0] = nv_hat
    
    if d==3:
        vec = np.array((nv[2],nv[0],nv[1])) # Random vector which is NOT nv
        E[:,1] = np.cross(nv, vec)
        #E[:,1] = E[:,1]/LA.norm(E[:,1])
           
        E[:,2] = np.cross(E[:,1], nv)
        #E[:,2] = E[:,1]/LA.norm(E[:,1])
    elif d>3:
        print('wrong method -- creates singularities')
        #generating E, for a 2D model it simply is: E = [dx [-dx(2)dx(1)]]
        E[0,1:d] = nv[1:d].T
        E[1:d,1:d] = -np.eye((d-1))*nv[0]

    if d>100: # General case
        for it_d in range(1,d):
            E[:d-(it_d+1), it_d] = nv[:d-(it_d+1)]*nv[d-(it_d+1)]
            E[d-(it_d+1), it_d] = -np.dot(nv[:d-(it_d+1)], nv[:d-(it_d+1)])*nv[d-(it_d+1)]
            E[:, it_d] = E[:, it_d]/LA.norm(E[:, it_d])
            
    E_orth = np.copy((E))
    E_orth[:,0] = nv
    
    # Linearize
    for ii in range(E.shape[1]):
        E[:,ii] = E[:,ii]/np.linalg.norm(E[:,ii])

    #print(E)
    
    return E, Gamma, E_orth
 
# def limit_vel(dx, v_max=1, x, x_attractor):
#     # Only start slowing down within radius of attractor 
#     dx_2 = sum(dx**2)
#     if v_max**2 > sum(dx**2): #velocity in normal range
#         return dx

#     # Get zero values
#     normXd = sqrt(sum(xd.^2,1));
#     ind0 = normXd ~= 0

#     xd[:,ind0] = xd(:,ind0).*np.tile(min(1./normXd(ind0),1)*velConst, (dim,1) )
    
#     xd[:,~ind0] = np.zeros(xd.shape[0] , x0.shape[1]) - ind0.shape[0] )
    
#     return dx/sqrt(dx_2)*v_max


def linearAttractorConst(x, x0 = 'default', velConst=2, distSlow=0.01):
    # change initial value for n dimensions
    # TODO -- constant velocity // maximum velocity
    
    dx = x0-x
    dx_mag = np.sqrt(np.sum(dx**2))

    if dx_mag: # nonzero value
        dx = min(1, 1/dx_mag)*velConst*dx
    
    return dx

# def obs_avoidance_rk4(dt, x, obs, obs_avoidance=obs_avoidance_interpolation, ds=linearAttractor, x0='default'):
def obs_avoidance_rk4(dt, x, obs, obs_avoidance=obs_avoidance_interpolation_moving, ds=linearAttractorConst, x0='default', k_f=0.75):

    # TODO -- add prediction of obstacle movement.
    # k1
    xd = ds(x, x0)*k_f
    xd = obs_avoidance(x, xd, obs)
    k1 = dt*xd

    # k2
    xd = ds(x+0.5*k1, x0)*k_f
    xd = obs_avoidance(x+0.5*k1, xd, obs)
    k2 = dt*xd

    # k3
    xd = ds(x+0.5*k2, x0)*k_f
    xd = obs_avoidance(x+0.5*k2, xd, obs)
    k3 = dt*xd

    # k4
    xd = ds(x+k3, x0)*k_f
    xd = obs_avoidance(x+k3, xd, obs)
    k4 = dt*xd

    # x final
    x = x + 1./6*(k1+2*k2+2*k3+k4) # + O(dt^5)

    return x


    
def orthogonalBasisMatrix(v):
    dim = v.shape[0]
    # Create orthogonal basis 
    V = np.eye((dim))
    
    return V


#def constVel_pos(xd, x, x_attr, kFact=0.3, v_max=1):
#    velFactor = np.min(kFact*np.linalg.norm(x-x_attr), v_max)
#    return xd /np.linalg.norm(xd)*velFactor
