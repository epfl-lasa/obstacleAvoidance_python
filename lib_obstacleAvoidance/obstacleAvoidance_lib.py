'''
Obstacle Avoidance Library with different options

@author Lukas Huber
@date 2018-02-15

'''

import numpy as np
import numpy.linalg as LA

import sys 
lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm_python/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)
    
from lib_obstacleAvoidance import *

def obs_avoidance_convergence(x, xd, obs):

    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    d = x.shape[0]
    Gamma = np.zeros((N_obs))

    # Linear and angular roation of velocity
    xd_dx_obs = np.zeros((d,N_obs))
    xd_w_obs = np.zeros((d,N_obs)) #velocity due to the rotation of the obstacle

    # Obstacle parameters
    # for i in range(len(varargin)):
    #     if 'xd_obs' in varargin:
    #         xd_dx_obs = varargin.xd_obs
    #     if 'w_obs' in varargin:
    #         w_obs = varargin.w_obs
            
        #If i==0:
        #     xd_dx_obs = varargin[i]
        # elif i==1:
        #     w_obs = varargin[i]
        #     if d==2 and w_obs.shape[0]==1 and w_obs.shape[2]==N_obs:
        #         for n in range(N):
        #             x_tmp = x-obs[n].x0
        #             xd_w_obs[:,n] = [-x_tmp[1]*x_tmp[0]]*w_obs[n] #cross(w,x_tmp)

            # if d==3 && len(w_obs)==d: #supporting situation
                #     xd_w_obs = cross(w_obs,x_tmp)

    # if d==3:
    #     E = np.zeros((d,d+1,N_obs))
    # else:
    E = np.zeros((d,d,N_obs))

    R = np.zeros((d,d,N_obs))
    M = np.eye(d)

    for n in range(N_obs):
        # rotating the query point into the obstacle frame of reference
        if obs[n].th_r: # Greater than 0
            R[:,:,n] = compute_R(d,obs[n].th_r)
        else:
            R[:,:,n] = np.eye(d)

        # Move to obstacle centered frame
        x_t = R[:,:,n].T @ (x-obs[n].x0)
        
        E[:,:,n], Gamma[n] = compute_basis_matrix(d,x_t,obs[n], R[:,:,n])
                        
        # if Gamma[n]<0.99: 
        #     print(Gamma[n])
    w = compute_weights(Gamma,N_obs)

    #adding the influence of the rotational and cartesian velocity of the
    #obstacle to the velocity of the robot
    #xd_obs = np.zeros((dim, N_obs))
    xd_obs = 0
    # for n in range(N_obs):
    #     x_temp = x-np.array(obs[n].x0)
    #     xd_w_obs = np.array([-x_temp[1], x_temp[0]])*w[n]

    #     xd_obs = xd_obs + w[n] * np.exp(-1/obs[n].sigma*(max([Gamma[n], 1])-1) )* (obs[n].xd + xd_w_obs)
        
        #the exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object

    #xd = xd-xd_obs[n] #computing the relative velocity with respect to the obstacle
    xd = xd-xd_obs #computing the relative velocity with respect to the obstacle

    #ordering the obstacle number so as the closest one will be considered at
    #last (i.e. higher priority)
    # obs_order  = np.argsort(-Gamma)
    
    for n in range(N_obs):
        # if 'rho' in obs[n]:
        if hasattr(obs[n], 'rho'):
            rho = obs[n].rho
        else:
            rho = 1

    #     if isfield(obs[n],'eigenvalue')
    #         d0 = obs[n].eigenvalue
    #     else:
        d0 = np.ones((E.shape[1]-1))

        if Gamma[n]==0:
            print('Gamma:', Gamma[n])
        D = w[n]*(np.hstack((-1,d0))/abs(Gamma[n])**(1/rho))
        #     if isfield(obs[n],'tailEffect') && ~obs[n].tailEffect && xdT*R(:,:,n)*E(:,1,n)>=0 #the obstacle is already passed, no need to do anything
        #         D(1) = 0.0

        if D[0] < -1.0:
            D[1:] = d0
            if xd.T @ R[:,:,n] @ E[:,1,n] < 0:
                D[0] = -1.0

        # TO DO --- Division sign....
        M = (R[:,:,n] @ E[:,:,n] @ np.diag(D+np.hstack((1,d0)) ) @ LA.pinv(E[:,:,n]) @ R[:,:,n].T) @ M

    # print('xd_init', xd)
    # print('M',M)
    # print('E',E[:,:,0])
    # print('R',R[:,:,0])

    
    xd = M @ xd #velocity modulation 
    #if LA.norm(M*xd)>0.05:
    #    xd = LA.norm(xd)/LA.norm(M*xd)*M @xd #velocity modulation

    
    xd = xd + xd_obs # transforming back the velocity into the global coordinate system
    return xd


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
    
    if hasattr(obs,'center_dyn'): # automatic adaptation of center 
        #R= compute_R(d, obs.th_r)
        E[:,0] = - (x_t - R.T @ (np.array(obs.center_dyn) - np.array(obs.x0)) )

        #E(:,1) = - (x_t - (obs.x_center*obs.a))
        #elif 'x_center' in obs: # For relative center
    #    E[:,0] = - (x_t - (obs.x_center*obs.a))
    else:
        E[:,0] = - x_t

    E[1:d,1:d] = -np.eye(d-1)*nv[0]

    # Make diagonal to circle to improve behavior
    nv_hat = -x_t
    
    #generating E, for a 2D model it simply is: E = [dx [-dx(2)dx(1)]]
    E[0,1:d] = nv[1:d].T
    E[1:d,1:d] = -np.eye((d-1))*nv[0]

    # if d == 3:
    #     E[:,+1] = [0-nv(3)nv(2)]
    return E, Gamma
