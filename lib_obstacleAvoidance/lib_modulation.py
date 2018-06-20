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
    
    xd_obs = np.zeros((d))
    
    for n in range(N_obs):
    #     x_temp = x-np.array(obs[n].x0)
    #     xd_w_obs = np.array([-x_temp[1], x_temp[0]])*w[n]

        if d==2:
            xd_w = np.cross(np.hstack(([0,0], obs[n].w)),
                            np.hstack((x-np.array(obs[n].x0),0)))
                            
            xd_w = xd_w[0:2]
        else:
            xd_w = np.cross( obs[n].w, x-obs[n].x0 )

        xd_obs = xd_obs + w[n]*np.exp(-1/obs[n].sigma*(max([Gamma[n],1])-1))*  ( np.array(obs[n].xd) + xd_w )
        #xd_obs = xd_obs + w[n]* ( np.array(obs[n].xd) + xd_w )
        
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

        M = (R[:,:,n] @ E[:,:,n] @ np.diag(D+np.hstack((1,d0)) ) @ LA.pinv(E[:,:,n]) @ R[:,:,n].T) @ M

    xd = M @ xd #velocity modulation
    #if LA.norm(M*xd)>0.05:
    #    xd = LA.norm(xd)/LA.norm(M*xd)*M @xd #velocity modulation
    xd = xd + xd_obs # transforming back the velocity into the global coordinate system
    return xd


def obs_avoidance_interpolation(x, xd, obs):
    # TODO -- speed up for multiple obstacles, not everything needs to be safed...
    
    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    d = x.shape[0]
    Gamma = np.zeros((N_obs))

    # Linear and angular roation of velocity
    xd_dx_obs = np.zeros((d,N_obs))
    xd_w_obs = np.zeros((d,N_obs)) #velocity due to the rotation of the obstacle

    #M = np.zero((d, d, N_obs))
    E = np.zeros((d,d,N_obs))
    R = np.zeros((d,d,N_obs))

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
    xd_obs = np.zeros((d))
    
    for n in range(N_obs):
        if d==2:
            xd_w = np.cross(np.hstack(([0,0], obs[n].w)),
                            np.hstack((x-np.array(obs[n].x0),0)))
            xd_w = xd_w[0:2]
        elif d==3:
            xd_w = np.cross( obs[n].w, x-obs[n].x0 )
        else:
            print('WARNING - NOT implemented for d={}'.format(d))

        #the exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object
        xd_obs = xd_obs + w[n]*np.exp(-1/obs[n].sigma*(max([Gamma[n],1])-1))*  ( np.array(obs[n].xd) + xd_w )

    xd = xd-xd_obs #computing the relative velocity with respect to the obstacle

    # Create orthogonal matrix
    #Ff = orthogonalBasisMatrix(xd)
    xd_norm = LA.norm(xd)
    if xd_norm:#nonzero
        xd_n = xd/xd_norm

    xd_t = np.array([xd_n[1], -xd_n[0]])
 
    Ff = np.array([xd_n, xd_t])

    #R = LA.inv(F)
    Rf = Ff.T # ?! True???

    M = np.zeros((d,d,N_obs))
    xd_hat = np.zeros((d, N_obs))
    xd_mags = np.zeros((N_obs))
    k_ds = np.zeros((d-1, N_obs))
    
    for n in range(N_obs):
        if hasattr(obs[n], 'rho'):
            rho = obs[n].rho
        else:
            rho = 1

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
        
        M[:,:,n] = (R[:,:,n] @ E[:,:,n] @ np.diag(D+np.hstack((1,d0)) ) @ LA.pinv(E[:,:,n]) @ R[:,:,n].T)
        xd_hat[:,n] = M[:,:,n] @ xd #velocity modulation
        xd_mags[n] = np.sqrt(np.sum(xd_hat[:,n]**2))
        xd_hat_n = xd_hat[:,n]/xd_mags[n]
        
        if not d==2:
            print('WARNING not implemented for d neq 2')

        Rfn = Rf @ xd_hat_n
        k_fn = Rfn[1:]
        kfn_norm = LA.norm(k_fn) # Normalize
        if kfn_norm:#nonzero
            k_fn = k_fn/ kfn_norm
        k_ds[:,n] = np.arccos(np.sum(xd_hat_n*xd_n))*k_fn.squeeze()
        
    xd_mags = np.sqrt(np.sum(xd_hat**2, axis=0) )

    # Weighted interpolation
    xd_mag = np.sum(xd_mags*w)
    k_d = np.sum(k_ds*np.tile(w, (d-1, 1)), axis=1)

    norm_kd = LA.norm(k_d)
    
    # Reverse k_d
    if norm_kd: #nonzero
        n_xd = Rf.T @ np.hstack((np.cos(norm_kd), np.sin(norm_kd)/norm_kd*k_d ))
    else:
        n_xd = np.zeros((d))
        n_xd[0] = 1

    xd = xd_mag*n_xd.squeeze()
    
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
    
def orthogonalBasisMatrix(v):
    dim = v.shape[0]
    # Create orthogonal basis 
    V = np.eye((dim))
    
    return V
    
