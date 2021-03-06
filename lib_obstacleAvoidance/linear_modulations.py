
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

import warnings


def obs_avoidance_ellipsoid(x, xd, obs):
    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    if N_obs ==0:
        return xd
    
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
        
        E[:,:,n], Gamma[n], E_ortho = compute_basis_matrix(d,x_t,obs[n], R[:,:,n])
        E[:,:,n] = E_ortho               
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
        #print('d0', d0)
        #print('E', E)
        d0 = np.ones((E.shape[1]-1))

        if Gamma[n]==0:
            if not w[n]==0:
                print('Gamma:', Gamma[n])
                print('n', n)
                print('w', w[n])
            D = w[n]*(np.hstack((-1,d0)))
        else:
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

def obs_avoidance_convergence(x, xd, obs):
    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    if N_obs ==0:
        return xd
    
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
        
        E[:,:,n], Gamma[n], E_ortho = compute_basis_matrix(d,x_t,obs[n], R[:,:,n])
                        
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
        #print('d0', d0)
        #print('E', E)
        d0 = np.ones((E.shape[1]-1))

        if Gamma[n]==0:
            if not w[n]==0:
                print('Gamma:', Gamma[n])
                print('n', n)
                print('w', w[n])
            D = w[n]*(np.hstack((-1,d0)))
        else:
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
    if N_obs ==0:
        return xd
    
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
        
        E[:,:,n], Gamma[n], E_ortho = compute_basis_matrix( d,x_t,obs[n], R[:,:,n])
                        
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
            warnings.warn('NOT implemented for d={}'.format(d))

        #the exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object
        xd_obs = xd_obs + w[n]*np.exp(-1/obs[n].sigma*(max([Gamma[n],1])-1))*  ( np.array(obs[n].xd) + xd_w )

    xd = xd-xd_obs #computing the relative velocity with respect to the obstacle

    # Create orthogonal matrix
    #Ff = orthogonalBasisMatrix(xd)
    xd_norm = LA.norm(xd)
    if xd_norm:#nonzero
        xd_n = xd/xd_norm
    else:
        xd_n=xd
        
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
            if not w[n] == 0:
                print('Gamma:', Gamma[n])
            D = w[n]*(np.hstack((-1,d0)))
        else:
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
        if xd_mags[n]: # Nonzero magnitude
            xd_hat_n = xd_hat[:,n]/xd_mags[n]
        else:
            xd_hat_n = xd_hat[:,n]
        
        if not d==2:
            warnings.warn('not implemented for d neq 2')

        Rfn = Rf @ xd_hat_n
        k_fn = Rfn[1:]
        kfn_norm = LA.norm(k_fn) # Normalize
        if kfn_norm:#nonzero
            k_fn = k_fn/ kfn_norm

            
        sumHat = np.sum(xd_hat_n*xd_n)
        if sumHat > 1 or sumHat < -1:
            sumHat = max(min(sumHat, 1), -1)
            warnings.warn(' cosinus out of bound!')
            
        k_ds[:,n] = np.arccos(sumHat)*k_fn.squeeze()
        
    xd_mags = np.sqrt(np.sum(xd_hat**2, axis=0) )

    # Weighted interpolation
    weightPow = 2 # Hyperparameter for several obstacles !!!!
    w = w**weightPow
    if not np.linalg.norm(w,2):
        warnings.warn('trivial weight.')
    w = w/np.linalg.norm(w,2)
    
    xd_mag = np.sum(xd_mags*w)
    k_d = np.sum(k_ds*np.tile(w, (d-1, 1)), axis=1)

    norm_kd = LA.norm(k_d)
    
    # Reverse k_d
    if norm_kd: #nonzero
        n_xd = Rf.T @ np.hstack((np.cos(norm_kd), np.sin(norm_kd)/norm_kd*k_d ))
    else:
        n_xd = Rf.T @  np.hstack((1, np.zeros((d-1)) ))
        
    xd = xd_mag*n_xd.squeeze()
    
    #if LA.norm(M*xd)>0.05:
    #    xd = LA.norm(xd)/LA.norm(M*xd)*M @xd #velocity modulation

    xd = xd + xd_obs # transforming back the velocity into the global coordinate system

    #if  (str(float(xd[0] )).lower() == 'nan' or
    #     str(float(xd[1] )).lower() == 'nan'):
        
    assert(not( str(float(xd[0] )).lower() == 'nan'))
    assert(not( str(float(xd[1] )).lower() == 'nan'))

    return xd



def obs_avoidance_nonlinear(x, xd, obs, ds,):
    # TODO -- speed up for multiple obstacles, not everything needs to be safed...

    if d==2: # TODO for higher dimensions
        f_nonLinear = xd
        # xd_nonLinear = xd
        # TODO: f_lin = ds(x_ref)
        f_linear = np.array([-1,0]) # velocity at center
        xd = f_linear
    
    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    if N_obs ==0:
        return xd
    
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
        
        E[:,:,n], Gamma[n], E_ortho = compute_basis_matrix( d,x_t,obs[n], R[:,:,n])
                        
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
            warnings.warn('NOT implemented for d={}'.format(d))

        #the exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object
        xd_obs = xd_obs + w[n]*np.exp(-1/obs[n].sigma*(max([Gamma[n],1])-1))*  ( np.array(obs[n].xd) + xd_w )

    xd = xd-xd_obs #computing the relative velocity with respect to the obstacle

    # Create orthogonal matrix
    #Ff = orthogonalBasisMatrix(xd)
    xd_norm = LA.norm(xd)
    if xd_norm:#nonzero
        xd_n = xd/xd_norm
    else:
        xd_n=xd
        
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
            if not w[n] == 0:
                print('Gamma:', Gamma[n])
            D = w[n]*(np.hstack((-1,d0)))
        else:
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
        if xd_mags[n]: # Nonzero magnitude
            xd_hat_n = xd_hat[:,n]/xd_mags[n]
        else:
            xd_hat_n = xd_hat[:,n]
        
        if not d==2:
            warnings.warn('not implemented for d neq 2')

        Rfn = Rf @ xd_hat_n
        k_fn = Rfn[1:]
        kfn_norm = LA.norm(k_fn) # Normalize
        if kfn_norm:#nonzero
            k_fn = k_fn/ kfn_norm

            
        sumHat = np.sum(xd_hat_n*xd_n)
        if sumHat > 1 or sumHat < -1:
            sumHat = max(min(sumHat, 1), -1)
            warnings.warn(' cosinus out of bound!')
            
        k_ds[:,n] = np.arccos(sumHat)*k_fn.squeeze()
        
    xd_mags = np.sqrt(np.sum(xd_hat**2, axis=0) )

    # Weighted interpolation
    weightPow = 2 # Hyperparameter for several obstacles !!!!
    w = w**weightPow
    if not np.linalg.norm(w,2):
        warnings.warn('trivial weight.')
    w = w/np.linalg.norm(w,2)
    
    xd_mag = np.sum(xd_mags*w)
    k_d = np.sum(k_ds*np.tile(w, (d-1, 1)), axis=1)

    norm_kd = LA.norm(k_d)
    
    # Reverse k_d
    if norm_kd: #nonzero
        n_xd = Rf.T @ np.hstack((np.cos(norm_kd), np.sin(norm_kd)/norm_kd*k_d ))
    else:
        n_xd = Rf.T @  np.hstack((1, np.zeros((d-1)) ))
        
    xd = xd_mag*n_xd.squeeze()
    
    #if LA.norm(M*xd)>0.05:
    #    xd = LA.norm(xd)/LA.norm(M*xd)*M @xd #velocity modulation

    xd = xd + xd_obs # transforming back the velocity into the global coordinate system

    #if  (str(float(xd[0] )).lower() == 'nan' or
    #     str(float(xd[1] )).lower() == 'nan'):
        
    assert(not( str(float(xd[0] )).lower() == 'nan'))
    assert(not( str(float(xd[1] )).lower() == 'nan'))

    if d==2: # TODO for higher dimensions
        # Get the angle between the different systems and the tangent plane
        # TODO - adjust for moving obstacles
        
        e = E[:,1]
        e_magnitude = LA.norm(e)
        if e_magnitude: # nonzero value
            e = e/e_magnitude
        # Get the different angles
        
        angle_linear = np.copysign(np.cos(e_magnitude.T @ xd),
                                   np.sin(e_magnitude.T @ xd) )
        magXd = LA.norm(xd)
        if magXd:
            angle_linear = angle_linear/magXd
        
        angle_initial_linear = np.copysign(np.cos(e_magnitude.T @ f_linear),
                                            np.sin(e_magnitude.T @ f_linear) )
        mag_initLin = LA.norm(f_linear)
        if mag_initLin:
            angle_initial_linear = angle_initial_linear/mag_initLin
            
        angle_initial_nonlinear = np.copysign(np.cos(e_magnitude.T @ f_nonlinear),
                                            np.sin(e_magnitude.T @ f_nonlinear) )
        mag_initNonlin = LA.norm(f_nonlinear)
        if mag_initNonlin:
            angle_initial_nonlinear = angle_initial_nonlinear/mag_initNonlin

        if angle_initial_linear:
            angle_nonlinear = angle_linear/angle_initial_linear*angle_initial_nonlinear
        else:
            print('and what else!')
    
    return xd


def obs_avoidance_interpolation_moving(x, xd, obs, attractor='none'):
    # TODO -- speed up for multiple obstacles, not everything needs to be safed...

    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    if N_obs ==0:
        return xd
    
    d = x.shape[0]
    Gamma = np.zeros((N_obs))
    
    if type(attractor)==str:
        
        if attractor=='default': # Define attractor position
            attractor = np.zeros((d))
            N_attr = 1
    else:
        N_attr = 1 # TODO -- measure length in case of several attractors, use matrix
                

    # Linear and angular roation of velocity
    xd_dx_obs = np.zeros((d,N_obs))
    xd_w_obs = np.zeros((d,N_obs)) #velocity due to the rotation of the obstacle

    #M = np.zero((d, d, N_obs))
    E = np.zeros((d,d,N_obs))
    E_orth = np.zeros((d,d,N_obs))
    
    R = np.zeros((d,d,N_obs))

    for n in range(N_obs):
        # rotating the query point into the obstacle frame of reference
        if obs[n].th_r: # Greater than 0
            R[:,:,n] = compute_R(d,obs[n].th_r)
        else:
            R[:,:,n] = np.eye(d)

        # Move to obstacle centered frame
        x_t = R[:,:,n].T @ (x-obs[n].x0)
        
        E[:,:,n], Gamma[n], E_orth[:,:,n] = compute_basis_matrix( d,x_t,obs[n], R[:,:,n])
        
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
            warnings.warn('NOT implemented for d={}'.format(d))

        #the exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object
        xd_obs_n = w[n]*np.exp(-1/obs[n].sigma*(max([Gamma[n],1])-1))*  ( np.array(obs[n].xd) + xd_w )
        # Only consider velocity of the obstacle in direction
        
        #xd_obs_n = E_orth[:,:,n] @ np.array(( max(np.linalg.inv(E_orth[:,:,n])[0,:] @ xd_obs_n,0),np.zeros(d-1) ))
        xd_obs_n = np.linalg.inv(E_orth[:,:,n]) @ xd_obs_n
        xd_obs_n[0] = np.max(xd_obs_n[0], 0)
        #xd_obs_n[1:] = np.zeros(d-1)
        xd_obs_n = E_orth[:,:,n] @ xd_obs_n

        xd_obs = xd_obs + xd_obs_n

    xd = xd-xd_obs #computing the relative velocity with respect to the obstacle

    # Create orthogonal matrix
    #Ff = orthogonalBasisMatrix(xd)
    xd_norm = LA.norm(xd)
    if xd_norm:#nonzero
        xd_n = xd/xd_norm
    else:
        xd_n=xd

    xd_t = np.array([xd_n[1], -xd_n[0]])

    Ff = np.array([xd_n, xd_t])

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
            if not w[n] == 0:
                print('Gamma:', Gamma[n])
            D = w[n]*(np.hstack((-1,d0)))
        else:
            D = w[n]*(np.hstack((-1,d0))/abs(Gamma[n])**(1/rho))
        #     if isfield(obs[n],'tailEffect') && ~obs[n].tailEffect && xdT*R(:,:,n)*E(:,1,n)>=0 #the obstacle is already passed, no need to do anything
        #         D(1) = 0.0

            if (not obs[n].tail_effect) and xd.T @ (R[:,:,n] @ E_orth[:,0, n]) >=0:
                # the obstacle is already passed, no need to do anything
                D[0] = 0.0
                
        
        if D[0] < -1.0:
            D[1:] = d0
            if xd.T @ R[:,:,n] @ E[:,1,n] < 0:
                D[0] = -1.0
        
        M[:,:,n] = (R[:,:,n] @ E[:,:,n] @ np.diag(D+np.hstack((1,d0)) ) @ LA.pinv(E[:,:,n]) @ R[:,:,n].T)
        xd_hat[:,n] = M[:,:,n] @ xd #velocity modulation
        xd_mags[n] = np.sqrt(np.sum(xd_hat[:,n]**2))
        if xd_mags[n]: # Nonzero magnitude
            xd_hat_n = xd_hat[:,n]/xd_mags[n]
        else:
            xd_hat_n = xd_hat[:,n]
        
        if not d==2:
            warnings.warn('not implemented for d neq 2')

        Rfn = Rf @ xd_hat_n
        k_fn = Rfn[1:]
        kfn_norm = LA.norm(k_fn) # Normalize
        if kfn_norm:#nonzero
            k_fn = k_fn/ kfn_norm

        sumHat = np.sum(xd_hat_n*xd_n)
        if sumHat > 1 or sumHat < -1:
            sumHat = max(min(sumHat, 1), -1)
            warnings.warn(' cosinus out of bound!')
            
        k_ds[:,n] = np.arccos(sumHat)*k_fn.squeeze()
        
    xd_mags = np.sqrt(np.sum(xd_hat**2, axis=0) )

    if not type(attractor)==str:
        # Enforce convergence in the region of the attractor
        d_a = np.linalg.norm(x - np.array(attractor)) # Distance to attractor
        
        w = compute_weights(np.hstack((Gamma, [d_a])), N_obs+N_attr)

        k_ds = np.hstack((k_ds, np.zeros((d-1, N_attr)) )) # points at the origin

        xd_mags = np.hstack((xd_mags, np.linalg.norm((xd))*np.ones(N_attr) ))
        
    # Weighted interpolation
    weightPow = 2 # Hyperparameter for several obstacles !!!!
    w = w**weightPow
    if not np.linalg.norm(w,2):
        warnings.warn('trivial weight.')
    w = w/np.linalg.norm(w,2)
    
    xd_mag = np.sum(xd_mags*w)
    k_d = np.sum(k_ds*np.tile(w, (d-1, 1)), axis=1)

    norm_kd = LA.norm(k_d)
    
    # Reverse k_d
    if norm_kd: #nonzero
        n_xd = Rf.T @ np.hstack((np.cos(norm_kd), np.sin(norm_kd)/norm_kd*k_d ))
    else:
        n_xd = Rf.T @ np.hstack((1, k_d ))

    xd = xd_mag*n_xd.squeeze()
    
    #if LA.norm(M*xd)>0.05:
    #    xd = LA.norm(xd)/LA.norm(M*xd)*M @xd #velocity modulation

    xd = xd + xd_obs # transforming back the velocity into the global coordinate system

    #if  (str(float(xd[0] )).lower() == 'nan' or
    #     str(float(xd[1] )).lower() == 'nan'):
    assert(not( str(float(xd[0] )).lower() == 'nan'))
    assert(not( str(float(xd[1] )).lower() == 'nan'))

    return xd


def obs_avoidance_nonlinear_radial(x, ds_init, obs, attractor='none'):
    # (dt, x, obs, obs_avoidance=obs_avoidance_interpolation_moving, ds=linearAttractorConst, x0='default', k_f=0.75):
    # TODO -- speed up for multiple obstacles, not everything needs to be safed...

    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    if N_obs ==0:
        return ds_init(x)
    
    d = x.shape[0]
    Gamma = np.zeros((N_obs))
    
    if type(attractor)==str:
        if attractor=='default': # Define attractor position
            attractor = np.zeros((d,1))
            N_attr = 1
        else: # none
            N_attr=0
    else:
        attractor = np.array(attractor)
        if len(attractor.shape)==1:
            attractor = np.array(([attractor])).T

        N_attr = attractor.shape[1]
            
    # Linear and angular roation of velocity
    xd_dx_obs = np.zeros((d,N_obs))
    xd_w_obs = np.zeros((d,N_obs)) #velocity due to the rotation of the obstacle

    #M = np.zero((d, d, N_obs))
    E = np.zeros((d,d,N_obs))
    E_orth = np.zeros((d,d,N_obs))
    
    R = np.zeros((d,d,N_obs))
    x_t = np.zeros((d, N_obs))
    

    for n in range(N_obs):
        # rotating the query point into the obstacle frame of reference
        if obs[n].th_r: # Greater than 0
            R[:,:,n] = compute_R(d,obs[n].th_r)
        else:
            R[:,:,n] = np.eye(d)

        # Move to obstacle centered frame
        x_t[:,n] = R[:,:,n].T @ (x-obs[n].x0)
        
        E[:,:,n], Gamma[n], E_orth[:,:,n] = compute_basis_matrix( d,x_t[:,n],obs[n], R[:,:,n])

    Gamma_a = []
    for a in range(N_attr):
        # Eucledian distance -- other options possible
        Gamma_a = LA.norm(x-attractor[:,a])
        # if Gamma[n]<0.99: 
        #     print(Gamma[n])
    # The attractors are also included in the weight
    w = compute_weights(np.hstack((Gamma,Gamma_a)),N_obs+N_attr)

    # Loop to find new DS-evaluation point
    delta_x = np.zeros((d))
    for o in range(N_obs):
        distToCenter = LA.norm(x_t[:,o])
        if distToCenter > 0:
            directionX = -x_t[:,o]/ distToCenter
        else:
            print("warning -- collision with obstacle!")
            delta_x = np.zeros((d))
            break

        #import pdb; pdb.set_trace() ## DEBUG ##
        
        rad_obs = findRadius(obs[o], directionX)
        
        # r_obs = norm(x_obs);

        # %delta_r = r_obs*(1-1/Gamma);
        p =1;
        delta_r = rad_obs*(1/Gamma[o])**p;

        delta_x = delta_x + w[o]* (R[:,:,o] @ (delta_r*directionX))
        
        # % Calculate now center
        #x_hat = (r-delta_r)/r_x*x_t;

        # Move x_hat to original coordinate system
        #x_hat = R[:,:,n].T*x_hat + obs[o].x0;
    
    xd = ds_init(x+delta_x)

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
            warnings.warn('NOT implemented for d={}'.format(d))

        #the exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object
        xd_obs_n = w[n]*np.exp(-1/obs[n].sigma*(max([Gamma[n],1])-1))*  ( np.array(obs[n].xd) + xd_w )
        # Only consider velocity of the obstacle in direction
        
        #xd_obs_n = E_orth[:,:,n] @ np.array(( max(np.linalg.inv(E_orth[:,:,n])[0,:] @ xd_obs_n,0),np.zeros(d-1) ))
        xd_obs_n = np.linalg.inv(E_orth[:,:,n]) @ xd_obs_n
        xd_obs_n[0] = np.max(xd_obs_n[0], 0)
        #xd_obs_n[1:] = np.zeros(d-1)
        xd_obs_n = E_orth[:,:,n] @ xd_obs_n

        xd_obs = xd_obs + xd_obs_n

    xd = xd-xd_obs #computing the relative velocity with respect to the obstacle

    # Create orthogonal matrix
    #Ff = orthogonalBasisMatrix(xd)
    xd_norm = LA.norm(xd)
    if xd_norm:#nonzero
        xd_n = xd/xd_norm
    else:
        xd_n=xd

    xd_t = np.array([xd_n[1], -xd_n[0]])

    Ff = np.array([xd_n, xd_t])

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
            if not w[n] == 0:
                print('Gamma:', Gamma[n])
            D = w[n]*(np.hstack((-1,d0)))
        else:
            D = w[n]*(np.hstack((-1,d0))/abs(Gamma[n])**(1/rho))
        #     if isfield(obs[n],'tailEffect') && ~obs[n].tailEffect && xdT*R(:,:,n)*E(:,1,n)>=0 #the obstacle is already passed, no need to do anything
        #         D(1) = 0.0
        if (not obs[n].tail_effect) and D[0] < -1.0:
            D[1:] = d0
            if xd.T @ R[:,:,n] @ E[:,1,n] < 0:
                D[0] = -1.0
        
        M[:,:,n] = (R[:,:,n] @ E[:,:,n] @ np.diag(D+np.hstack((1,d0)) ) @ LA.pinv(E[:,:,n]) @ R[:,:,n].T)
        xd_hat[:,n] = M[:,:,n] @ xd #velocity modulation
        xd_mags[n] = np.sqrt(np.sum(xd_hat[:,n]**2))
        if xd_mags[n]: # Nonzero magnitude
            xd_hat_n = xd_hat[:,n]/xd_mags[n]
        else:
            xd_hat_n = xd_hat[:,n]
        
        if not d==2:
            warnings.warn('not implemented for d neq 2')

        Rfn = Rf @ xd_hat_n
        k_fn = Rfn[1:]
        kfn_norm = LA.norm(k_fn) # Normalize
        if kfn_norm:#nonzero
            k_fn = k_fn/ kfn_norm

        sumHat = np.sum(xd_hat_n*xd_n)
        if sumHat > 1 or sumHat < -1:
            sumHat = max(min(sumHat, 1), -1)
            warnings.warn(' cosinus out of bound!')
            
        k_ds[:,n] = np.arccos(sumHat)*k_fn.squeeze()
        
    xd_mags = np.sqrt(np.sum(xd_hat**2, axis=0) )

    if N_attr:
        # Enforce convergence in the region of the attractor
        #d_a = np.linalg.norm(x - np.array(attractor)) # Distance to attractor
        #w = compute_weights(np.hstack((Gamma, [d_a])), N_obs+N_attr)
        k_ds = np.hstack((k_ds, np.zeros((d-1, N_attr)) )) # points at the origin
        xd_mags = np.hstack((xd_mags, np.linalg.norm((xd))*np.ones(N_attr) ))
        
    # Weighted interpolation
    weightPow = 2 # Hyperparameter for several obstacles !!!!
    w = w**weightPow
    if not np.linalg.norm(w,2):
        warnings.warn('trivial weight.')
    w = w/np.linalg.norm(w,2)
    
    xd_mag = np.sum(xd_mags*w)
    k_d = np.sum(k_ds*np.tile(w, (d-1, 1)), axis=1)

    norm_kd = LA.norm(k_d)
    
    # Reverse k_d
    if norm_kd: #nonzero
        n_xd = Rf.T @ np.hstack((np.cos(norm_kd), np.sin(norm_kd)/norm_kd*k_d ))
    else:
        n_xd = Rf.T @ np.hstack((1, k_d ))

    xd = xd_mag*n_xd.squeeze()

    closestAttr = np.argmin( LA.norm(np.tile(x, (N_attr,1)).T - attractor, axis=0) )
    xd = constVelocity_distance(xd, x, x0=attractor[:,closestAttr],
                                velConst = 10.0, distSlow=0.1)

    #if LA.norm(M*xd)>0.05:
    #    xd = LA.norm(xd)/LA.norm(M*xd)*M @xd #velocity modulation

    xd = xd + xd_obs # transforming back the velocity into the global coordinate system

    #if  (str(float(pxd[0] )).lower() == 'nan' or
    #     str(float(xd[1] )).lower() == 'nan'):
    assert(not( str(float(xd[0] )).lower() == 'nan'))
    assert(not( str(float(xd[1] )).lower() == 'nan'))

    return xd



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


def obs_avoidance_nonlinear_derivative(x, ds_init, obs, attractor='none'):
    # (dt, x, obs, obs_avoidance=obs_avoidance_interpolation_moving, ds=linearAttractorConst, x0='default', k_f=0.75):
    # TODO -- speed up for multiple obstacles, not everything needs to be safed...

    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    if N_obs ==0:
        return ds_init(x)
    
    d = x.shape[0]
    Gamma = np.zeros((N_obs))
    
    if type(attractor)==str:
        if attractor=='default': # Define attractor position
            attractor = np.zeros((d,1))
            N_attr = 1
        else: # none
            N_attr=0
    else:
        attractor = np.array(attractor)
        if len(attractor.shape)==1:
            attractor = np.array(([attractor])).T

        N_attr = attractor.shape[1]
            
    # Linear and angular roation of velocity
    xd_dx_obs = np.zeros((d,N_obs))
    xd_w_obs = np.zeros((d,N_obs)) #velocity due to the rotation of the obstacle

    #M = np.zero((d, d, N_obs))
    E = np.zeros((d,d,N_obs))
    D = np.zeros((d,d,N_obs))
    E_orth = np.zeros((d,d,N_obs))
    
    R = np.zeros((d,d,N_obs))
    x_t = np.zeros((d, N_obs))
    
    for n in range(N_obs):
        # rotating the query point into the obstacle frame of reference
        if obs[n].th_r: # Greater than 0
            R[:,:,n] = compute_R(d,obs[n].th_r)
        else:
            R[:,:,n] = np.eye(d)

        # Move to obstacle centered frame
        x_t[:,n] = R[:,:,n].T @ (x-obs[n].x0)
        
        E[:,:,n], Gamma[n], E_orth[:,:,n] = compute_basis_matrix( d,x_t[:,n],obs[n], R[:,:,n])

    Gamma_a = []
    for a in range(N_attr): # Eucledian distance -- other options possible
        Gamma_a = LA.norm(x-attractor[:,a])
        
    # The attractors are also included in the weight
    w = compute_weights(np.hstack((Gamma,Gamma_a)),N_obs+N_attr)

    # Loop to find new DS-evaluation point
    delta_x = np.zeros((d))
    for o in range(N_obs):
        distToCenter = LA.norm(x_t[:,o])
        if distToCenter > 0:
            directionX = -x_t[:,o]/ distToCenter
        else:
            print("WARNING -- collision with obstacle!")
            delta_x = np.zeros((d))
            break
        
        rad_obs = findRadius(obs[o], directionX)
        
        p = 1
        delta_r = rad_obs*(1/Gamma[o])**p

        delta_x = delta_x + w[o]* (R[:,:,o] @ (delta_r*directionX))
        
        # % Calculate now center
        #x_hat = (r-delta_r)/r_x*x_t;

        # Move x_hat to original coordinate system
        #x_hat = R[:,:,n].T*x_hat + obs[o].x0;
    
    xd = ds_init(x+delta_x)

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
            warnings.warn('NOT implemented for d={}'.format(d))

        #the exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object
        xd_obs_n = w[n]*np.exp(-1/obs[n].sigma*(max([Gamma[n],1])-1))*  ( np.array(obs[n].xd) + xd_w )
        # Only consider velocity of the obstacle in direction
        
        #xd_obs_n = E_orth[:,:,n] @ np.array(( max(np.linalg.inv(E_orth[:,:,n])[0,:] @ xd_obs_n,0),np.zeros(d-1) ))
        xd_obs_n = np.linalg.inv(E_orth[:,:,n]) @ xd_obs_n
        xd_obs_n[0] = np.max(xd_obs_n[0], 0)
        #xd_obs_n[1:] = np.zeros(d-1)
        xd_obs_n = E_orth[:,:,n] @ xd_obs_n

        xd_obs = xd_obs + xd_obs_n

    xd = xd-xd_obs #computing the relative velocity with respect to the obstacle

    # Create orthogonal matrix
    #Ff = orthogonalBasisMatrix(xd)
    xd_norm = LA.norm(xd)
    if xd_norm:#nonzero
        xd_n = xd/xd_norm
    else:
        xd_n=xd

    xd_t = np.array([xd_n[1], -xd_n[0]])

    Ff = np.array([xd_n, xd_t])

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
            if not w[n] == 0:
                print('Gamma:', Gamma[n])
            D = w[n]*(np.hstack((-1,d0)))
        else:
            D = w[n]*(np.hstack((-1,d0))/abs(Gamma[n])**(1/rho))
        #     if isfield(obs[n],'tailEffect') && ~obs[n].tailEffect && xdT*R(:,:,n)*E(:,1,n)>=0 #the obstacle is already passed, no need to do anything
        #         D(1) = 0.0
        if (not obs[n].tail_effect) and D[0] < -1.0:
            D[1:] = d0
            if xd.T @ R[:,:,n] @ E[:,1,n] < 0:
                D[0] = -1.0

        D[:,:,n] = np.diag(D+np.hstack((1,d0)) )
        
        M[:,:,n] = (R[:,:,n] @ E[:,:,n] @ D[:,:n]  @ LA.pinv(E[:,:,n]) @ R[:,:,n].T)
        xd_hat[:,n] = M[:,:,n] @ xd #velocity modulation
        xd_mags[n] = np.sqrt(np.sum(xd_hat[:,n]**2))
        if xd_mags[n]: # Nonzero magnitude
            xd_hat_n = xd_hat[:,n]/xd_mags[n]
        else:
            xd_hat_n = xd_hat[:,n]
        
        if not d==2:
            warnings.warn('not implemented for d neq 2')

        Rfn = Rf @ xd_hat_n
        k_fn = Rfn[1:]
        kfn_norm = LA.norm(k_fn) # Normalize
        if kfn_norm:#nonzero
            k_fn = k_fn/ kfn_norm

        sumHat = np.sum(xd_hat_n*xd_n)
        if sumHat > 1 or sumHat < -1:
            sumHat = max(min(sumHat, 1), -1)
            warnings.warn(' cosinus out of bound!')
            
        k_ds[:,n] = np.arccos(sumHat)*k_fn.squeeze()
        
    xd_mags = np.sqrt(np.sum(xd_hat**2, axis=0) )

    if N_attr:
        # Enforce convergence in the region of the attractor
        #d_a = np.linalg.norm(x - np.array(attractor)) # Distance to attractor
        #w = compute_weights(np.hstack((Gamma, [d_a])), N_obs+N_attr)
        k_ds = np.hstack((k_ds, np.zeros((d-1, N_attr)) )) # points at the origin
        xd_mags = np.hstack((xd_mags, np.linalg.norm((xd))*np.ones(N_attr) ))
        
    # Weighted interpolation
    weightPow = 2 # Hyperparameter for several obstacles !!!!
    w = w**weightPow
    if not np.linalg.norm(w,2):
        warnings.warn('trivial weight.')
    w = w/np.linalg.norm(w,2)
    
    xd_mag = np.sum(xd_mags*w)
    k_d = np.sum(k_ds*np.tile(w, (d-1, 1)), axis=1)

    norm_kd = LA.norm(k_d)
    
    # Reverse k_d
    if norm_kd: #nonzero
        n_xd = Rf.T @ np.hstack((np.cos(norm_kd), np.sin(norm_kd)/norm_kd*k_d ))
    else:
        n_xd = Rf.T @ np.hstack((1, k_d ))

    xd = xd_mag*n_xd.squeeze()

    closestAttr = np.argmin( LA.norm(np.tile(x, (N_attr,1)).T - attractor, axis=0) )
    xd = constVelocity_distance(xd, x, x0=attractor[:,closestAttr],
                                velConst = 10.0, distSlow=0.1)

    #if LA.norm(M*xd)>0.05:
    #    xd = LA.norm(xd)/LA.norm(M*xd)*M @xd #velocity modulation

    xd = xd + xd_obs # transforming back the velocity into the global coordinate system

    #if  (str(float(pxd[0] )).lower() == 'nan' or
    #     str(float(xd[1] )).lower() == 'nan'):
    assert(not( str(float(xd[0] )).lower() == 'nan'))
    assert(not( str(float(xd[1] )).lower() == 'nan'))

    return xd

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
