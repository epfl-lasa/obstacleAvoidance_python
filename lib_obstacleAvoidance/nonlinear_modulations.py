
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
from lib_modulation import *


import warnings


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

def obs_avoidance_nonlinear_hirarchy(x, ds_init, obs, attractor='none'):
    # (dt, x, obs, obs_avoidance=obs_avoidance_interpolation_moving, ds=linearAttractorConst, x0='default', k_f=0.75):
    # TODO -- speed up for multiple obstacles, not everything needs to be safed...


    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    if N_obs ==0:
        return ds_init(x)

    max_hirarchy = 0
    hirarchy_array = np.zeros(N_obs)
    for oo in range(N_obs):
        hirarchy_array[oo] = obs[oo].hirarchy
        if obs[oo].hirarchy>max_hirarchy:
            max_hirarchy = obs[oo].hirarchy
    
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

    # Radial displacement position with the first element being the original one
    m_x = np.zeros((d, max_hirarchy+2)) 
    m_x[:,0] = x
    # import pdb; pdb.set_trace() ## DEBUG ##

    # TODO - x_t only be computated later?
    for n in range(N_obs):
        # rotating the query point into the obstacle frame of reference
        if obs[n].th_r: # Greater than 0
            R[:,:,n] = compute_R(d,obs[n].th_r)
        else:
            R[:,:,n] = np.eye(d)

        # Move to obstacle centered frame
        # x_t[:,n] = R[:,:,n].T @ (x-obs[n].x0)
        
        # E[:,:,n], Gamma[n], E_orth[:,:,n] = compute_basis_matrix(d, x_t[:,n],obs[n], R[:,:,n])

    Gamma_a = []
    for a in range(N_attr):
        # Eucledian distance -- other options possible
        Gamma_a = LA.norm(x-attractor[:,a])
        # if Gamma[n]<0.99: 
        #     print(Gamma[n])

    # The attractors are included in the weight
    # TODO try relative weight
    weight = compute_weights(np.hstack((Gamma,Gamma_a)),N_obs+N_attr)
    # weight[np.hstack((ind_hh_low, np.ones(N_attr))), hh] = compute_weights(np.hstack((Gamma[ind_hh],Gamma_a)),np.sum(ind_hh)+N_attr)
    # weight = np.zeros(N_obs, max_hirarchy)

    for hh in range(max_hirarchy, -1, -1): # backward loop
        
        ind_hirarchy = (hirarchy_array==hh)
        ind_hirarchy_low = (hirarchy_array<=hh)
        # weight[np.hstack((ind_hh_low, np.ones(N_attr))), hh] = compute_weights(np.hstack((Gamma[ind_hh],Gamma_a)),np.sum(ind_hh)+N_attr)
        
        # Loop to find new DS-evaluation point
        delta_x = np.zeros((d,))
        for o in np.arange(N_obs)[ind_hirarchy]:
            x_t[:,o] = R[:,:,o].T @ (m_x[:,hh]-obs[o].x0)
        
            E[:,:,o], Gamma[o], E_orth[:,:,o] = compute_basis_matrix(d, x_t[:,o],obs[o], R[:,:,o])

            x_t_rel = (m_x[:,hh])
            distToCenter = LA.norm(x_t[:,o])
            if distToCenter > 0:
                directionX = -x_t[:,o]/ distToCenter
            else:
                print("warning -- collision with obstacle!")
                delta_x = np.zeros((d))
                break
        
            rad_obs = findRadius(obs[o], directionX)
        
            p =1 # hyperparameter
            delta_r = rad_obs*(1/Gamma[o])**p

            delta_x = delta_x + weight[o]* (R[:,:,o] @ (delta_r*directionX))

        m_x[:, hh+1] = x+delta_x # For each hirarchy level, there is one mean radial displacement
    
    xd = ds_init(m_x[:,-1])
        
    #adding the influence of the rotational and cartesian velocity of the
    #obstacle to the velocity of the robot
    xd_obs = np.zeros((d))

     # Relative velocity
    # TODO based on hirarchy level ? YES / NO ?
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
        xd_obs_n = weight[n]*np.exp(-1/obs[n].sigma*(max([Gamma[n],1])-1))*  ( np.array(obs[n].xd) + xd_w )
        # Only consider velocity of the obstacle in direction
        
        #xd_obs_n = E_orth[:,:,n] @ np.array(( max(np.linalg.inv(E_orth[:,:,n])[0,:] @ xd_obs_n,0),np.zeros(d-1) ))
        xd_obs_n = LA.inv(E_orth[:,:,n]) @ xd_obs_n
        xd_obs_n[0] = np.max(xd_obs_n[0], 0)
        #xd_obs_n[1:] = np.zeros(d-1)
        xd_obs_n = E_orth[:,:,n] @ xd_obs_n

        xd_obs = xd_obs + xd_obs_n

    # compute velocity of to the obstacle with respect to the obstacle frame of reference
    xd = xd-xd_obs 

    for hh in range(0, max_hirarchy, 1): # forward loop

        weight_hirarchy = np.zeros((N_obs + N_attr))
        ind_w = np.hstack((ind_hirarchy_low, np.ones(N_attr)))>0 # TODO convert to bool
        weight_hirarchy[ind_w] = weight[ind_w] / LA.norm(weight[ind_w])
        

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

        # for n in range(N_obs):
        for n in np.arange(N_obs)[ind_hirarchy]:
            if hasattr(obs[n], 'rho'):
                rho = obs[n].rho
            else:
                rho = 1

            d0 = np.ones((E.shape[1]-1))

            if Gamma[n]==0:
                if not weight_hirarchy[n] == 0:
                    print('Gamma:', Gamma[n])
                D = weight_hh[n]*(np.hstack((-1,d0)))
            else:
                D = weight_hirarchy[n]*(np.hstack((-1,d0))/abs(Gamma[n])**(1/rho))
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
        weightPow = 1 # Hyperparameter for several obstacles !!!!
        weight_hirarchy = weight_hirarchy**weightPow
        if not LA.norm(weight_hirarchy,2):
            warnings.warn('trivial weight.')
        weight_hirarchy = weight_hirarchy/LA.norm(weight_hirarchy,2)

        xd_mag = np.sum(xd_mags*weight_hirarchy)
        k_d = np.sum(k_ds*np.tile(weight_hirarchy, (d-1, 1)), axis=1)

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

