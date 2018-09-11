# python3 -- set path

import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt

# 3D plot
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# draw a vector
#from matplotlib.patches import FancyArrowPatch
#from mpl_toolkits.mplot3d import proj3d

from math import pi

import sys

# Custom path
lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

from dynamicalSystem_lib import *

# Custom functions
def multiSin3D(relPos, r1=2, r2=1, r_z=1,  nSin=4, dPhi=0*pi/4, pp=4):
    # Get maximum radius
    ang_xy = np.arctan2(relPos[1], relPos[0])
    r_xy =  r1+r2*np.cos((dPhi+ang_xy)*nSin)

    # Assume all of them are ellipses
    ang_z = np.arctan2(relPos[2], np.sqrt(relPos[0]*relPos[0] + relPos[1]*relPos[1]) )
    
    r = (1/( (np.abs(np.sin(ang_z)/r_z ) )**pp +  ( np.abs( np.cos(ang_z)/r_xy))**pp ) )**(1/pp)
    
    # Definition ellipses
    #xy^2/a1^2 + z^2/a2^2 =1
    #(v*cos(zeta))^2 /a1^2 + (v*sin(zeta))^2/a2^2 =1
    #=> v = sqrt(a1^2*a2^2/((a1^2*(sin(zeta)**2)+ a2^2*(cos(zeta))**2 )))
    return r


def cross3D(relPos, r1=0.5, r2=1.5, r_z=1, rc=0.1,  nRay=4, dPhi=0*pi/4, pp=6):
    rc0 = rc

    if rc > (r2-r1)/4 or rc > r1/2:
        rc = np.max([(r2-r1)/4, r1/2])
        print('WARNING --- rc adapted.')

    # Get maximum radius
    ang_xy = np.arctan2(relPos[1], relPos[0])

    ang_xy = np.mod(ang_xy + dPhi + pi/nRay,  2*pi/nRay)
    ang_xy = np.abs(ang_xy  - pi/(nRay))
    
    #print(ang_xy)
    if ang_xy < np.arctan2(r1-rc, r2):
        r_xy = r2/np.cos(ang_xy)
    elif ang_xy < np.arctan2(r1, r2-rc):

        # Get angle between two
        p_edge = [r2-rc, r1-rc]
        p_dist = np.sqrt(p_edge[0]*p_edge[0]+p_edge[1]*p_edge[1])
        dPhi = np.arccos((p_edge[0]*np.cos(ang_xy) + p_edge[1]*np.sin(ang_xy))
                         /(p_dist) )
        r_xy = np.cos(dPhi)*p_dist + np.sqrt(rc*rc-(np.sin(dPhi)*p_dist)**2)
        
        #r_xy =0
    elif ang_xy < np.arctan2(r1, r1+rc):
        r_xy = r1/np.sin(ang_xy)
    else:
        # Get angle between two
        p_edge = [r1+rc, r1+rc]
        p_dist = np.sqrt(p_edge[0]*p_edge[0]+p_edge[1]*p_edge[1])
        dPhi = np.arccos((p_edge[0]*np.cos(ang_xy) + p_edge[1]*np.sin(ang_xy))
                         /(p_dist) )
        r_xy = np.cos(dPhi)*p_dist - np.sqrt(rc*rc-(np.sin(dPhi)*p_dist)**2)

    #r_xy =  r1+r2*np.cos((dPhi+ang_xy)*nSin)

    # Assume all of them are ellipses
    ang_z = np.arctan2(relPos[2], np.sqrt(relPos[0]*relPos[0] + relPos[1]*relPos[1]) )

    
    r = (1/( (np.abs(np.sin(ang_z)/r_z ) )**pp +  ( np.abs( np.cos(ang_z)/r_xy))**pp ) )**(1/pp)
    
    # Definition ellipses
    #xy^2/a1^2 + z^2/a2^2 =1
    #(v*cos(zeta))^2 /a1^2 + (v*sin(zeta))^2/a2^2 =1
    #=> v = sqrt(a1^2*a2^2/((a1^2*(sin(zeta)**2)+ a2^2*(cos(zeta))**2 )))
    return r


def differentiation_num(relPos, func, d_rel=1e-4):# TO FINISH
    d = np.array(relPos).shape[0]
    nv = np.zeros(d)

    dl = LA.norm(relPos)*d_rel
    for ii in range(d):
        dt_vec = np.zeros(d)
        dt_vec[ii] = dl/2
        nv[ii] = (LA.norm(relPos+dt_vec)/func(relPos+dt_vec)-
                  LA.norm(relPos-dt_vec)/func(relPos-dt_vec))/dl
        
    nv = nv/LA.norm(nv) # Normalize
    return nv

# def d_multSin(ang, r1=2, r2=1, nSin=4, dPhi=pi/6): # derivatvie with respect to angle
#     return -r2*nSin*np.sin((dPhi+ang)*nSin)


def obs_avoidance_interpolation(x, xd, obs):
    # TODO -- speed up for multiple obstacles, not everything needs to be safed...
    
    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    if N_obs ==0:
        return xd
    
    d = x.shape[0]
    Gamma = np.zeros((N_obs))

    # Linear and angular roation of velocity
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
    
    xd = xd + xd_obs # transforming back the velocity into the global coordinate system

    return xd


def compute_basis_matrix(d, x_t, obs):
    # For an arbitrary shape, the next two lines are used to find the shape segment
    th = np.arctan2(x_t[1],x_t[0])
    
    R_obs = obs.rad_func(x_t)
    #for ind in range(partitions)
    if hasattr(obs, 'sf'):
        R_obs = R_obs*sf
    elif hasattr(obs, 'sf_a'):
        R_obs = R_obs + sf_a

    Gamma = LA.norm(x_t)/R_obs

    E = np.zeros((d, d))

    # Calulate Reference Direction
    if hasattr(obs,'reference'):  
        E[:,0] = - (x_t - R.T @ (np.array(obs.reference) - np.array(obs.x0)) )
    else:
        E[:,0] = - x_t

    # Make diagonal to circle to improve behavior
    nv_hat = -x_t

    # Nv
    nv = differentiation_num(x_t, func)
        
    if d==2:
        #generating E, for a 2D model it simply is: E = [dx [-dx(2)dx(1)]]
        E[0,1] = nv[1]
        E[1,1] = -nv[0]
    elif d==3:
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


# ---------------------------------------------------------
# Start script
# ---------------------------------------------------------

print('\n\script analysis-metric start.\n\n')

# Choose obstacle function -- change for different obstacle shape
rFunc = multSin
drFunc = d_multSin

nResol = 50

#phi_xy, phi_z = np.mgrid[-pi/2:pi/2:nResol*1j, -pi+(2*pi/nResol):pi:nResol*1j]
phi_xy, phi_z = np.mgrid[-pi/2:pi/2:nResol*1j, -pi:pi:nResol*1j]
#phi_xy, phi_z = np.mgrid[-pi/2:pi/2:nResol*1j, -pi/2:pi/2:nResol*1j]
#phi_xy, phi_z = np.mgrid[-pi:pi:nResol*1j, -0:0:nResol*1j]

pos = np.dstack((np.cos(phi_z)*np.cos(phi_xy),
                 np.cos(phi_z)*np.sin(phi_xy),
                 np.sin(phi_z) ))

for ii in range(nResol):
    for jj in range(nResol):
        #pos[ii,jj,:] = multiSin3D(pos[ii,jj,:])*pos[ii,jj,:]
        pos[ii,jj,:] = cross3D(pos[ii,jj,:])*pos[ii,jj,:]


# Create figure    
fig3 = plt.figure(figsize=plt.figaspect(0.44)*1.0)
ax3 = fig3.add_subplot(111, projection='3d')

ax3.plot_wireframe(pos[:,:,0], pos[:,:,1], pos[:,:,2], color=[0.2,0.2,0.2])
#ax3.plot_surface(pos[:,:,0], pos[:,:,1], pos[:,:,2], color=[0.2,0.3,0.2], alpha=0.5, rstride=4, cstride=4)
        
        # ax3.plot_surface(obsGrid[:,0].reshape(obsVarShape),
                         # obsGrid[:,1].reshape(obsVarShape),
                         # obsGrid[:,2].reshape(obsVarShape),
                         # color=cols[oo], alpha=0.5, rstride=4, cstride=4)
