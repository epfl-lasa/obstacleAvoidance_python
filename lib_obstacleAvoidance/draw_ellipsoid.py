import numpy as np
from math import pi, cos, sin, copysign

import warnings

import sys 
lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm_python/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)
    
from lib_obstacleAvoidance import compute_R

    
def obs_draw_ellipsoid(obs,ns, nargout=1):
     # Add description -- see MAT LAB
    d = len(obs[0].x0)
    
    if d > 3 or d < 2:
        print('Error: obs_draw_ellipsoid only supports 2D or 3D obstacle')
        x_obs = []
        x_obs_sf = []
        return

    if d == 2:
        theta = np.linspace(-pi,pi, num=ns)
        numPoints = ns
    else:
        theta, phi = np.meshgrid(np.linspace(-pi,pi, num=ns[0]),np.linspace(-pi/2,pi/2,num=ns[1]) ) #
        numPoints = ns[0]*ns[1]
        theta = theta.T
        phi = phi.T

    N = len(obs) #number of obstacles
    x_obs = np.zeros((d,numPoints,N))

    if nargout > 1:
        x_obs_sf = np.zeros((d,numPoints,N))

    for n in range(N):
        # clear ind -- TODO what is this line
        # rotating the query point into the obstacle frame of reference
        R = compute_R(d,obs[n].th_r)
            #if obs[n].th_r.shape[0] == d and obs[n].th_r.shape[1] == d:

        # For an arbitrary shap, the next two lines are used to find the shape segment
        if hasattr(obs[n],'partition'):
            warnings.warn('Warning - partition no finished implementing')
            for i in range(obs[n].partition.shape[0]):
                ind[i,:] = theta>=(obs[n].partition[i,1]) & theta<=(obs[n].partition[i,1])
            [i, ind]=max(ind)
        else:
            ind = 0
       
        #a = obs[n].a[:,ind]
        #p = obs[n].p[:,ind]

        # TODO -- add partition index
        a = obs[n].a[:]
        p = obs[n].p[:]

        if d == 2:
            x_obs[0,:,n] = a[0]*np.cos(theta)
            #import pdb; pdb.set_trace() ## DEBUG ##
            
            x_obs[1,:,n] = np.copysign(a[1], theta)*(1 - np.cos(theta)**(2*p[0]))**(1./(2.*p[1]))
        else:
            x_obs[0,:,n] = a[0,:]*np.cos(phi)*np.cos(theta)
            warnings.warn('Not implemented yet...')
            # TODO --- next line probs wrong. Power of zero...
            #x_obs[1,:,n] = copysign(a(1,:), (theta))*cos(phi)*(1 - 0.^(2.*p(3,:)) - cos(theta)**(2*p(0,:))).^(1./(2.*p(2,:))) 
            #x_obs[2,:,n] = a[3,:]*sign(phi).*(1 - (sign(theta).*cos(phi).*(1 - 0.^(2.*p(3,:)) - cos(theta).^(2.*p(1,:))).^(1./(2.*p(2,:)))).^(2.*p(2,:)) - (cos(phi).*cos(theta)).^(2.*p(1,:))).^(1./(2.*p(3,:)))

        if nargout > 1:
            if hasattr(obs[n], 'sf'):
                if obs[n].sf.shape[0] == 1:
                    x_obs_sf[:,:,n] = R@(x_obs[:,:,n]*obs[n].sf) + np.tile(obs[n].x0,(1,np))
                else:
                    x_obs_sf[:,:,n] = R@(x_obs[:,:,n]*np.tile(obs[n].sf,(1,np))) + np.tile(obs[n].x0, (1,numPoints) )
            else:
                x_obs_sf[:,:,n] = R @ x_obs[:,:,n] + np.tile(obs[n].x0,(1,numPoints))
                
        x_obs[:,:,n] = R@x_obs[:,:,n] + np.tile(np.array([obs[n].x0]).T,(1,numPoints))
        
    if nargout > 1:
        return x_obs, x_obs_sf
    else:
        return  x_obs

        # Introduce Concave Objects
        # if 'concaveAngle' in obs[n]:
        #     x_obs(:,:,n) = ellipsFold(x_obs(:,:,n),obs{n}.x0, obs{n}.concaveAngle)
        #     x_obs_sf(:,:,n) = ellipsFold(x_obs_sf(:,:,n),obs{n}.x0, obs{n}.concaveAngle)
        

