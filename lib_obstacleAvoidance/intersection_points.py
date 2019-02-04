''''
@author Lukas Huber
@email lukas.huber@epfl.ch
@date 2018-10-12
'''

import numpy as np

from math import pi

import matplotlib.pyplot as plt # only for debugging
import warnings

import sys

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)
from class_obstacle import *


#def intersection_points(obs, intersection_obs, marg_dynCenter=1.3, N_distStep=3, resol_max=1000, N_resol = 16 ):
def intersection_points(obs):

    N_obs = len(obs)
    if N_obs<=1:  # only one or no obstacle
        return []
    
    # only implemented for 2d
    dim = np.array(obs[0].x0).shape[0]

    if dim>2:
        print('WARNING --- Dimensions bigger than 2!!!')
        return [] # TODO implement for d>2
        
    

    x_obs_sf = [] # points of obstacles surface
    phi_obs_sf = []
    for ii in range(N_obs):
        x, phi = draw_ellipsoid(obs[ii])
        x_obs_sf.append(x)
        phi_obs_sf.append(phi)

    N_points = x_obs_sf[0].shape[1]

    point_intersec = [] # intersection points for obstacles
    ang_intersec = [] # intersection points for obstacles
    obs_intersec = [] # intersection points for obstacles

    for o1 in range(N_obs):
        for o2 in range(o1+1, N_obs):
            
            # Do it both way round, to decrease numerical based mistakes
            Gamma1 = np.sum(((x_obs_sf[o1]-np.tile(obs[o2].x0, (N_points,1)).T) / np.tile(obs[o2].a, (N_points, 1)).T ) ** (2*np.tile(obs[o2].p,(N_points,1)).T ),axis=0)
            Gamma2 = np.sum(((x_obs_sf[o2]-np.tile(obs[o1].x0, (N_points,1)).T) / np.tile(obs[o1].a, (N_points, 1)).T ) ** (2*np.tile(obs[o1].p,(N_points,1)).T ),axis=0)
            
            if np.sum(Gamma1<1) + np.sum(Gamma2<1): # Nonzero
                if obs[o1].hirarchy < obs[o1].hirarchy:
                    obs_intersec.append([o1,o2]) # add obstacles to list
                    obs[o1].children.append(o2)
                    obs[o2].parent = o1
                elif obs[o1].hirarchy > obs[o1].hirarchy:
                    obs_intersec.append([o2,o1]) # add obstacles to list
                    obs[o2].children.append(o1)
                    obs[o1].parent = o2
                else:
                    print('WARNING -- obstacles of the same hirarchy are intersecting')    
                
                # find points where the intersection starts
                indInt = np.where(np.logical_xor(Gamma1<1, np.hstack((Gamma1[1:], Gamma1[0]))<1))[0]
                point_intersec.append(np.zeros((2, indInt.shape[0]))) # add obstacles to list
                ang_intersec.append(np.zeros((2, ceil(indInt.shape[0]/2)) ) ) # add obstacles to list
                
                for ii in range(indInt.shape[0]):
                    phi_temp = phi_obs_sf[o1]
                    ind_temp = indInt[ii]

                    # used to fill in ang_intersec
                    angle_iterator = -1
                    
                    # print('indTemp', ind_temp)
                    if ind_temp < N_points-1:
                        phiRange=[phi_temp[ind_temp], phi_temp[ind_temp+1]]
                    else:
                        phiRange=[phi_temp[ind_temp], phi_temp[0]]

                    for it_prec in range(2):
                        x_obs_temp, phi_temp = draw_ellipsoid(obs[o1], phiRange = phiRange)

                        Gamma1 = np.sum(((x_obs_temp-np.tile(obs[o2].x0, (N_points,1)).T) / np.tile(obs[o2].a, (N_points, 1)).T ) ** (2*np.tile(obs[o2].p,(N_points,1)).T ),axis=0)
                        ind_temp = np.where(np.logical_xor(Gamma1[:-1]<1, np.hstack((Gamma1[1:]))<1))[0]
                        ind_temp = ind_temp[0]
                        phiRange = [phi_temp[ind_temp], phi_temp[ind_temp+1] ]

                    point_intersec[-1][:,ii] = np.mean(np.vstack((x_obs_temp[:,ind_temp+1], x_obs_temp[:, ind_temp])).T, axis=1)
                    if Gamma1[ind_temp]<1: # point in obstacle
                        angle_iterator+=1
                        ang_intersec[-1][0,angle_iterator] = np.arctan2(point_intersec[-1][1,ii],
                                                                        point_intersec[-1][0,ii])
                    else:
                        ang_intersec[-1][1,angle_iterator] = np.arctan2(point_intersec[-1][1,ii],
                                                                        point_intersec[-1][0,ii])
    return obs_intersec, point_intersec

def init_ds_children(x, obs, angle_surfaceIntersection): # find parent --
    if np.array(x).shape[0] > 2: 
        # LIMITATION -- only 2D, possible but complex in higher dimensions
        print("WARNING -- not defined for d==", d)
        
    ang_x = np.arctan2()
    angle_iterator = 0
    for ii in range(angle_surfaceIntersection.shape[1]):
        dAngle_lower = angleSubtraction(ang_x, angle_surfaceIntersection[0,ii])
        ii_plus = np.mod(ii+1, angle_surfaceIntersection.shape[1])
        dAngle_higher = angleSubtraction(angle_surfaceIntersection[0,ii_plus], ang_x)
        if (dAngle_lower + dAngle_higher < 2*pi):
            break
    angle_iterator = ii

    dAngle_higher = angleSubtraction(angle_surfaceIntersection[1,angle_iterator], ang_x)
    
    if dAngle_lower+dAngle_higerh < 2*pi:
        dAngle_middle = 0.5*(dAngle_higher - dAngle_lower)
        ds_mod_1 = 0
        ds_mod_2 = 0
    else: #only modulate magnitude
        ds_mod_1 = 0
        ds_mod_2 = 0
    return 0

def angleSubtraction(ang1, ang2):
    dAng = ang1-ang2
    while dAng > 2*pi:
        dAng = dAng -2*pi
    while dAng < 0:
        dAng = dAng + 2*pi
    return dAng
    
def draw_ellipsoid(obs, N = 50, phiRange = [0.,0.]):
    if phiRange[0] == phiRange[1]:
        phiRange[1] = phiRange[0]+2*pi/(N+1)*N

    phi = np.linspace(phiRange[0], phiRange[1], N)
    
    surfacePoints = np.zeros((2,N))

    # TODO --- add p[0]/p[1] > 1
    #if p[0]>1 or p[1]>1:
    # print('WARNING -- implement for p>1')
        
    cosPhi = np.cos(phi)
    surfacePoints[0,:] = obs.a[0]*np.copysign(np.abs(cosPhi)**(1/obs.p[0]), cosPhi)
    
    sinPhi = np.sin(phi)
    surfacePoints[1,:] = obs.a[1]*np.copysign(np.abs(sinPhi)**(1/obs.p[1]), sinPhi)

    surfacePoints = np.array(obs.rotMatrix) @ surfacePoints
    
    surfacePoints = surfacePoints + np.tile(obs.x0 , (N, 1)).T

    
    return surfacePoints, phi  # TODO -- remvove the output of the angle here, but calculate automatically

obs = []

a=[1, 3]
p=[2, 2]
x0=[6,0]
th_r=90/180*pi
sf=1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

a=[0.3, 3]
p=[2, 2]
x0=[4,1]
th_r=0/180*pi
sf=1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

obs_intersec, points = intersection_points(obs)

if True:
    plt.figure()

    print('obs',obs[0])
    print('obs',obs[0].a)
    x_surf, ell = draw_ellipsoid(obs[0])
    plt.plot(x_surf[0,:], x_surf[1,:], 'r.')
    
    x_surf, ell = draw_ellipsoid(obs[1])
    plt.plot(x_surf[0,:], x_surf[1,:], 'b.')

    plt.plot(points[0][0,:], points[0][1,:],'gx')
    
    plt.axis('equal')
    plt.show()

print('')
print('finitio')
print('')

