'''
Different python tool for obstacle avoidance

@author lukashuber
@date 2018-04-30
'''

import numpy as np
import warnings

from math import cos, sin


         
def compute_R(d, th_r):
    if th_r == 0:
        rotMatrix = np.eye(d)


    # rotating the query point into the obstacle frame of reference
    if d==2:
        rotMatrix = np.array([[cos(th_r), -sin(th_r)],
                              [sin(th_r),  cos(th_r)]])
    elif d==3:
        R_x = np.array([[1, 0, 0,],
                        [0, np.cos(th_r[0]), np.sin(th_r[0])],
                        [0, -np.sin(th_r[0]), np.cos(th_r[0])] ])

        R_y = np.array([[np.cos(th_r[1]), 0, -np.sin(th_r[1])],
                        [0, 1, 0],
                        [np.sin(th_r[1]), 0, np.cos(th_r[1])] ])

        R_z = np.array([[np.cos(th_r[2]), np.sin(th_r[2]), 0],
                        [-np.sin(th_r[2]), np.cos(th_r[2]), 0],
                        [ 0, 0, 1] ])

        rotMatrix = R_x.dot(R_y).dot(R_z)
    else:
        warnings.warn('rotation not yet defined in dimensions d > 3 !')
        rotMatrix = np.eye(d)

    return rotMatrix



def compute_weights(distMeas, N, distMeas_min=1, weightType='inverseGamma'):
    #UNTITLED5 Summary of this function goes here
    #   Detailed explanation goes here
        
    distMeas = np.array([max(distMeas[i]-distMeas_min,0) for i in range(N)])

    w = np.zeros(([1,N]))

    #Gamma(Gamma<1) = 1
    #Gamma = Gamma-1
    if weightType == 'inverseGamma':
        zeroInd = distMeas==0
        if np.sum(zeroInd): # one element equal to zero -- avoid null division
            if np.sum(zeroInd) > 1:
                warnings.warn('DS on two boundaries. Collision might not be avoided. \n')
            
            w = zeroInd*1
        else:
            w = 1/distMeas

            w = w/np.sum((w)) # Normalization

    elif weightType == 'khansari':
       for i in range(N):
            ind = [i for i in range(N)]
            ind[i] = []
            w[i] = prod(distMeas(ind)/(distMeas(i)+distMeas(ind)))


    elif weightType == 'khansari_normalized':
       for i in range(N):
           ind = [i for i in range(N)]
           ind[i] = []
           w[i] = prod(distMeas(ind)/(distMeas(i)+distMeas(ind)))

       # Add normalization -- not in original
       w = w/sum[w]

    else:
        warnings.warn("Unkown weighting method.")

    #print('w', w)

    return w



def obs_check_collision_2d(obs_list, XX, YY):
    # No obstacles
    if len(obs_list) == 0:
        return

    dim = 2 

    dim_points = XX.shape
    if len(dim_points)==1:
        N_points = dim_points[0]
    else:
        N_points = dim_points[0]*dim_points[1]
    
    points = np.array(([np.reshape(XX,(N_points,)) , np.reshape(YY, (N_points,)) ] ))
    # At the moment only implemented for 2D
    collision = np.zeros( dim_points )

    N_points = points.shape[1]

    noColl = np.ones((1,N_points))

    for it_obs in range(len(obs_list)):
        # \Gamma = \sum_{i=1}^d (xt_i/a_i)^(2p_i) = 1
        R = compute_R(d,obs_list[it_obs].th_r)

        Gamma = sum( ( 1/obs_list[it_obs].sf * R.T @ (points - np.tile(np.array([obs_list[it_obs].x0]).T,(1,N_points) ) ) / np.tile(np.array([obs_list[it_obs].a]).T, (1, N_points)) )**(np.tile(2*np.array([obs_list[it_obs].p]).T, (1,N_points)) ) )

        noColl = (noColl* Gamma>1)

    return np.reshape(noColl, dim_points)


def obs_check_collision(points, obs_list=[]):
    # No obstacles
    if len(obs_list) == 0:
        return

    dim = points.shape[0]
    N_points = points.shape[1]

    # At the moment only implemented for 2D
    collision = np.zeros((N_points))

    noColl = np.ones((1,N_points))

    for it_obs in range(len(obs_list)):
        # \Gamma = \sum_{i=1}^d (xt_i/a_i)^(2p_i) = 1
        R = compute_R(dim,obs_list[it_obs].th_r)

        Gamma = sum( ( 1/obs_list[it_obs].sf * R.T @ (points - np.tile(np.array([obs_list[it_obs].x0]).T,(1,N_points) ) ) / np.tile(np.array([obs_list[it_obs].a]).T, (1, N_points)) )**(np.tile(2*np.array([obs_list[it_obs].p]).T, (1,N_points)) ) )

        noColl = (noColl* Gamma>1)

    return noColl
