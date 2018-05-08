import numpy as np
from numpy import linalg as LA

from math import ceil, sin, cos, sqrt

def obs_common_section( obs):
    #OBS_COMMON_SECTION finds common section of two ore more obstacles 
    # at the moment only solution in two d is implemented

    # Intersction surface
    intersection_obs = []
    intersection_sf = []
    intersection_sf_temp = []
    it_intersect = 0

    N_obs = len(obs)
    # No intersection region 
    if(N_obs == 1):
        return

    # Ext for more dimensions
    d = 2

    N_points = 12 # Choose number of points each iteration
    Gamma_steps = 5 # Increases computational cost

    # figure(11)
    # clf('reset')
    # for ii = 1:size(x_obs_sf,3)
    #     plot(x_obs_sf(1,:,ii),x_obs_sf(2,:,ii),'b.') hold on

    rotMat = np.zeros((d,d, N_obs))
    
    for it_obs in range(N_obs):
        rotMat[:,:,it_obs] = np.array(( obs[it_obs].rotMatrix ))

    for it_obs1 in range(N_obs):
        intersection_with_obs1 = False

        # Check if current obstacle 'it_obs1' has already an intersection with another
        # obstacle 
        memberFound = False
        for ii in range(len(intersection_obs)):
            if it_obs1 in intersection_obs[ii]:
                memberFound=True
                continue

        if memberFound:
            continue 

        for it_obs2 in range(it_obs1+1,N_obs):
            # Check if obstacle has already an intersection with another
            # obstacle 
            memberFound=False
            for ii in range(len(intersection_obs)):
                if it_obs2 in intersection_obs[ii]:
                    memberFound=True 
                    continue

            if memberFound: continue 

            if intersection_with_obs1:# Modify intersecition part
                obsCloseBy = True

                # Roughly check dimensions before starting expensive calculation
                for ii in intersection_obs:
                    if it_intersect[ii]:
                        if LA.norm(obs[ii].x0 - obs[it_obs2].x0) > max(obs[ii].a) + max(obs[it_obs2].a):
                            # Obstacles to far apart
                            obsCloseBy = False     
                            break

                if obsCloseBy:
                    N_inter = intersection_sf[it_intersect].shape[0] # Number of intersection points

                    # R = compute_R(d,obs[it_obs2].th_r)
                    Gamma_temp = ( (intersection_sf[it_intersect]-np.tile(obs[it_obs2].x0,(1,N_inter) ) )/ np.tile(obs[it_obs2].a,(N_inter)) ) ** (2*obs[it_obs2].p)
                    Gamma = [sum(1/obs[it_obs2].sf*rotMat[:,:,it_obs2].T @ Gamma_temp[ii,:]) for ii in range(Gamma_temp.shape[0])]
                    ind = [(Gamma[ii]<1)**ii for i in range(len(Gamma)) ]
                    if sum(ind):
                        intersection_sf[it_intersect] = intersection_sf[it_intersect][:,ind]
                        intersection_obs[it_intersect] = intersection_obs[it_intersect] + [it_obs2]
            else:
                # Roughly check dimensions before starting expensive calculation
                if sqrt(sum([obs[it_obs1].x0[ii]-obs[it_obs2].x0[ii] for ii in range(d)])**2) < max(obs[it_obs1].a) + max(obs[it_obs2].a):
                    # Obstacles are close enough

                    # get all points of obs2 in obs1
                    # R = compute_R(d,obs[it_obs1].th_r)
                    # \Gamma = \sum_[i=1]^d (xt_i/a_i)^(2p_i) = 1
                    N_points = len(obs[it_obs1].x_obs_sf[0])
                    Gamma_temp = ( (np.array(obs[it_obs2].x_obs_sf)-np.tile(obs[it_obs1].x0,(N_points,1)).T ) / np.tile(obs[it_obs1].a, (N_points,1)).T ) ** (2*np.tile(obs[it_obs1].p, (N_points,1)).T)
                    Gamma = [sum(1/obs[it_obs1].sf *  rotMat[:,:,it_obs1].T @ Gamma_temp[:,i]) for i in range(Gamma_temp.shape[0])]
                    ind_gamma = [(Gamma[i]<1)*i for i in range(len(Gamma))]

                    
                    intersection_sf_temp = np.array(obs[it_obs2].x_obs_sf[:,ind_gamma])
                    # Get all poinst of obs1 in obs2
                    #                 R = compute_R(d,obs[it_obs2].th_r)
                    Gamma_temp = ( (np.array(obs[it_obs1].x_obs_sf)-np.tile(obs[it_obs2].x0,(N_points,1)).T ) / np.tile(obs[it_obs2].a, (N_points,1)).T ) ** (2*np.tile(obs[it_obs2].p, (N_points,1)).T)
                    Gamma = [sum(1/obs[it_obs2].sf *  rotMat[:,:,it_obs2].T @ Gamma_temp[:,i]) for i in range(Gamma_temp.shape[0])]
                    ind_gamma = [(Gamma[i]<1)*i for i in range(len(Gamma))]

                    # Gamma_temp = ( (np.array(x_obs_sf[it_obs1][:,:])-np.tile(obs[it_obs2].x0,(1,N_points)) ) /np.tile(obs[it_obs2].a, (1, N_points) ) )**(2*obs[it_obs2].p)
                    # Gamma = [sum( 1/obs[it_obs2].sf*rotMatr[:,:,it_obs2].T @ Gamma_temp[:,ii]) for ii in range(Gamma_temp.shape[0])]
                    #ind = [(Gamma[ii]<1) for ii in range(Gamma.shape[0])]
                    intersection_sf_temp = np.hstack((intersection_sf_temp, obs[it_obs1].x_obs_sf[:,ind_gamma] ))

                    if intersection_sf_temp.shape[0] >0:
                        it_intersect = it_intersect + 1
                        intersection_with_obs1 = True
                        intersection_sf.append(intersection_sf_temp)
                        intersection_obs.append([it_obs1,it_obs2])

                        # Increase resolution by sampling points within
                        # obstacle, too
                        for ii in range(1,Gamma_steps):
                            N_points_interior = ceil(N_points/Gamma_steps*ii)

                            # obstaacles of 2 in 1
                            for kk in range(2):
                                if kk == 0:
                                    it_obs1_ = it_obs1
                                    it_obs2_ = it_obs2
                                    
                                elif kk ==1: # Do it both ways
                                    it_obs1_ = it_obs2
                                    it_obs2_ = it_obs1

                                # x_obs_sf_interior = np.zeros(N_points_interior, dim, 2) # for two obstacles
                                print('a_temp_outside', np.array(obs[it_obs1_].a)/Gamma_steps*ii)
                                x_obs_sf_interior = obs[it_obs1_].draw_ellipsoid(numPoints=N_points_interior, a_temp = np.array(obs[it_obs1_].a)/Gamma_steps*ii)

                                Gamma_temp = ( (x_obs_sf_interior-np.tile(obs[it_obs2_].x0,(N_points_interior,1)).T ) / np.tile(obs[it_obs2_].a, (N_points_interior,1)).T ) ** (2*np.tile(obs[it_obs2_].p, (N_points_interior,1)).T)
                                Gamma = [sum(1/obs[it_obs2_].sf *  rotMat[:,:,it_obs2_].T @ Gamma_temp[:,i]) for i in range(Gamma_temp.shape[0])]
                                ind_gamma = [(Gamma[i]<1)*i for i in range(len(Gamma))]
                                intersection_sf[it_intersect] = np.hstack((intersection_sf[it_intersect],x_obs_sf_interior[:,ind_gamma,kk] ))
                                # plot(intersection_sf[it_intersect](1,:), intersection_sf[it_intersect](2,:), 'ro')
                                # Check center point
                                if 1 > sum( (1/obs[it_obs2_].sf*rotMat[:,:,it_obs2_].T @ (obs[it_obs1_].x0-obs[it_obs2_].x0)/ obs[it_obs2_].a) ** (2*obs[it_obs2_].p),1):
                                    intersection_sf[it_intersect] = [intersection_sf[it_intersect],obs[it_obs1_].x0]

    #if intersection_with_obs1 continue 
    if len(intersection_sf)==0:
        return 

    for ii in range(len(intersection_obs)):
    #     plot(intersection_sf[ii](1,:),intersection_sf[ii](2,:),'x')
        intersection_sf[ii] = np.unique(intersection_sf[ii], axis=0)

        # Get numerical mean
        x_center_dyn= np.mean(intersection_sf[ii], 2)
        for it_obs in intersection_obs[ii]:
            obs[it_obs].x_center_dyn = x_center_dyn

        # sort points according to angle
    #     intersec_sf_cent = intersection_sf - repmat(x_center_dyn,1,size(intersection_sf,2))


        # TODO - replace atan2 for speed
    #     [~, ind] = sort( atan2(intersec_sf_cent(2,:), intersec_sf_cent(1,:)))

    #     intersection_sf = intersection_sf(:, ind)
    #     intersection_sf = [intersection_sf, intersection_sf(:,1)]

    #     intersection_obs = [1:size(obs,2)]

    return obs, intersection_obs 
