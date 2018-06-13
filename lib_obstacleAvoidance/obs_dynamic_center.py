import numpy as np

from math import pi

import matplotlib.pyplot as plt # only for debugging
import warnings

def dynamic_center(obs, intersection_obs, marg_dynCenter=1.3, N_distStep=3, resol_max=1000, N_resol = 16 ):
    # Calculate the dynamic center for the applicaiton of the modulation
    # matrix. A smooth transition is desired between objects far away (center
    # in middle) and to convex obstacles intersecting. 
    #
    # This might not always be the completely optimal position of the
    # dynamic_center of the obstacles, but it results in a continous movement
    # of it and keeps computational cost low
    #
    # TODO --- 
    # Increase sampling, when less than e.g. 1/9 of original points are still 
    # available decrease inital number of sampling points to 27 initially
    # do 3x -> sampling accuracy of 3^3*3^2*3^2 = 3^7 = 2187

    N_obs = len(obs)
    if N_obs<=1:  # only one or no obstacle
        return []

    # only implemented for 2d
    dim = obs[0].d

    if(dim>2):
        warnings.warn('Not implemented for higher order than d=2! \n')
        return []

    # Resolution of outside plot
    # MAYBE - Change to user fixed size -- replot first oneh
    x_obs_sf = []
    for ii in range(N_obs):
        x_obs_sf.append(np.array(obs[ii].x_obs_sf))
        
    #N_resol = x_obs_sf[0].shape[1] 


    # Center normalized vector to surface points
    rotMatrices = []
    
    for ii in range(N_obs):
        rotMatrices.append(np.array(obs[ii].rotMatrix))

    # Calculate distance between obstacles
    weight_obs_temp = np.zeros((N_obs,N_obs))

    #x_middle = zeros(dim, N_obs, N_obs)
    x_cyn_temp = np.zeros((dim, N_obs, N_obs)) 

    # TODO - remove after debugging
    #plt.figure(12) #, 'Position', [300,600,400,400])
    # plt.ion()
    # plt.show()
    # plt.plot(obs[0].x_obs_sf[0][:],obs[0].x_obs_sf[1][:],'k-')
    # plt.plot(obs[1].x_obs_sf[0][:],obs[1].x_obs_sf[1][:],'k-')
    # plt.axis('equal')

    intersection_temp = []
    for ii_list in intersection_obs:
        for jj in ii_list:
            if jj not in intersection_temp:
                intersection_temp.append(jj)
    intersection_obs = intersection_temp
        
    for ii in range(N_obs):# Center around this obstacle
        rotMat = rotMatrices[ii] # Rotation Matrix

        # Check if intersection already exists
        # TODO - make shorter (one liner)

        if ii in intersection_obs:
            continue
        
        # only iterate over half the obstacles
        for jj in range(ii+1,N_obs):             
            # Check if intersection already exists
            if jj in intersection_obs:
                continue

            # For ellipses:
            #ref_dist = marg_dynCenter*0.5*sqrt(0.25*(sqrt(sum(obs[ii].a)^2+max(obs[jj].a)^2))
            dist_contact = 0.5*(np.sqrt(np.sum(np.array(obs[ii].a)**2))) + np.sqrt(np.sum(np.array(obs[jj].a)**2))
            ref_dist = dist_contact*marg_dynCenter

            #ref_dist(ii,jj) = ref_dist(jj,ii) # symmetric

            # Inside consideration region -- are obstacles close to each other
            # TODO - second power. Does another one work to?! it should...
            ind = np.sum( (x_obs_sf[jj]-np.tile(obs[ii].x0,(x_obs_sf[jj].shape[1],1)).T )**2, axis=0) < ref_dist**2
            
            if not sum(ind):
                delta_dist = ref_dist + 1 # greater than reference distance
                continue # Obstacle too far away
    #        else
    #             fprintf('entering looop --- TODO remove \n')

            # Set increment step
            step_dist = (ref_dist)/(N_distStep)
            dist_start = 0

            #resol = x_obs_sf[0].shape[1] # usually first point is double..
            resol = N_resol # total resolution

            thetaRange = [0, 2*pi] # Start and point of ellipse arc
            a_range_old = [0,0] # Radius with which the ellipse arc was drawn

            itCount = 0 # Iteration counter
            itMax = 100 # Maximum number of iterations

            # Tries to find the distance to ellipse
            while(resol < resol_max):
                # Continue while small resolution
                for it_gamma in range(N_distStep):
                    delta_dist = (step_dist*(it_gamma+1))+dist_start # Renew distance

                    ind_intersec, n_intersec, x_obs_temp, a_range_old = check_for_intersection(obs[jj], obs[ii], delta_dist, N_resol, thetaRange, a_range_old, rotMatrices[ii])

                    # plt.plot(x_obs_temp[0,:], x_obs_temp[1,:], '--')
                    # plt.plot(x_obs_temp[0,ind_intersec], x_obs_temp[1,ind_intersec], 'k.')
                    # plt.show()
                    
                    # Increment iteratoin counter
                    itCount += 1

                    if n_intersec:
                        # Intersection found
                        dist_start = (delta_dist-step_dist) # Reset start position
                        step_dist = step_dist/(N_distStep) # Increase distance resolution

                        # Increase resolution of outline - plot
                        if n_intersec < len(ind_intersec)-2:
                            # all intersection in middle -  [ 0 0 1 1 1 0 ] 
                            # extrema 1 - [ 0 0 0 1 1 1 ]
                            # extrema 2 - [ 1 1 1 0 0 0 ]
                            ind_pos = ind_intersec.nonzero()[0]

                            indLow = ind_pos[0]
                            indHigh = ind_pos[-1]
                                
                            #indLow = ind_intersec,1, 'first')
                            #indHigh = find(ind_intersec,1,'last')

                            if resol > N_resol:
                                # only arc of convex obstacle is observed
                                indLow = max([indLow-1, 0]) 
                                indHigh= min([indHigh+1, N_resol-1])
                            else:

                                if indHigh == ind_intersec.shape[0] and indLow == 1:
                                    # split at border - [ 1 1 0 0 0 1 ] -- only
                                    # relevant when analysing original convex
                                    # obstacle
                                    #indHigh = find(not ind_intersec,1, 'first') - 1
                                    #indLow = find(not ind_intersec,1,'last')  + 1
                                    ind_pos = (not ind_intersec).nonzero()[0]
                                    
                                    indLow = ind_pos[-1] + 1
                                    indHigh = ind_pos[0] - 1

                                # Increse resolution of obstacle
                                n_intersec += 2 # Add one point to left, one to the right

                                #indLow = find(intersection_ind_temp,1)-1
                                indLow -= 1
                                if indLow < 0:
                                    indLow = N_resol-1

                                #indHigh = find(intersection_ind_temp,1,'last')+1
                                indHigh += 1
                                if indHigh >= N_resol:
                                    indHigh = 0  

                            xRange =  np.vstack((x_obs_temp[:,indLow], x_obs_temp[:,indHigh] )).T
                            # TODO - remove after debugging
    #                         plot( xRange(1,:), xRange(2,:),'g --')

                        else: # too few intersections
                            xRange = np.vstack((x_obs_temp[:,0], x_obs_temp[:,-1] )).T # keep same range
                    
                        x_start = rotMatrices[jj].T @ (xRange[:,0] - obs[jj].x0)
                        x_ = rotMatrices[jj].T @ (xRange[:,1] - obs[jj].x0) # TODO - replace with mor usefull name
                        x_Arc= rotMatrices[jj].T @ (x_obs_temp - np.tile( obs[jj].x0, (x_obs_temp.shape[1],1)).T)

                        # TODO remove these
                        if max([abs(x_[0]), abs(x_start[0])]) > abs(a_range_old[1]):
                            warnings.warn('Numeric apprximation of intersection finder could have a complex angle values.')

                            thetaRange = [np.copysign(1,x_start[1])*np.arccos(min([1,max([-1, (x_start[0]/a_range_old[0]) ]) ]) ), np.copysign(1,x_[1])*np.arccos(min([1,max([-1, (x_[0]/a_range_old[0])]) ]) )]
                            if sum([thetaRange[i].imag for i in range(len(thetaRange)) ]):
                                warnings.warn('Stayed complex!')
                        else:
                            thetaRange = [np.copysign(1,x_start[1])*np.arccos(x_start[0]/a_range_old[0]), np.copysign(1,x_[1])*np.arccos(x_[0]/a_range_old[0])]

                        # Resolution of the surface mapping - 2D
                        resol = resol/(n_intersec-1)*N_resol

                        temp, n_intersec, x_obs_temp, a_range_old = check_for_intersection(obs[jj], obs[ii],
                                                                                           dist_start, N_resol, thetaRange, a_range_old, rotMatrices[ii])

                        while (n_intersec > 0):
                            # The increasing resolution caused new points,
                            # lower value of dist_0 is not bounding anymore
                            dist_start = dist_start - step_dist

                            temp, n_intersec, x_obs_temp, a_range_old = check_for_intersection(obs[jj], obs[ii], dist_start, N_resol, thetaRange, a_range_old, rotMatrices[ii])
                            
                        break #current for loop after reset resolution dist_start

                if itCount > itMax: # Resolution max not reached in time
                    warnings.warn('No close intersection found ...\n')
                    break # Emergency exiting -- in case of slow convergence

                if delta_dist >= ref_dist:
                    break

            # Negative step
            if(delta_dist == 0): # Obstacles are touching: weight is only assigned to one obstacle
                weight_obs_temp[ii,jj] = -1
            elif delta_dist >= ref_dist: # Obstacle is far away
                weight_obs_temp[ii,jj] = 0
                continue
            else:
                weight_obs_temp[ii,jj] = max(1/delta_dist -1/(ref_dist-dist_contact),0) # if too far away/
            weight_obs_temp[jj,ii] = weight_obs_temp[ii,jj]

            # Position the middle of shortest line connecting both obstacles
            x_middle = np.mean(x_obs_temp[:,ind_intersec],axis=1)
            
            plt.plot(x_middle[0],x_middle[1],'ro')

            # Desired Gamma in (0,1) to be on obstacle
            #Gamma_dynCenter = max(1-delta_dist/(ref_dist-dist_contact),realmin) # TODO REMOVE
            Gamma_dynCenter = max([1-delta_dist/(ref_dist-dist_contact),0])

            # Desired position of dynamic_center if only one obstacle existed
            Gamma_intersec = np.sum( ( (rotMatrices[ii].T @ (x_middle-np.array(obs[ii].x0) ) ) / (np.array(obs[ii].sf)*np.array(obs[ii].a) )  )**(2*np.array(obs[ii].p) ), axis=0)
            
            x_cyn_temp[:,ii,jj] = rotMatrices[ii] @ (rotMatrices[ii].T @ (x_middle - np.array(obs[ii].x0) ) *( np.tile(Gamma_dynCenter/Gamma_intersec, (dim) )  )**(1./(2*np.array(obs[ii].p) ) ) )
            
            # Desired pogsition if only one obstacle exists 
            Gamma_intersec = np.sum( ( (rotMatrices[jj].T @ (x_middle-np.array(obs[jj].x0) ) ) / (np.array(obs[jj].sf)*np.array(obs[jj].a) )  ) **(2*np.array(obs[jj].p) ), axis=0)

            #x_cyn_temp[:,jj,ii] = rotMatrices[jj] @ ( rotMatrices[jj].T @ (x_middle - np.array(obs[jj].x0) ) * np.tile( (Gamma_dynCenter/Gamma_intersec),(dim)) ) **(1./(2*np.array(obs[jj].p)) )) # TODO rotation matrix?
            x_cyn_temp[:,jj,ii] = rotMatrices[jj] @ (rotMatrices[jj].T @ (x_middle - np.array(obs[jj].x0) ) *( np.tile(Gamma_dynCenter/Gamma_intersec, (dim) )  )**(1./(2*np.array(obs[jj].p) ) ) )
            
    for ii in range(N_obs): # Assign dynamic center 
        if ii in intersection_obs:
            continue # Don't reasign dynamic center if intersection exists

        if np.sum(abs(weight_obs_temp[ii,:])): # Some obstacles are close to each other
            # Check if there are points on the surface of the obstacle
            pointOnSurface = (weight_obs_temp == -1) 
            if np.sum(pointOnSurface):
                weight_obs= 1*pointOnSurface # Bool to float
            else:
                weight_obs = weight_obs_temp[:,ii]/ np.sum(weight_obs_temp[:,ii])

            # Linear interpolation if at least one close obstacle --- MAYBE
            # change to nonlinear
            x_centDyn = np.squeeze(x_cyn_temp[:,ii,:])

            obs[ii].center_dyn = np.sum(x_centDyn * np.tile(weight_obs,(dim,1)), axis=1) + obs[ii].x0
            #plt.plot(obs[ii].center_dyn[0], obs[ii].center_dyn[1], 'k.')
        else: # default center otherwise
            obs[ii].center_dyn = obs[ii].x0


def check_for_intersection(obs_samp, obs_test, delta_dist, N_resol, thetaRange, a_xRange_old, rotMat):
    a_range = obs_samp.a+delta_dist # New ellipse axis

    x_obs_temp = drawEllipse_bound(obs_samp, N_resol=N_resol, theta_range=thetaRange, axis_length=a_range)
    
    Gamma = np.sum( ( 1/obs_test.sf * rotMat.T @ (x_obs_temp- np.tile(obs_test.x0,(N_resol,1)).T ) / (np.tile(obs_test.a,(N_resol,1)).T+delta_dist) )**(2*np.tile(obs_test.p,(N_resol,1)).T ), axis=0)

    intersection_ind_temp = Gamma<1
    n_intersection = np.sum(intersection_ind_temp) #, x_range, N_resol)

    return intersection_ind_temp, n_intersection, x_obs_temp, a_range


def drawEllipse_bound(obs, N_resol=16, theta_range=[0,2*pi], axis_length=[0,0]):
    th_r = obs.th_r
    x0 = obs.x0
    p = obs.p

    if not np.sum(axis_length):
        axis_length = obs.a 

    dim = obs.d # Dimension

    R = np.array(obs.rotMatrix)

    theta_range[1]= (theta_range[1]<theta_range[0])*2*pi + theta_range[1] # ensure thata theta1 > theta2

    theta_min = theta_range[0]
    dTheta = (theta_range[1]- theta_range[0])/(N_resol-1)

    # Angles of evaluation
    theta = np.array([theta_min + (it-1)*dTheta for it in range(N_resol)])
    theta = theta - (theta>pi)*2*pi

    # New boundary points
    x_obs = np.zeros((dim, N_resol))
    x_obs[0,:] = axis_length[0]*np.cos(theta)
    x_obs[1,:] = axis_length[1]*np.copysign(1,theta)*(1 - np.cos(theta)** (2*np.tile(p[0],(1,N_resol))))**(1./(2*np.tile(p[1],(1,N_resol))))
    
    x_obs = R @ x_obs # Rotate obstacles 
    x_obs += np.tile(x0, (N_resol,1)).T # Recenter around x0
    
    return x_obs
