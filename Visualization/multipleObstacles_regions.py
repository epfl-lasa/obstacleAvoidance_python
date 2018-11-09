# coding: utf-8

from math import pi

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)


from draw_ellipsoid import *
from lib_obstacleAvoidance import obs_check_collision_2d
from class_obstacle import *
from lib_modulation import *
from obs_common_section import *

from PIL import Image

from dynamicalSystem_lib import *

def visualizationRegions(x_range=[0,10],y_range=[0,10], resolutionField=10, obs=[], sysDyn_init=False, xAttractor = np.array(([0,0])), saveFigure = False, figName='default', noTicks=True):
    
    fig, ax = plt.subplots(figsize=(7.8,6))
    
    # Numerical hull of ellipsoid
    for n in range(len(obs)):
        obs[n].draw_ellipsoid(numPoints=50) # 50 points resolution

    # Create meshrgrid of points
    N_x = resolutionField
    N_y = resolutionField
    YY, XX = np.mgrid[y_range[0]:y_range[1]:N_y*1j, x_range[0]:x_range[1]:N_x*1j]

    # Initialize array
    xd_init = np.zeros((2,N_x,N_y))
    xd_mod  = np.zeros((2,N_x,N_y))

    # BooleanVar Matrix - True if the modulated DS could be written as a lienar combination of obstacle n and the initial DS
    linearComb = np.zeros((len(obs), N_x, N_y))
    rgb_regions = np.zeros((N_x, N_y, 3), 'float32')

    for ix in range(N_x):
        for iy in range(N_y):
            pos = np.array([XX[ix,iy],YY[ix,iy]])
            
            xd_init[:,ix,iy] = linearAttractor(pos, x0 = xAttractor ) # initial DS
            
            xd_mod[:,ix,iy] = obs_avoidance_interpolation(pos, xd_init[:,ix,iy],obs) # modulataed DS with IFDs
            xd_mod[:,ix,iy] = constVelocity(xd_mod[:,ix,iy], pos)
            
            rotation_direction = np.cross(xd_init[:,ix,iy], xd_mod[:,ix,iy])

            
            for n in range(len(obs)):
                # Rotation if there were only obstacle 'n'
                xd_mod_n = obs_avoidance_interpolation(pos, xd_init[:,ix,iy],[obs[n]])

                rotation_direction_n = np.cross(xd_init[:,ix,iy], xd_mod_n)
                
                if np.copysign(1,rotation_direction) == np.copysign(1,rotation_direction_n):
                    rgb_regions[ix,iy,n] = 0.5
                    
                #rotation_direction_n = np.cross(pos-obs[n].x0,xd_mod[:,ix,iy])
                #pos_direction = np.cross( obs[n].x0, pos-obs[n].x0)
                #if rotation_direction_n *  pos_direction > 0:
                    #rgb_regions[ix,iy,n] = 0.5

            # alphaVal=0.6
            # if np.sum(linearComb[:,ix,iy]) == 2: # purple
            #     plt.plot(XX[ix,iy], YY[ix,iy], 's', color=[128/255,0,128/255], alpha=alphaVal)
            # elif linearComb[1,ix,iy]:  # blue
            #     plt.plot(XX[ix,iy], YY[ix,iy], 's', color=[0.1,0.1,0.7], alpha=alphaVal)
            # elif linearComb[0,ix,iy]: # red
            #     plt.plot(XX[ix,iy], YY[ix,iy], 's', color=[0.7,0.1,0.1], alpha=alphaVal)
            # else:# gray
            #     plt.plot(XX[ix,iy], YY[ix,iy], 's', color=[0.3,0.3,0.3], alpha=alphaVal)
    
    
    if sysDyn_init:
        #fig_init, ax_init = plt.subplots(figsize=(10,8))
        fig_init, ax_init = plt.subplots(figsize=(5,2.5))
        res_init = ax_init.streamplot(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[(0.3,0.3,0.3)])
        #res_init = ax_init.quiver(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[(0.3,0.3,0.3)])
        #res_init = ax_init.quiver(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[(0.3,0.3,0.3)])
        
        ax_init.plot(xAttractor[0],xAttractor[1], 'k*')
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlim(x_range)
        plt.ylim(y_range)

        if safeFigure:
            print('implement figure saving')

    collisions = obs_check_collision_2d(obs, XX, YY)
    
    dx1_noColl = np.squeeze(xd_mod[0,:,:]) * collisions
    dx2_noColl = np.squeeze(xd_mod[1,:,:]) * collisions

    fx1_noColl = np.squeeze(xd_init[0,:,:]) * collisions
    fx2_noColl = np.squeeze(xd_init[1,:,:]) * collisions
    
    #res_ifd = ax_ifd.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=[0.5,0.5,0.5])
    #res_ifd = ax_ifd.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=[29./255,29./255,199./255])
    #res_init = ax.streamplot(XX, YY,fx1_noColl, fx2_noColl, color=[0.1,0.1,0.6])
    res_mod = ax.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=[0.0,0.0,0.0])
    #res_init = ax.quiver(XX, YY,fx1_noColl, fx2_noColl, color=[0.1,0.1,0.6])
    #res_mod = ax.quiver(XX, YY,dx1_noColl, dx2_noColl, color=[0.5,0.5,0.5])
    
    ax.plot(xAttractor[0],xAttractor[1], 'k*',linewidth=18.0, markersize=18)
    
    plt.gca().set_aspect('equal', adjustable='box')

    ax.set_xlim([x_range[0],x_range[1]])
    ax.set_ylim(y_range)

    if noTicks:
        plt.tick_params(axis='both', which='major',bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    else:
        plt.xlabel(r'$\xi_1$', fontsize=16)
        plt.ylabel(r'$\xi_2$', fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    
    # Draw obstacles
    obs_polygon = []
    for n in range(len(obs)):
        x_obs_sf = obs[n].x_obs_sf # todo include in obs_draw_ellipsoid
        obs_polygon.append( plt.Polygon(obs[n].x_obs))
        obs_polygon[n].set_color(np.array([176,124,124])/255)
        plt.gca().add_patch(obs_polygon[n])
        
        #x_obs_sf_list = x_obs_sf[:,:,n].T.tolist()
        plt.plot([x_obs_sf[i][0] for i in range(len(x_obs_sf))],
            [x_obs_sf[i][1] for i in range(len(x_obs_sf))], 'k--')

        ax.plot(obs[n].x0[0],obs[n].x0[1],'k.')
        if hasattr(obs[n], 'center_dyn'):# automatic adaptation of center 
            ax.plot(obs[n].center_dyn[0],obs[n].center_dyn[1], 'k+', markersize=16, linewidth=20)

    # Draw image after setting plot axis
    dx = (x_range[1]-x_range[0])/(N_x-1)/2
    dy = (y_range[1]-y_range[0])/(N_y-1)/2
    ax.imshow(rgb_regions, extent = (x_range[0]-dx,x_range[1]+dx,y_range[0]-dy,y_range[1]+dy), origin='lower')
    

    plt.ion()
    plt.show()

options = [0]
for option in options:
    if option==0:
        xAttractor = np.array([0,0])
        centr = [2, 2.5]

        obs = []
        obs.append(Obstacle(
            a = [1,3],
            p = [1,1],
            x0 = [-2, 0],
            th_r = -30/180*pi,
            sf = 1.0))
        
        obs.append(Obstacle(
            a = [1.1,1.4],
            p = [1,1],
            x0 = [2, 3],
            th_r = 0/180*pi,
            sf = 1.0))
        
        obs.append(Obstacle(
            a = [0.8,0.3],
            p = [1,1],
            x0 = [-3, 3],
            th_r = -30/180*pi,
            sf = 1.0))
        
        x_range = [-5,5]
        y_range = [-3,6]
