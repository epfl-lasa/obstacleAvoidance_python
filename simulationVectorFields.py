'''
Obstacle Avoidance Algorithm script with vecotr field

@author LukasHuber
@date 2018-02-15
'''

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

#from math import sin, cos, atan2,
#first change the cwd to the script path
#scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
#os.chdir(scriptPath)

import sys
 
# Custom libraries
from dynamicalSystem_lib import *

#if not isinstance("./lib_obstacleAvoidance", sys.path):
#lib_string = "./lib_obstacleAvoidance/"
lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm_python/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm_python/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

from draw_ellipsoid import *
from lib_obstacleAvoidance import obs_check_collision
from class_obstacle import *
from obstacleAvoidance_lib import *
from obs_common_section import *
from obs_dynamic_center import *

def Simulation_vectorFields(x_range=[0,10],y_range=[0,10], N_y=10, obs=[], sysDyn_init=False, xAttractor = np.array(([0,0])), safeFigure = False):

    fig_ifd, ax_ifd = plt.subplots(figsize=(10,8))    
    
    # Numerical hull of ellipsoid
    for n in range(len(obs)):
        obs[n].draw_ellipsoid() # 50 points resolution

    # Adjust dynamic center
    intersection_obs = obs_common_section(obs)
    print('intersection_obs', intersection_obs)
    dynamic_center(obs, intersection_obs)

    # Create meshrgrid of points
    N_x = N_y
    YY, XX = np.mgrid[y_range[0]:y_range[1]:N_y*1j, x_range[0]:x_range[1]:N_x*1j]

    # Initialize array
    xd_init = np.zeros((2,N_x,N_y))
    xd_IFD  = np.zeros((2,N_x,N_y))
    for ix in range(N_x):
        for iy in range(N_y):
            pos = np.array([XX[ix,iy],YY[ix,iy]])
            
            xd_init[:,ix,iy] = linearAttractor(pos, x0 = xAttractor ) # initial DS
            
            xd_IFD[:,ix,iy] = obs_avoidance_convergence(pos, xd_init[:,ix,iy],obs) # modulataed DS with IFDs

    if sysDyn_init:
        fig_init, ax_init = plt.subplots(figsize=(10,8))
        res_init = ax_init.streamplot(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[(0.3,0.k3,0.3)])
        
        ax_init.plot(xAttractor[0],xAttractor[1], 'k*')
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlim(x_range)
        plt.ylim(y_range)

        if safeFigure:
            print('implement figure saving')
        
    collisions = obs_check_collision(obs, XX, YY)
    
    dx1_noColl = np.squeeze(xd_IFD[0,:,:]) * collisions
    dx2_noColl = np.squeeze(xd_IFD[1,:,:]) * collisions
    
    #res_ifd = ax_ifd.streamplot(XX, YY,xd_IFD[0,:,:], xd_IFD[1,:,:], color='k')
    res_ifd = ax_ifd.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=[0.5,0.5,0.5])
    #res_ifd = ax_ifd.vectofield(XX, YY,dx1_noColl, dx2_noColl, color=[0.5,0.5,0.5])
    
    ax_ifd.plot(xAttractor[0],xAttractor[1], 'k*',linewidth=7.0)
    
    plt.gca().set_aspect('equal', adjustable='box')

    ax_ifd.set_xlim([x_range[0],x_range[1]])
    ax_ifd.set_ylim(y_range)

    plt.xlabel(r'$\xi_1$')
    plt.ylabel(r'$\xi_2$')

    # Draw obstacles
    obs_polygon = []

    #obs_alternative = obs_draw_ellipsoid()
    for n in range(len(obs)):
        x_obs_sf = obs[n].x_obs # todo include in obs_draw_ellipsoid
        obs_polygon.append( plt.Polygon(obs[n].x_obs))
        obs_polygon[n].set_color(np.array([176,124,124])/255)
        plt.gca().add_patch(obs_polygon[n])
        
        #x_obs_sf_list = x_obs_sf[:,:,n].T.tolist()
        plt.plot([x_obs_sf[i][0] for i in range(len(x_obs_sf))],
                 [x_obs_sf[i][1] for i in range(len(x_obs_sf))], 'k--')

        ax_ifd.plot(obs[n].x0[0],obs[n].x0[1],'k.')
        if hasattr(obs[n], 'center_dyn'):# automatic adaptation of center 
            ax_ifd.plot(obs[n].center_dyn[0],obs[n].center_dyn[1], 'r+')
            
    plt.ion()
    plt.show()
    
    if safeFigure:
        print('implement figure saving')

        # Remove transparency
        #axins.patch.set_alpha(1)

### -------------------------------------------------
# Start main function
#plt.close("all") # close figures

posAttractor = [0,0]

option=0

if option==0:
    obs = []

    a=[0.2, 1]
    p=[3,3]
    x0=[1.5,1]
    th_r=0/180*pi
    sf=1
    
    xd=[5,0]
    x_start = 0
    x_end = 2
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end))
    #obs[n].center_dyn = np.array([2,1.4])

    # Obstacle 2
    a = [0.4,0.2]
    p = [4,4]
    x0 = [1.9,1.3]
    th_r = 0/180*pi
    sf = 1
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
    #obs[n].center_dyn = np.array([2,1.4])

    xlim = [-1,4]
    ylim = [-0.1,3]

if option==1:
    ### Create obstacle 
    obs = []

    a = [5,1]
    p = [1,1]
    x0 = [5,0]
    th_r = 30/180*pi
    sf = 1
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

    a = [7,2]
    p = [1,1]
    x0 = [7,1]
    th_r = -40/180*pi
    sf = 1
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

if option==2:
    ### Create obstacle 
    obs = []

    a = [5,2]
    p = [1,1]
    x0 = [4,2]
    th_r = 30/180*pi
    sf = 1
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

    a = [7,2]
    p = [1,1]
    x0 = [7,-2]
    th_r = -40/180*pi
    sf = 1
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

if option==3:
    obs = []

    a=[0.5, 0.8]
    p=[1,1]
    x0=[1.5,1.3]
    th_r=0/180*pi
    sf=1
    w = 0
    xd=[0,5]
    x_start = 0
    x_end = 2
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))

    xlim = [-1,4]
    ylim = [-0.1,3]
    
N_points = 10

xAttractor = np.array([0,1.3])

Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor)
#Simulation_vectorFields([-10,10],[-10,10], 30, 30, obs)

# # For testing reasons
pos = np.array([0,3])
# xd_init = linearAttractor(pos, x0 = posAttractor ) # initial DS
# xd_IFD = IFD(pos, xd_init,obs) # modulataed DS with IFD

print('finished script')
