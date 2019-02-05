# coding: utf-8
'''
Obstacle Avoidance Algorithm script with vecotr field

@author LukasHuber
@date 2018-02-15
'''

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#from math import sin, cos, atan2,
#first change the cwd to the script path
#scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
#os.chdir(scriptPath)
import sys
 
# Custom libraries
from dynamicalSystem_lib import *

#if not isinstance("./lib_obstacleAvoidance", sys.path):
#lib_string = "./lib_obstacleAvoidance/"
lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

from draw_ellipsoid import *
from lib_obstacleAvoidance import obs_check_collision_2d
from class_obstacle import *
from lib_modulation import *
from obs_common_section import *
from obs_dynamic_center_3d import *

def Simulation_vectorFields(x_range=[0,10],y_range=[0,10], resolutionField=10, obs=[], sysDyn_init=False, xAttractor = np.array(([0,0])), saveFigure = False, figName='default', noTicks=True, showLabel=True, figureSize=(7.,6), obs_avoidance_func=obs_avoidance_convergence, attractingRegion=False, drawVelArrow=False, colorCode=False, streamColor=[0.05,0.05,0.7], obstacleColor=[], plotObstacle=True, plotStream=True, figHandle=[], alphaVal=1, dynamicalSystem=linearAttractor, nonlinear=False, hirarchyLabel=True):
    #fig_ifd, ax_ifd = plt.subplots(figsize=(10,8)) 
    if len(figHandle): 
        fig_ifd, ax_ifd = figHandle[0], figHandle[1] 
    else: 
        fig_ifd, ax_ifd = plt.subplots(figsize=figureSize) 
    
    # Numerical hull of ellipsoid 
    for n in range(len(obs)): 
        obs[n].draw_ellipsoid(numPoints=50) # 50 points resolution 

    # Adjust dynamic center 
    if nonlinear:
        intersection_obs = obs_common_section_hirarchy(obs)
    else:
      intersection_obs = obs_common_section(obs)  

    #dynamic_center_3d(obs, intersection_obs) 

    # Create meshrgrid of points
    N_x = resolutionField
    N_y = resolutionField
    YY, XX = np.mgrid[y_range[0]:y_range[1]:N_y*1j, x_range[0]:x_range[1]:N_x*1j]

    
    if attractingRegion: # Forced to attracting Region
        def obs_avoidance_temp(x, xd, obs):
            #return obs_avoidance_func(x, xd, obs, attractor=xAttractor)
            return obs_avoidance_func(x, xd, obs, xAttractor)
        obs_avoidance= obs_avoidance_temp
    else:
        obs_avoidance = obs_avoidance_func
        
    # Initialize array
    xd_init = np.zeros((2,N_x,N_y))    
    xd_mod  = np.zeros((2,N_x,N_y))
    for ix in range(N_x):
        for iy in range(N_y):
            if nonlinear:
                xd_mod[:,ix,iy] = obs_avoidance_func(np.array([XX[ix,iy],YY[ix,iy]]), dynamicalSystem, obs, attractor=xAttractor)
                # xd_mod[:,ix,iy] = xd_init[:,ix,iy]
            else:
                pos = np.array([XX[ix,iy],YY[ix,iy]])
                xd_init[:,ix,iy] = dynamicalSystem(pos, x0=xAttractor) # initial DS
                #xd_init[:,ix,iy] = constVelocity(xd_init[:,ix,iy], pos)
                xd_mod[:,ix,iy] = obs_avoidance(pos, xd_init[:,ix,iy], obs) # modulataed DS with IFD
                #xd_mod[:,ix,iy] = constVelocity(xd_mod[:,ix,iy], pos)
    
    if sysDyn_init:
        #fig_init, ax_init = plt.subplots(figsize=(10,8))
        fig_init, ax_init = plt.subplots(figsize=(5,2.5))
        res_init = ax_init.streamplot(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[(0.3,0.3,0.3)])
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

    if plotStream:
        if colorCode:
            velMag = np.linalg.norm(np.dstack((dx1_noColl, dx2_noColl)), axis=2 )/6*100
                        
            strm = res_ifd = ax_ifd.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=velMag, cmap='winter', norm=matplotlib.colors.Normalize(vmin=0, vmax=10.) )
            #fig_cc = plt.figure()
            #fig_cc.colorbar(strm.lines)
        else:
            # res_ifd = ax_ifd.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=streamColor)
            res_ifd = ax_ifd.quiver(XX, YY,dx1_noColl, dx2_noColl, color=streamColor)
            #res_ifd.set_alpha(alphaVal)
        #res_ifd = ax_ifd.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=[29./255,29./255,199./255])
        #res_ifd = ax_ifd.vectorfield(XX, YY,dx1_noColl, dx2_noColl, color=[0.5,0.5,0.5])
        #res_ifd = ax_ifd.quiver(XX, YY,dx1_noColl, dx2_noColl, color=[0.5,0.5,0.5])

        ax_ifd.plot(xAttractor[0],xAttractor[1], 'k*',linewidth=18.0, markersize=18)
    
    plt.gca().set_aspect('equal', adjustable='box')

    ax_ifd.set_xlim([x_range[0],x_range[1]])
    ax_ifd.set_ylim(y_range)

    if noTicks:
        plt.tick_params(axis='both', which='major',bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    if showLabel:
        plt.xlabel(r'$\xi_1$', fontsize=16)
        plt.ylabel(r'$\xi_2$', fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    # Draw obstacles
    if plotObstacle:
        obs_polygon = []
        for n in range(len(obs)):
            x_obs_sf = obs[n].x_obs_sf # todo include in obs_draw_ellipsoid
            obs_polygon.append( plt.Polygon(obs[n].x_obs))
            #print(obstacleColor)
            if len(obstacleColor)==len(obs):
                obs_polygon[n].set_color(obstacleColor[n])
            else:
                obs_polygon[n].set_color(np.array([176,124,124])/255)
            plt.gca().add_patch(obs_polygon[n])

            #x_obs_sf_list = x_obs_sf[:,:,n].T.tolist()
            plt.plot([x_obs_sf[i][0] for i in range(len(x_obs_sf))],
                [x_obs_sf[i][1] for i in range(len(x_obs_sf))], 'k--')

            ax_ifd.plot(obs[n].x0[0],obs[n].x0[1],'k.')
            if hasattr(obs[n], 'center_dyn'):# automatic adaptation of center 
                ax_ifd.plot(obs[n].center_dyn[0],obs[n].center_dyn[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)
                # ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
                dx = 0.1
                ax_ifd.annotate('{}'.format(obs[n].hirarchy), xy=np.array(obs[n].x0)+0.08, textcoords='data', size=16, weight="bold")

            if drawVelArrow and np.linalg.norm(obs[n].xd)>0:
                col=[0.5,0,0.9]
                fac=5 # scaling factor of velocity
                ax_ifd.arrow(obs[n].x0[0], obs[n].x0[1], obs[n].xd[0]/fac, obs[n].xd[1]/fac, head_width=0.3, head_length=0.3, linewidth=10, fc=col, ec=col, alpha=1)

    plt.ion()
    plt.show()
    
    if saveFigure:
        plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + figName + '.eps', bbox_inches='tight')
        #plt.grid(True)
        print('implement figure saving')
        # Remove transparency
        #axins.patch.set_alpha(1)

        return fig_ifd, ax_ifd