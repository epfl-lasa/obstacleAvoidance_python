# coding: utf-8

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
lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

#from draw_ellipsoid import *
#from lib_obstacleAvoidance import obs_check_collision_2d
#from class_obstacle import *
#from lib_modulation import *
#from obs_common_section import *
#from obs_dynamic_center_3d import *
from simulationVectorFields import *

def pltLines(pos0, pos1, xlim=[-100,100], ylim=[-100,100]):
    if pos1[0]-pos0[0]: # m < infty
        m = (pos1[1] - pos0[1])/(pos1[0]-pos0[0])
        n
        ylim=[0,0]
        ylim[0] = pos0[1] + m*(xlim[0]-pos0[0])
        ylim[1] = pos0[1] + m*(xlim[1]-pos0[0])
    else:
        xlim = [pos1[0], pos1[0]]
    
    plt.plot(xlim, ylim, '--', color=[0.3,0.3,0.3], linewidth=2)

options = [7]
for option in options:
    if option==-1:

        xlim = [-0.8,4.2]
        ylim = [-2,2]

        xAttractor = [0,0]

        N_points = 100
        saveFigures = True
        

        # Linear system and circular obstacle
        obs = []

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem', noTicks=True)
        
        a=[1, 1]
        p=[1,1]
        x0=[1.5,0]
        th_r=0/180*pi
        sf=1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_avoidanceCircle', noTicks=True)


    if option==-1:
        # Two ellipses placed at x1=0 with dynamic center diplaced and center line in gray
        obs = []
        a=[0.4, 1]
        p=[1, 1]
        x0=[1.5, 0]
        th_r=0/180*pi
        sf=1
        
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        xlim = [-0.5,4]
        ylim = [-2,2]

        xAttractor = [0,0]

        N_points = 100

        obs[0].center_dyn = x0
        
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=False, figName='ellipse_centerMiddle', noTicks=True, )
        
    if option==0:
        # Two ellipses placed at x1=0 with dynamic center diplaced and center line in gray
        obs = []
        a=[0.4, 1]
        p=[1,1]
        x0=[1.5,0]
        th_r=0/180*pi
        sf=1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        #obs[n].center_dyn = np.array([2,1.4])

        xlim = [-0.5,4]
        ylim = [-2,2]

        xAttractor = [0,0]

        N_points = 100

        obs[0].center_dyn = x0
        
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=False, figName='ellipse_centerMiddle', noTicks=True)

        pltLines(xAttractor, obs[0].center_dyn)
        plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + 'ellipseCenterMiddle_centerLine' + '.eps', bbox_inches='tight')       
        rat = 0.6
        obs[0].center_dyn = [x0[0] - rat*np.sin(th_r)*a[1],
                             x0[1] - rat*np.cos(th_r)*a[1]]
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=False, figName='ellipse_centerNotMiddle', noTicks=True)
        pltLines(xAttractor, obs[0].center_dyn)
        plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + 'ellipseCenterNotMiddle_centerLine' + '.eps', bbox_inches='tight')
        
    if option==0.1:
        # Two ellipses placed at x1=0 with dynamic center diplaced and center line in gray -- with color Code for velocity
        obs = []
        a=[0.5, 1.4]
        p=[1,1]
        x0=[2,0]
        th_r=0/180*pi
        sf=1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        #obs[n].center_dyn = np.array([2,1.4])

        xlim = [-0.5,4]
        ylim = [-2,2]

        xAttractor = [0,0]

        N_points = 100

        #obs[0].center_dyn = [10,10]

        Simulation_vectorFields(xlim, ylim, N_points, [], xAttractor=xAttractor, saveFigure=False, figName='ellipse_centerNotMiddle', noTicks=True, obs_avoidance_func=obs_avoidance_ellipsoid, showLabel=False, colorCode=True)
        plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + 'ellipse_initialLinear_colMap' + '.eps', bbox_inches='tight')

        
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=False, figName='ellipse_centerNotMiddle', noTicks=True, obs_avoidance_func=obs_avoidance_ellipsoid, showLabel=False, colorCode=True)
        plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + 'ellipse_localMinima_colMap' + '.eps', bbox_inches='tight')

        obs[0].center_dyn = x0
        
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=False, figName='ellipse_centerMiddle', noTicks=True, showLabel=False, colorCode=True)
        pltLines(xAttractor, obs[0].center_dyn)
        plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + 'ellipseCenterMiddle_centerLine_pres_colMap' + '.eps', bbox_inches='tight')

        rat = -0.9
        obs[0].center_dyn = [x0[0] - rat*np.sin(th_r)*a[1],
                             x0[1] - rat*np.cos(th_r)*a[1]]
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=False, figName='ellipse_centerNotMiddle', noTicks=True, showLabel=False, colorCode=True)
        pltLines(xAttractor, obs[0].center_dyn)
        plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + 'ellipseCenterNotMiddle_centerLine_pres_colMap' + '.eps', bbox_inches='tight')

        

    if option==1:
        # Two ellipses combined to represent robot arm -- remove outlier plotting when saving!!
        obs = []
        obs.append(Obstacle(
            a=[1.1, 1.4],
            p=[2,2],
            x0=[3.0, 1.3],
            th_r=0/180*pi,
            sf=1))

        obs.append(Obstacle(
            a=[2, 0.4],
            p=[2,2],
            x0=[2.5, 3],
            th_r=-20/180*pi,
            sf=1))

        xlim = [-1.0, 6.5]
        ylim = [-0.3, 5.2]

        xAttractor = [0,0]

        N_points = 100

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=False, figName='convexRobot', noTicks=True)

    if option==2:
        # Decomposition several obstacles - obstacle 1, obstacle 2, both obstacles
        xAttractor = np.array([0,0])
        centr = [2, 2.5]

        obs = []
        a = [0.5,1.2]
        p = [1,1]
        x0 = [-1.0, 3.2]
        th_r = -60/180*pi
        sf = 1.0
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [1.4,1.0]
        p = [3,3]
        x0 = [1.2, 1.5]
        th_r = -30/180*pi
        sf = 1.0
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        
        xlim = [-2.5,3.2]
        ylim = [-0.3, 5.2]
        
        N_points = 100

        saveFigures = True

        col1= [0.7,0.3,0]
        fig, ax =Simulation_vectorFields(xlim, ylim, N_points, [obs[0]], xAttractor=xAttractor, saveFigure=False, figName='linearCombination_obstacle0', noTicks=True, streamColor=col1, obstacleColor=[col1], alphaVal=0.7)

        col2 = [0.05,0.3,0.05]
        Simulation_vectorFields(xlim, ylim, N_points, [obs[1]], xAttractor=xAttractor, saveFigure=saveFigures, figName='linearCombination_obstacle_overlay', noTicks=True, streamColor=col2, obstacleColor=[col2], figHandle=[fig, ax], alphaVal=0.7)
        
        #Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearCombination_obstaclesBoth', noTicks=True, obstacleColor = [col1,col2], plotStream=False)

        #, figHandle=[fig,ax])

        saveFigure=True
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearCombination_obstaclesBoth', noTicks=True)

    if option==3:
        # Three obstacles touching, with and without common center
        N_points = 100
        saveFigures=True

        xlim = [-1,7]
        xlim = [xlim[ii]*0.8 for ii in range(2)]
        
        ylim = [-3.5,3.5]
        ylim = [ylim[ii]*0.8 for ii in range(2)]

        ### Three obstacles touching - convergence
        xAttractor = np.array([0,0])
        centr = [2, 0]
        dx = 0.5
        dy = 2.5
        
        obs = []
        a = [0.6,0.6]
        p = [1,1]
        x0 = [2.+dx, 3.2-dy]
        th_r = -60/180*pi
        sf = 1.2
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        #obs[0].center_dyn = centr
        
        a = [1,0.4]
        p = [1,3]
        x0 = [1.5+dx, 1.6-dy]
        th_r = +60/180*pi
        sf = 1.2
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        #obs[1].center_dyn = centr

        a = [1.2,0.2]
        p = [2,2]
        x0 = [3.3+dx,2.1-dy]
        th_r = -20/180*pi
        sf = 1.2
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        #obs[2].center_dyn = centr


        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='three_obstacles_touching')


    #if option==3.1:
        xAttractor = np.array([0,0])
        centr = [1.5, 0]

        dx = 0.5
        dy = 3.0
        
        ### Three obstacles touching -- no common center, no convergence
        obs = []
        a = [0.6,0.6]
        p = [1,1]
        x0 = [2.5+dx, 4.1-dy]
        th_r = 0*pi
        sf = 1.2
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [1.2,0.2]
        p = [2,2]
        x0 = [2.3+dx,3.1-dy]
        th_r = 20/180*pi
        sf = 1.2
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        
        a = [1,0.4]
        p = [1,4]
        x0 = [2.5+dx, 2.0-dy]
        th_r= -25/180*pi
        sf = 1.2
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        
       
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='three_obstacles_touching_noConvergence')

    if option==4:
        # Obstacles being overlapping an being perpendicular or parallel to flow 
        #N_points = 100
        saveFigures=True

        xlim = [-1,7]
        ylim = [-3.5,3.5]

        ### Three obstacles touching - convergence
        xAttractor = np.array([0,0])
        centr = [2, 2.5]
        
        obs = []
        a = [1.4,0.3]
        p = [1,1]
        x0 = [3, 0.9]
        th_r = +40/180*pi
        sf = 1.2
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [1.2,0.4]
        p = [1,3]
        x0 = [3, -1.0]
        th_r = -40/180*pi
        sf = 1.2
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='twoObstacles_concaveRegion_front')
        
        obs = []
        a = [1.8,0.3]
        p = [1,1]
        x0 = [4.0, -0.4]
        th_r = +60/180*pi
        sf = 1.2
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [1.2,0.4]
        p = [1,3]
        x0 = [2.3, -0.0]
        th_r = -40/180*pi
        sf = 1.2
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='twoObstacles_concaveRegion_top')
        
    if option==5:
        # Obstacles being overlapping an being perpendicular or parallel to flow 
        N_points = 100
        saveFigures=False

        xlim = [-1,4]
        ylim = [-2,2]

        ### Three obstacles touching - convergence
        xAttractor = np.array([0,0])
        
        obs = []
        a = [1.1,1.1]
        p = [1,1]
        x0 = [1.6, 0.0]
        th_r = +0/180*pi
        sf = 1.0
        xd = [-3,3]
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd))

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='movingObstacle_movingFrame', obs_avoidance_func=obs_avoidance_interpolation, drawVelArrow=True)

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='movingObstacle_partialMovingFrame', obs_avoidance_func=obs_avoidance_interpolation_moving, drawVelArrow=True)

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='movingObstacle_forcedAttraction', obs_avoidance_func=obs_avoidance_interpolation_moving, attractingRegion=True, drawVelArrow=True)

        obs=[]
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='movingObstacle_notMoving', obs_avoidance_func=obs_avoidance_interpolation_moving, drawVelArrow=True)

    if option==6:
        # TOOD -- radial displacement algorithm
        # Obstacles being overlapping an being perpendicular or parallel to flow 
        N_points = 100
        saveFigures=False

        xlim = [-1,4]
        ylim = [-2,2]

        ### Three obstacles touching - convergence
        xAttractor = np.array([0,0])
        
        obs = []
        a = [0.4,1.5]
        p = [1,1]
        x0 = [1.6, 0.0]
        th_r = +0/180*pi
        sf = 1.0
        xd = [-3,3]
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='movingObstacle_notMoving', obs_avoidance_func=obs_avoidance_radialDisplace)

    if option==7:
        # Obstacle Avoidance with and without tail effect
        
        N_resol = 100
        saveFigures=True

        xlim = [-0.1,4]
        ylim = [-2,2]

        ### Three obstacles touching - convergence
        xAttractor = np.array([0,0])
        
        obs = []
        a = [0.35,1.4]
        p = [1,1]
        x0 = [1.9, 0.0]
        th_r = -15/180*pi
        sf = 1.0
        xd = [-3,3]
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, tail_effect=False))
        

        Simulation_vectorFields(xlim, ylim, N_resol, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='ellipse_tailEffectOff', obs_avoidance_func=obs_avoidance_interpolation_moving, showLabel=False)

        obs[0].tail_effect = True

        Simulation_vectorFields(xlim, ylim, N_resol, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='ellipse_tailEffectOn', obs_avoidance_func=obs_avoidance_interpolation_moving, showLabel=False)
