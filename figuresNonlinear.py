# coding: utf-8
'''
Obstacle Avoidance Algorithm script with vecotr field

@author LukasHuber
@date 2018-02-15
'''


# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

from math import pi

#first change the cwd to the script path
#scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
#os.chdir(scriptPath)

import sys

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

# Custom libraries
# from simulationVectorFields import *
from vectorField_visualization import *
from dynamicalSystem_lib import *
# from nonlinear_modulations import *

saveFigures=True
options=[6]

for option in options:
    if option==-1:
        theta = 0*pi/180
        n = 0

        pos = np.zeros((2))
        pos[0]= obs[n].a[0]*np.cos(theta)
        pos[1] = np.copysign(obs[n].a[1], theta)*(1 - np.cos(theta)**(2*obs[n].p[0]))**(1./(2.*obs[n].p[1]))

        pos = obs[n].rotMatrix @ pos + obs[n].x0
        
        xd = obs_avoidance_nonlinear_radial(pos, nonlinear_stable_DS, obs, attractor=xAttractor)
    if option==0:

        xlim = [-0.8,7]
        ylim = [-3.3,3.3]

        xAttractor=[0,0]

        N_points=50
        
        obs=[]
        # Obstacle 2
        a = [0.4,2.2]
        p = [1,1]
        x0 = [6,0]
        th_r = 0/180*pi
        sf = 1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        #obs[n].center_dyn = np.array([2,1.4])

        
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinearSystem', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_radial, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)

    if option==1:
        N_resolAxis=100

        xlim = [-0.8,5]
        ylim = [-2.5,2.5]

        xAttractor=[0,0]

        N_it = 4
        for ii in range(N_it):
            obs=[]
            a = [0.5, 2]
            p = [1,1]
            x0 = [2.2, 0.1]
            th_r = 30/180*pi
            sf = 1
            
            #if ii>0:
            if True:
                # Obstacle 2
                a = [a[jj]*(ii+0.01)/(N_it-1) for jj in range(len(a))]
                obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        
            Simulation_vectorFields(xlim, ylim, N_resolAxis, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinearGrowing'+str(ii), noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_radial, dynamicalSystem=nonlinear_wavy_DS, nonlinear=True)

    if option==2:
        xlim = [0.3,13]
        ylim = [-6,6]

        xAttractor=[0,0]

        N_points=100
        #saveFigures=True

        obs=[]
        # Obstacle 2
        a = [0.9,5.0]
        p = [1,1]
        x0 = [5.5,0]
        th_r = -30/180*pi
        sf = 1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        #obs[n].center_dyn = np.array([2,1.4])

        
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_modulation', noTicks=True, obs_avoidance_func=obs_avoidance_interpolation_moving, dynamicalSystem=nonlinear_stable_DS, nonlinear=False)

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_displacement', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_radial, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)

        obs = []
        Simulation_vectorFields(xlim, ylim, N_points, obs=[], xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_noObs', noTicks=True, obs_avoidance_func=obs_avoidance_interpolation_moving, dynamicalSystem=nonlinear_stable_DS, nonlinear=False)


    if option==3:
        xlim = [0.3,13]
        ylim = [-6,6]

        xAttractor=[0,0]

        N_points=110
        #saveFigures=True

        obs=[]
        
        a = [1.0,1.0]
        p = [1,1]
        x0 = [5.5,0]
        th_r = 20/180*pi
        sf = 1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))


        a = [2.0,0.8]
        p = [1,1]
        x0 = [2,1]
        th_r = 50/180*pi
        sf = 1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [2.5,1.5]
        p = [1,1]
        x0 = [8,4]
        th_r = 30/180*pi
        sf = 1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [0.4,2.2]
        p = [1,1]
        x0 = [10,-3]
        th_r = 80/180*pi
        sf = 1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [0.9,1.1]
        p = [1,1]
        x0 = [3,-4]
        th_r = 80/180*pi
        sf = 1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_multipleObstacles', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_radial, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)

        obs = []
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_multipleObstacles_initial', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_radial, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)


    if option==4:
        xlim = [0.3,13]
        ylim = [-6,6]

        xAttractor=[0,0]

        N_points=120
        #saveFigures=True

        obs=[]
        
        a = [.80,3.0]
        p = [1,1]
        x0 = [5.5,-1]
        th_r = 40/180*pi
        sf = 1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        obs[-1].center_dyn = np.array([ 3.87541829,  0.89312174])

        a = [1.0,3.0]
        p = [1,1]
        x0 = [5.0,2]
        th_r = -50/180*pi
        sf = 1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        obs[-1].center_dyn = np.array([ 3.87541829,  0.89312174])
        
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_intersectingObstacles', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_radial, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)

        obs = []
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_intersectingObstacles_initial', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_radial, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)
        
    if option==5:
        xlim = [-3,10]
        ylim = [-6,6]

        xAttractor=[0,0]

        N_points=110

        obs=[]
        # Obstacle 2
        a = [2.0, 3.5]
        p = [1,1]
        x0 = [5.3, -.4]
        th_r = -30/180*pi
        sf = 1
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_convergence', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_radial, dynamicalSystem=nonlinear_wavy_DS, nonlinear=True)

        # Simulation_vectorFields(xlim, ylim, N_points, obs, saveFigure=saveFigures, figName='nonlinear_convergence_none', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_radial, dynamicalSystem=nonlinear_wavy_DS, nonlinear=True)
    if option==6:
        xlim = [0.3,13]
        ylim = [-6,6]

        xAttractor=[0,0]

        N_points=20
        #saveFigures=True

        obs=[]
        
        a = [2.0,2.0]
        p = [1,1]
        x0 = [5.5,1]
        th_r = 20/180*pi
        sf = 1
        hirarchy=0
        parent = 'root'
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, hirarchy=hirarchy, parent=parent))

        a = [3,0.8]
        p = [1,1]
        x0 = [8,3]
        th_r = 30/180*pi
        sf = 1
        hirarchy=1
        parent = obs[0]
        # obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, hirarchy=hirarchy, parent=parent))

        a = [0.4,2.2]
        p = [1,1]
        x0 = [7, 0]
        th_r = 80/180*pi
        sf = 1
        hirarchy=1
        parent = obs[0]
        # obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, hirarchy=hirarchy, parent=parent))

        a = [0.4,2.2]
        p = [1,1]
        x0 = [8,-2]
        th_r = 20/180*pi
        sf = 1
        hirarchy=2
        parent = obs[-1]
        # obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, hirarchy=hirarchy, parent=parent))

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_treeOfStars', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_hirarchy, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)
