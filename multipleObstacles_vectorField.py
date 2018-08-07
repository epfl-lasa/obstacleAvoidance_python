'''
Several examples
'''

#lib_string = "./lib_obstacleAvoidance/"

import sys

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

from draw_ellipsoid import *
from class_obstacle import *
from lib_modulation import *

from simulationVectorFields import *

def pltLine(pos0, pos1, xlim=[-100,100], ylim=[-100,100]):

    if pos1[0]-pos0[0]: # m < infty
        m = (pos1[1] - pos0[1])/(pos1[0]-pos0[0])
        
        ylim=[0,0]
        ylim[0] = pos0[1] + m*(xlim[0]-pos0[0])
        ylim[1] = pos0[1] + m*(xlim[1]-pos0[0])

    else:
        xlim = [pos1[0], pos1[0]]
    
    plt.plot(xlim, ylim, 'r--')

options = [3]
for option in options:
    if option == 0:
        xAttractor = np.array([0,0])
        centr = [2, 2.5]

        obs = []
        a = [0.2,5]
        p = [1,1]
        x0 = [0.5, 5]
        th_r = -25/180*pi
        sf = 1.0
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [1.1,1]
        p = [1,1]
        x0 = [0.5, 1.5]
        th_r = -30/180*pi
        sf = 1.0
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        
        #xlim = [-4,4]
        #ylim = [-0.1,5]

        #xlim = [-2,0]
        #ylim = [1,3]

        xlim = [-0.7, 0.3]
        ylim = [2.3,3.0]

        N_points = 50

        Simulation_vectorFields(xlim, ylim, N_points, [obs[0]], xAttractor=xAttractor, saveFigure=True,
                                figName='linearCombination_obstacle0_zoom', noTicks=False)

        Simulation_vectorFields(xlim, ylim, N_points, [obs[1]], xAttractor=xAttractor, saveFigure=True,
                                figName='linearCombination_obstacle1_zoom', noTicks=False)

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=True,
                                figName='linearCombination_obstacles_zoom', noTicks=False)

    if option == 1:
        xAttractor = np.array([0,0])
        centr = [2, 2.5]

        obs = []
        a = [0.2,5]
        p = [1,1]
        x0 = [0.5, 5]
        th_r = -25/180*pi
        sf = 1.0
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [1.1,1]
        p = [1,1]
        x0 = [0.5, 1.5]
        th_r = -30/180*pi
        sf = 1.0
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        
        xlim = [-0.7, 0.3]
        ylim = [2.3,3.0]

        N_points = 50

        Simulation_vectorFields(xlim, ylim, N_points, [obs[0]], xAttractor=xAttractor, saveFigure=True,
                                figName='linearCombination_separate_obstacle0_zoom', noTicks=False)

        Simulation_vectorFields(xlim, ylim, N_points, [obs[1]], xAttractor=xAttractor, saveFigure=True,
                                figName='linearCombination_separate_obstacle1_zoom', noTicks=False)

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=True,
                                figName='linearCombination_separate_obstacles_zoom', noTicks=False)

    if option == 2:
        xAttractor = np.array([0,0])
        centr = [2, 2.5]

        obs = []
        a = [1.1,1.2]
        p = [1,1]
        x0 = [-1, 1.5]
        th_r = -25/180*pi
        sf = 1.0
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [3,0.4]
        p = [1,1]
        x0 = [2, 3]
        th_r = 0/180*pi
        sf = 1.0
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
        
        xlim = [-4,4]
        ylim = [-0.1,5]

        #xlim = [-2,-1]
        #ylim = [2,3]

        #xlim = [-1.7,-1.5]
        #ylim = [2.3,2.5]

        xlim = [-1.625,-1.618]
        ylim = [2.432,2.440]

        N_points = 20
        
        saveFig = False
        
        Simulation_vectorFields(xlim, ylim, N_points, [obs[0]], xAttractor=xAttractor, saveFigure=saveFig, figName='linearCombination_separate_obstacle0', noTicks=False)
        pltLine(xAttractor, obs[0].x0)
        pltLine(xAttractor, obs[1].x0)

        Simulation_vectorFields(xlim, ylim, N_points, [obs[1]], xAttractor=xAttractor, saveFigure=saveFig, figName='linearCombination_separate_obstacle1', noTicks=False)
        pltLine(xAttractor, obs[0].x0)
        pltLine(xAttractor, obs[1].x0)

        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFig, figName='linearCombination_separate_obstacles', noTicks=False)
        pltLine(xAttractor, obs[0].x0)
        pltLine(xAttractor, obs[1].x0)

    if option == 3:
        xAttractor = np.array([0,0])
        centr = [2, 2.5]

        obs = []
        a = [1.1,1.2]
        p = [1,1]
        x0 = [-1, 1.5]
        th_r = -25/180*pi
        sf = 1.0
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        a = [3,0.4]
        p = [1,1]
        x0 = [0, 4]
        th_r = 20/180*pi
        sf = 1.0
        obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))


        obs.append(Obstacle(
            a=[1.2,0.4],
            p=[1,1],
            x0=[3,3],
            th_r=-30/180*pi,
            sf=1.0 
            ))
        xlim = [-4,4]
        ylim = [-0.1,6]

        #xlim = [-1,1]
        #ylim = [-1,1]

        #xlim = [-2.5,-1]
        #ylim = [2,3]

        #xlim = [-2,0]
        #ylim = [2,3]

        #xlim = [-1.65,-1.55]
        #ylim = [2.4,2.48]        

        N_points = 20
        
        saveFig = False
        
        Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFig, figName='linearCombination_separate_obstacles', noTicks=False)
        
        pltLine(xAttractor, obs[0].x0)
        pltLine(xAttractor, obs[1].x0)
        pltLine(xAttractor, obs[2].x0)
