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
from dynamicalSystem_lib import *

def pltLines(pos0, pos1, xlim=[-100,100], ylim=[-100,100]):
    if pos1[0]-pos0[0]: # m < infty
        m = (pos1[1] - pos0[1])/(pos1[0]-pos0[0])
        
        ylim=[0,0]
        ylim[0] = pos0[1] + m*(xlim[0]-pos0[0])
        ylim[1] = pos0[1] + m*(xlim[1]-pos0[0])
    else:
        xlim = [pos1[0], pos1[0]]
    
    plt.plot(xlim, ylim, '--', color=[0.3,0.3,0.3], linewidth=2)

def getGammmaValue_ellipsoid(ob, x_t):
    return np.sum((x_t/ob.a)**(2*p), axis=0)

def findBoundaryPoint(ob, direction):
    # Numerical search -- TODO analytic
    dirNorm = LA.norm(direction,2)
    if dirNorm:
        direction = direction/dirNorm
    else:
        print('No feasible direction is given')
        return ob.x0

    a = [min(x0.a), max(x0.a)]

    repetion = 6
    steps = 10
    # repetition
    for ii in range(repetition):
        magnitudDir = np.linspace(a[0], a[1], num=steps)
        
        Gamma = getGammmaValue_ellipsoid(ob, direction*magnitudDir)
        posBoundary = np.where(Gamma<1)[0][-1]

        a[0] = magnitudeDir[posBoundary]

        posBoundary +=1
        while magnitudeDir[posBoundary]<=Gamma:
            posBoundary+1

        a[1] = magnitudeDir[posBoundary]

    return (a[0]+a[1])/2.0*direction + x0

def saddlePointsForConcave(obs, xAttractor=[0,0], ds_init):
    hirarchyList = [0]

    saddlePoints_entr = []
    saddlePoints_exit = []

        
    for hirarchy in hirarchyList:
        for o in range(len(obs)):
            if obs[o].hirarchy == hirarchy:
                if hirarchy: # nonzero
                    saddlePoints_entr.append(findBoundaryPoint(obs[o], obs[obs[o].parent].saddle_entr - obs[o].x0 ) ) 
                    saddlePoints_exit.append(findBoundaryPoint(obs[o], obs[obs[o].parent].saddle_exit - obs[o].x0 ) )
                else: # hirarchy == 0
                    saddlePoints_front.append(findBoundaryPoint(obs[o], -ds_init(obs[o].x0) ) )
                    saddlePoints_back.append(findBoundaryPoint(obs[o], ds_init(obs[o].x0)))

                if len(obs[o].children):
                    hirarchyList.append(hirarchyList[-1]+1)
                continue

    return [], []

#options = [0]
#for option in options:
#    if option==0:

xlim = [-1,14]
ylim = [-2,7]

xAttractor = [0,0]

N_points = 40                   
saveFigures = True 

# globally_stable_DS(x, x0=[0,0]):

# Linear system and circular obstacle
obs = []

a=[1, 4]
p=[2,2]
x0=[6,0]
th_r=90/180*pi
sf=1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

a=[1, 3]
p=[2,2]
x0=[8+.8,1.5]
th_r=0/180*pi
sf=1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

a=[1, 3]
p=[2,2]
x0=[4-0.8,1.5]
th_r=0/180*pi
sf=1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

saddlePoints, attrPoints = saddlePointsForConcave(obs, xAttractor)

Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinearSys_avoidanceEllipse', noTicks=True, figureSize=(12,9) ) 
