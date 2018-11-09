'''
Obstacle Avoidance - Visuatlization Algroith

@author lukashuber
@date 2018-02-15
'''
import matplotlib as plt

import numpy as nup
from math import pi

import sys

#from dynamicalSystem_lib import *

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm_python/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm_python/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

def visualisation_vectorField(x_range=[0,10],y_range=[0,10], N_y=10, obs=[], sysDyn_init=False, xAttractor = np.array(([0,0])), safeFigure = False):
    fig = []
    ax = []
    for ii in range(1):
        fig_temp, ax_temp = plt.subplots(figsize=(10,8))
        fig.append(fig_temp)
        ax.append(ax_temp)

    
    ax[0].plt

    print('To continue')
    


