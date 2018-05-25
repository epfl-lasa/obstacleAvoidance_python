'''
Dynamic Simulation - Obstacle Avoidance Algorithm

@author LukasHuber
@date 2018-05-24

'''

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np


# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers

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


print('Script started ')


obs = []
n = 0
a = [3,1]
p = [1,1]
x0 = [4,0]
th_r = 30/180*pi
sf = 1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))


N_points = len(obs[0].x0)

fig, ax = plt.subplots()
#fig.set_size_inches(14.40, 10.80)
fig.set_size_inches(12, 9)


#return fig, ax lines

#def dynamicSimulation(dt, arg1, arg2):
    
#x0, xT,  obs, tolerance=0.01, N_simuMax=600, dt=0.02):

class AnimatedScatter():
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, x0, obs=[], numpoints=50):
            N_simuMax = 600

        self.dim = x0.shape[0]

        self.N_points = len(x0)

        self.x_pos = np.zeros((dim, N_simuMax, N_points))
        self.xd_ds = np.zeros((dim, N_simuMax, N_points))
        self.t = []
        self.t.append(t[-1]+dt)

        self.converged = False
    
        self.iSim = 0

        self.numpoints = numpoints
        
        self.lines = []

        # Setup the figure and axes.
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):

        for ii in range(N_points):
            line, = plt.plot([], [], animated=True)
            self.lines.append(line)
            
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.lines[0], self.lines[1],

    def data_stream(self):
        return 0

    def update(self, dt):
        # update figure
        # TODO
        # update figure

        # Calculate DS
        for j in range(N_points):
            xd_temp = linearAttractor(self.x_pos[:,iSim, j])
            self.xd_ds[:,iSim,j] = obs_avoidance_convergence(x_pos[:,iSim, j], xd_temp, obs)

        # Integration
        # TODO - second order?
        self.x_pos[:,iSim+1,: ] = self.x_pos[:,iSim, j] + self.xd_ds[:,iSim,j]*dt
        self.t[iSim+1] = (iSim+1)*dt

        for j in range(self.N_points):
            self.lines[j].set_xdata(x_pos[:,iSim,:j+1])
            self.lines[j].set_ydata(x_pos[:,iSim,:j+1])

        # Check collision

        # Check convergence
        if self.converged:
            break

        self.iSim = self.iSim + 1

    def show(self):
        plt.show()

        
#if __name__ == '__main__':
    #a = AnimatedScatter()
#    a.show()



#######

x0 = np.array(

a = AnimatedScatter()
a.show()


print('Script finished ')
