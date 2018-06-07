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
#from matplotlib.animation import writers

import time

from math import pi

#first change the cwd to the script path
#scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
#os.chdir(scriptPath)

import sys
 
# ---------- Import Custom libraries ----------
from dynamicalSystem_lib import *

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


# ----------  Start script ----------
print()
print(' ----- Script <<dynamic simulation>> started. ----- ')
print()

pause=False
def onClick(event): 
    global pause
    pause ^= True

class Animated():
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, x0, obs=[], N_simuMax = 600, dt=0.01, attractorPos=[0,0], convergenceMargin=0.01, xRange=[-1,1], yRange=[-1,1], sleepPeriod=0.03):

        # Initialize class variables
        self.obs = obs
        self.N_simuMax = N_simuMax
        self.dt = dt
        self.attractorPos = attractorPos
        self.sleepPeriod=sleepPeriod

        # last three values are observed for convergence
        self.convergenceMargin = convergenceMargin
        self.lastConvergences = [convergenceMargin for i in range(3)] 

        # Get current simulation time
        self.old_time = time.time()
        
        self.dim = x0.shape[0]
        self.N_points = x0.shape[1]

        self.x_pos = np.zeros((self.dim, self.N_simuMax, self.N_points))
        
        self.x_pos[:,0,:] = x0
        
        self.xd_ds = np.zeros(( self.dim, self.N_simuMax, self.N_points ))
        #self.t = np.linspace(( 0, self.N_simuMax*self.dt, num=self.N_simuMax ))
        self.t = np.linspace(0,self.N_simuMax,num=self.N_simuMax)

        self.converged = False
    
        self.iSim = 0

        self.lines = []

        # Setup the figure and axes.
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(12, 9)
        
        self.ax.set_xlim(xRange)
        self.ax.set_ylim(yRange)
        # Set axis etc.
        plt.gca().set_aspect('equal', adjustable='box')
            

        # Set up plot
        self.setup_plot()

        self.fig.canvas.mpl_connect('button_press_event', onClick)  # Button click enabled

        # Adjust dynamic center
        intersection_obs = obs_common_section(self.obs)
        dynamic_center(self.obs, intersection_obs)
        
        # Then setup FuncAnimation.
        self.ani = FuncAnimation(self.fig, self.update, interval=1, 
                                           init_func=self.setup_plot, blit=True)

        self.itCount = 0

        
    def setup_plot(self):
        # Draw obstacles
        obs_polygon = []

        # Numerical hull of ellipsoid
        for n in range(len(obs)):
            obs[n].draw_ellipsoid() # 50 points resolution
            
        for n in range(len(self.obs)):
            x_obs_sf = self.obs[n].x_obs # todo include in obs_draw_ellipsoid
            
            obs_polygon.append( plt.Polygon(self.obs[n].x_obs))
            plt.gca().add_patch(obs_polygon[n])

            #x_obs_sf_list = x_obs_sf[:,:,n].T.tolist()
            plt.plot([x_obs_sf[i][0] for i in range(len(x_obs_sf))],
                     [x_obs_sf[i][1] for i in range(len(x_obs_sf))], 'k--')

            self.ax.plot(self.obs[n].x0[0],self.obs[n].x0[1],'k.')
            
            if hasattr(self.obs[n], 'center_dyn'):# automatic adaptation of center 
                self.ax.plot(self.obs[n].center_dyn[0],obs[n].center_dyn[1], 'k+')
                
        for ii in range(self.N_points):
            line, = plt.plot([], [], animated=True)
            self.lines.append(line
)
        # Draw attractor
        if type(self.attractorPos) != str:
            plt.plot(self.attractorPos[0], self.attractorPos[1], 'k*', linewidth=7.0)


        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        #return self.lines[0], self.lines[1]
        return self.lines
    
    def update(self, dt):
        # TODO
        # update figure
        if pause:
            self.old_time=time.time()
            return self.lines
        
        # Calculate DS
        #print('')
        #print('')
        #print('update start')
        #print('iSim{}'.format(self.iSim))
        
        for j in range(self.N_points):
            xd_temp = linearAttractor(self.x_pos[:,self.iSim, j])

            #print('DS init:', xd_temp)
            #print('pos X', self.x_pos[:,self.iSim,j])
            #print('DS modu:', obs_avoidance_convergence(self.x_pos[:,self.iSim, j], xd_temp, self.obs))
            
            self.xd_ds[:,self.iSim,j] = obs_avoidance_convergence(self.x_pos[:,self.iSim, j], xd_temp, self.obs)

        # Integration
        # TODO - second order?
        
        self.x_pos[:,self.iSim+1,:] = self.x_pos[:,self.iSim, :] + self.xd_ds[:,self.iSim, :]*self.dt
        self.t[self.iSim+1] = (self.iSim+1)*self.dt

        for j in range(self.N_points):
            # self.lines[j].set_xdata(self.x_pos[:,self.iSim,:j+1])
            # self.lines[j].set_ydata(self.x_pos[:,self.iSim,:j+1])
            self.lines[j].set_xdata(self.x_pos[0,:self.iSim+1,j])
            self.lines[j].set_ydata(self.x_pos[1,:self.iSim+1,j])

            #print('xPos1:',self.x_pos[0,:self.iSim+1,:j])
            #print('xPos2:',self.x_pos[1,:self.iSim+1,:j])
            
        # Check collision

        # Check convergence
        #if self.converged:
        # break

        self.iSim = self.iSim + 1

        self.itCount += 1
        
        #print('update finished')

        # Pause for constant simulation speed
        self.old_time = self.sleep_const(self.old_time)
        
        self.check_convergence()
        
        #return self.lines[0], self.lines[1]
        return self.lines

    def check_convergence(self):
        self.lastConvergences[0] = self.lastConvergences[1]
        self.lastConvergences[1] = self.lastConvergences[2]

        self.lastConvergences[2] =  np.sum(abs(self.x_pos[:,self.iSim,:] - np.tile(self.attractorPos, (self.N_points,1) ).T ))

        if (sum(self.lastConvergences) < self.convergenceMargin) or (self.iSim+1>=self.N_simuMax):
            self.ani.event_source.stop()
            
            if (self.iSim>=self.N_simuMax-1):
                print('Maximum number of {} iterations reached without convergence.'.format(self.N_simuMax))
            else:
                print('Convergence with tolerance of {} reached after {} iterations.'.format(sum(self.lastConvergences), self.iSim+1) )
        
    def show(self):
        plt.show()

    def sleep_const(self,  old_time=0):
        next_time = old_time+self.sleepPeriod
        
        now = time.time()
        
        sleep_time = next_time - now # get sleep time
        sleep_time = min(max(sleep_time, 0), self.sleepPeriod) # restrict in sensible range

        time.sleep(sleep_time)

        return next_time

    

#if __name__ == '__main__':
    #a = AnimatedScatter()
#    a.show()

#### Create starting points

N = 100
x_init = np.vstack((np.ones(N)*20,
           np.linspace(-15,15,num=N)))


### Create obstacle 
obs = []

a = [5,2]
p = [1,1]
x0 = [13.5,2]
th_r = 30/180*pi
sf = 1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

a = [7,2]
p = [1,1]
x0 = [7,-2]
th_r = -40/180*pi
sf = 1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

xRange = [-1,20]
yRange = [-10,10]

#if __name__ == '__main__':
if True:
    a = Animated(x_init, obs, xRange=xRange, yRange=yRange, dt=0.01, N_simuMax=600, convergenceMargin=0.3, sleepPeriod=0.03 )
    a.show()


print()
print('---- Script finished ---- ')
print() # THE END
