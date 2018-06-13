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

# 3D Animation utils
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
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
from lib_modulation import *
from obs_common_section import *
from obs_dynamic_center import *


# --------------  Start script --------------
print()
print(' ----- Script <<dynamic simulation>> started. ----- ')
print()

###### Create function to allow pause (one click) and stop (double click) on figure #####
pause=False
pause_start = 0
def onClick(event): 
    global pause
    global pause_start
    global anim
    
    pause ^= True
    if pause:
        pause_start = time.time()
    else:
        dT = time.time()-pause_start
        if dT < 0.3: # Break simulation at double click
            print('Animation exited.')
            anim.ani.event_source.stop()

            
##### Anmation Function #####
class Animated():
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, x0, obs=[], N_simuMax = 600, dt=0.01, attractorPos='default', convergenceMargin=0.01, xRange=[-10,10], yRange=[-10,10], zRange=[-10,10], sleepPeriod=0.03):

        self.dim = x0.shape[0]
        
        # Initialize class variables
        self.obs = obs
        self.N_simuMax = N_simuMax
        self.dt = dt
        if attractorPos == 'default':
            self.attractorPos = self.dim*[0.0]
        else:
            self.attractorPos = attractorPos
            
        self.sleepPeriod=sleepPeriod

        # last three values are observed for convergence
        self.convergenceMargin = convergenceMargin
        self.lastConvergences = [convergenceMargin for i in range(3)] 

        # Get current simulation time
        self.old_time = time.time()
        self.pause_time = self.old_time
        
        self.N_points = x0.shape[1]

        self.x_pos = np.zeros((self.dim, self.N_simuMax, self.N_points))
        
        self.x_pos[:,0,:] = x0
        
        self.xd_ds = np.zeros(( self.dim, self.N_simuMax, self.N_points ))
        #self.t = np.linspace(( 0, self.N_simuMax*self.dt, num=self.N_simuMax ))
        self.t = np.linspace(0,self.N_simuMax,num=self.N_simuMax)

        self.converged = False
    
        self.iSim = 0

        self.lines = [] # Container to keep line plots
        self.patches = [] # Container to keep patch plotes
        self.contour = []
        self.centers = []
        self.cent_dyns = []

        # Setup the figure and axes.
        if self.dim==2:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.set_size_inches(12, 9)
        
        self.ax.set_xlim(xRange)
        self.ax.set_ylim(yRange)
        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')
        if self.dim==3:
            self.ax.set_zlim(zRange)
            self.ax.set_zlabel('x3')
            #self.ax.view_init(elev=0.3, azim=0.4)
        
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

    def setup_plot(self):
        print('setup started')
        # Draw obstacle
        self.obs_polygon = []
        
        # Numerical hull of ellipsoid
        for n in range(len(obs)):
            obs[n].draw_ellipsoid() # 50 points resolution

        for n in range(len(self.obs)):
            if self.dim==2:
                self.obs_polygon.append( plt.Polygon(self.obs[n].x_obs, animated=True,))
                self.obs_polygon[n].set_color(np.array([176,124,124])/255)
                self.obs_polygon[n].set_alpha(0.3)
                patch_o = plt.gca().add_patch(self.obs_polygon[n])
                self.patches.append(patch_o)

                if self.obs[n].x_end > 0:
                    cont, = plt.plot([],[],  'k--', animated=True)
                else:
                    cont, = plt.plot([self.obs[n].x_obs_sf[0][ii] for ii in range(len(self.obs[n].x_obs_sf[0]))],
                                     [self.obs[n].x_obs_sf[1][ii] for ii in range(len(self.obs[n].x_obs_sf[0]))],
                                     'k--', animated=True)
                self.contour.append(cont)
            else:
                N_resol=50 # TODO  save internally from assigining....
                self.obs_polygon.append(
                    self.ax.plot_surface(
                        np.reshape([obs[n].x_obs[i][0] for i in range(len(obs[n].x_obs))],
                                   (N_resol,-1)),
                        np.reshape([obs[n].x_obs[i][1] for i in range(len(obs[n].x_obs))],
                                   (N_resol,-1)),
                        np.reshape([obs[n].x_obs[i][2] for i in range(len(obs[n].x_obs))],
                                   (N_resol, -1))  )  )

            # Center of obstacle
            center, = self.ax.plot([],[],'k.', animated=True)    
            self.centers.append(center)
            
            if hasattr(self.obs[n], 'center_dyn'):# automatic adaptation of center
                cent_dyn, = self.ax.plot([],[], 'k+', animated=True)
                self.cent_dyns.append(cent_dyn)
                
        for ii in range(self.N_points):
            line, = plt.plot([], [], animated=True)
            self.lines.append(line)


        if self.dim==2:
            plt.plot(self.attractorPos[0], self.attractorPos[1], 'k*', linewidth=7.0)
        else:
            plt.plot([self.attractorPos[0]], [self.attractorPos[1]], [self.attractorPos[2]], 'k*', linewidth=7.0)

        return (self.lines + self.obs_polygon + self.contour + self.centers + self.cent_dyns)
    
    def update(self, dt):
        if pause:        # NO ANIMATION -- PAUSE
            self.old_time=time.time()
            return (self.lines + self.obs_polygon + self.contour + self.centers + self.cent_dyns)

        intersection_obs = obs_common_section(self.obs)
        dynamic_center(self.obs, intersection_obs)

        # Calculate DS
        for j in range(self.N_points):
            xd_temp = linearAttractor(self.x_pos[:,self.iSim, j], self.attractorPos)
            self.xd_ds[:,self.iSim,j] = obs_avoidance_convergence(self.x_pos[:,self.iSim, j], xd_temp, self.obs)
        self.x_pos[:,self.iSim+1,:] = self.x_pos[:,self.iSim, :] + self.xd_ds[:,self.iSim, :]*self.dt
        self.t[self.iSim+1] = (self.iSim+1)*self.dt

        # Update lines
        for j in range(self.N_points):
            self.lines[j].set_xdata(self.x_pos[0,:self.iSim+1,j])
            self.lines[j].set_ydata(self.x_pos[1,:self.iSim+1,j])
            if self.dim==3:
                self.lines[j].set_3d_properties(zs=self.x_pos[2,:self.iSim+1,j])


        
        # ========= Check collision ----------
        #collisions = obs_check_collision(self.x_pos[:,self.iSim+1,:], obs)
        #collPoints = np.array()

        #print('TODO --- collision observation')
        #collPoints = self.x_pos[:,self.iSim+1,collisions]

        # if collPoints.shape[0] > 0:
        #     plot(collPoints[0,:], collPoints[1,:], 'rx')
        #     print('Collision detected!!!!')
        for o in range(len(obs)):# update obstacles if moving
            obs[o].update_pos(self.t[self.iSim], self.dt) # Update obstacles

            self.centers[o].set_xdata(obs[o].x0[0])
            self.centers[o].set_ydata(obs[o].x0[1])
            if self.dim==3:
                self.centers[o].set_3d_properties(zs=obs[o].x0[2])

            if hasattr(self.obs[o], 'center_dyn'):# automatic adaptation of center
                self.cent_dyns[o].set_xdata(obs[o].center_dyn[0])
                self.cent_dyns[o].set_ydata(obs[o].center_dyn[1])
                if self.dim==3:
                    self.cent_dyns[o].set_3d_properties(zs=obs[o].center_dyn[2])

                    
            if obs[o].x_end > self.t[self.iSim]:
                if self.dim ==2: # only show safety-contour in 2d, otherwise not easily understandable
                    self.contour[o].set_xdata([self.obs[o].x_obs_sf[0][ii] for ii in range(len(self.obs[o].x_obs_sf[0]))])
                    self.contour[o].set_ydata([self.obs[o].x_obs_sf[1][ii] for ii in range(len(self.obs[o].x_obs_sf[1]))])

                if self.dim==2:
                    self.obs_polygon[o].xy = self.obs[o].x_obs
                else:
                    self.obs_polygon[o].xyz = self.obs[o].x_obs
        self.iSim += 1 # update simulation counter
        self.check_convergence() # Check convergence 
        
        # Pause for constant simulation speed
        self.old_time = self.sleep_const(self.old_time)
        self.pause_time = self.old_time

        return (self.lines + self.obs_polygon + self.contour + self.centers + self.cent_dyns)

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
N = 3

simuCase=1
if simuCase==0:
    ### Create obstacle 
    obs = []

    a = [5, 2]
    p = [1, 1]
    x0 = [13.5, 2]
    th_r = 30/180*pi
    sf = 1.

    xd=[0, 0]
    w = 0

    x_start = 0
    x_end = 2
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))

    a = [7,2]
    p = [1,1]
    x0 = [7,-2]
    th_r = -40/180*pi
    sf = 1.

    xRange = [-1,20]
    yRange = [-10,10]
    zRange = [-10,10]
    #obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
    
elif simuCase==1:
    N = 10
    x_init = np.vstack((np.ones(N)*1,
                        np.linspace(-1,1,num=N),
                        np.linspace(-1,1,num=N) ))
    ### Create obstacle 
    obs = []

    x0 = [0.5,0.2,0.0]
    a = [0.4,0.1,0.1]
    #a = [4,4,4]
    p = [10,1,1]
    th_r = [0, 0, 30./180*pi]
    sf = 1.

    xd=[0,0,0]
    w = [0,0,0]

    x_start = 0
    x_end = 2
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))

    ### Create obstacle
    x0 = [0.5,-0.2,0]
    a = [0.4,0.1,0.1]
    p = [10,1,1]
    th_r = [0, 0, -30/180*pi]
    sf = 1

    xd=[0,0,0]
    w = [0,0,0]

    x_start = 0
    x_end = 2
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))
    xRange = [-0.2,1.8]
    yRange = [-1,1]
    zRange = [-1,1]


#if __name__ == '__main__':
if True:
    anim = Animated(x_init, obs, xRange=xRange, yRange=yRange, zRange=zRange, dt=0.005, N_simuMax=200000, convergenceMargin=0.3, sleepPeriod=0.01, )
    anim.show()

print()
print('---- Script finished ---- ')
print() # THE END
 
