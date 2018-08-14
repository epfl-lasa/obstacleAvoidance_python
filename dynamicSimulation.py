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

# 3D Animatcoion utils
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

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

from draw_ellipsoid import *
from lib_obstacleAvoidance import obs_check_collision
from class_obstacle import *
from lib_modulation import *
from obs_common_section import *
#from obs_dynamic_center import *
from obs_dynamic_center_3d import *

from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

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

        #self.simuColors=[]

        
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

        self.x_pos = np.zeros((self.dim, self.N_simuMax+2, self.N_points))
        
        self.x_pos[:,0,:] = x0
        
        self.xd_ds = np.zeros(( self.dim, self.N_simuMax+1, self.N_points ))
        #self.t = np.linspace(( 0, self.N_simuMax*self.dt, num=self.N_simuMax ))
        self.t = np.linspace(0,self.N_simuMax+1,num=self.N_simuMax+1)*dt

        self.converged = False
    
        self.iSim = 0

        self.lines = [] # Container to keep line plots
        self.startPoints = [] # Container to keep line plots
        self.endPoints = [] # Container to keep line plots        
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
        #self.fig.set_size_inches(14, 9)
        self.fig.set_size_inches(12, 8)
        
        self.ax.set_xlim(xRange)
        self.ax.set_ylim(yRange)
        #self.ax.set_xlabel('x1')
        #self.ax.set_ylabel('x2')
        if self.dim==3:
            self.ax.set_zlim(zRange)
            self.ax.set_zlabel('x3')
            #self.ax.view_init(elev=0.3, azim=0.4)

        # Set axis etc.
        plt.gca().set_aspect('equal', adjustable='box')

        # Set up plot
        #self.setup_plot()
        #self.tt1 = self.ax.text(.5, 1.05, '', transform = self.ax.transAxes, va='center', animated=True, )
        
        # Adjust dynamic center
        #intersection_obs = obs_common_section(self.obs)
        #dynamic_center(self.obs, intersection_obs)
        #dynamic_center_3d(self.obs, intersection_obs)
        
        # Then setup FuncAnimation        
        self.ani = FuncAnimation(self.fig, self.update, interval=1, frames = self.N_simuMax-2, repeat=False, init_func=self.setup_plot, blit=True, save_count=self.N_simuMax-2)

    def setup_plot(self):
        print('setup started')
        # Draw obstacle
        self.obs_polygon = []
        
        # Numerical hull of ellipsoid
        for n in range(len(self.obs)):
            self.obs[n].draw_ellipsoid(numPoints=50) # 50 points resolution

        for n in range(len(self.obs)):
            if self.dim==2:
                emptyList = [[0,0] for i in range(50)]
                #self.obs_polygon.append( plt.Polygon(self.obs[n].x_obs, animated=True,))
                self.obs_polygon.append( plt.Polygon(emptyList, animated=True,))
                self.obs_polygon[n].set_color(np.array([176,124,124])/255)
                self.obs_polygon[n].set_alpha(0.8)
                patch_o = plt.gca().add_patch(self.obs_polygon[n])
                self.patches.append(patch_o)

                if self.obs[n].x_end > 0:
                    cont, = plt.plot([],[],  'k--', animated=True)
                else:
                    cont, = plt.plot([self.obs[n].x_obs_sf[ii][0] for ii in range(len(self.obs[n].x_obs_sf))],
                                     [self.obs[n].x_obs_sf[ii][1] for ii in range(len(self.obs[n].x_obs_sf))],
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
            line, = plt.plot([], [], '--', lineWidth = 4, animated=True)
            self.lines.append(line)
            point, = plt.plot(self.x_pos[0,0,ii],self.x_pos[1,0,ii], '*k', markersize=10, animated=True)
            if self.dim==3:
                point, = plt.plot(self.x_pos[0,0,ii],self.x_pos[1,0,ii], self.x_pos[2,0,ii], '*k', markersize=10, animated=True)
            self.startPoints.append(point)
            point, = plt.plot([], [], 'bo', markersize=15, animated=True)
            self.endPoints.append(point)


        if self.dim==2:
            plt.plot(self.attractorPos[0], self.attractorPos[1], 'k*', linewidth=7.0)
        else:
            plt.plot([self.attractorPos[0]], [self.attractorPos[1]], [self.attractorPos[2]], 'k*', linewidth=7.0)

        self.fig.canvas.mpl_connect('button_press_event', onClick)  # Button click enabled

        self.tt1 = self.ax.text(.5, 8.2, '', va='center', fontsize=20)

        print('setup finished')

        return (self.lines + self.obs_polygon + self.contour + self.centers + self.cent_dyns + self.startPoints + self.endPoints + [self.tt1])
    
    def update(self, iSim):
        #if saveFigure:
        print('iteration num {} -- frame = {}'.format(self.iSim, iSim))
        if pause:        # NO ANIMATION -- PAUSE
            self.old_time=time.time()
            return (self.lines + self.obs_polygon + self.contour + self.centers + self.cent_dyns + self.startPoints + self.endPoints)

        #intersection_obs = obs_common_section(self.obs)
        #dynamic_center_3d(self.obs, intersection_obs)

        RK4_int = True
        if RK4_int: # Runge kutta integration
            for j in range(self.N_points):
                self.x_pos[:, self.iSim+1,j] = obs_avoidance_rk4(self.dt, self.x_pos[:,self.iSim,j], self.obs, x0=self.attractorPos)
        else: # Simple euler integration
            # Calculate DS
            for j in range(self.N_points):
                xd_temp = linearAttractor_const(self.x_pos[:,self.iSim, j], self.attractorPos, velConst=1.6, distSlow=0.01)
                #self.xd_ds[:,self.iSim,j] = obs_avoidance_convergence(self.x_pos[:,self.iSim, j], xd_temp, self.obs)
                self.xd_ds[:,self.iSim,j] = obs_avoidance_interpolation(self.x_pos[:,self.iSim, j], xd_temp, self.obs)
                #self.xd_ds[:,self.iSim,j] = obs_avoidance_interpolation_bad(self.x_pos[:,self.iSim, j], xd_temp, self.obs)
                self.x_pos[:,self.iSim+1,:] = self.x_pos[:,self.iSim, :] + self.xd_ds[:,self.iSim, :]*self.dt
        
        self.t[self.iSim+1] = (self.iSim+1)*self.dt

        # Update lines
        for j in range(self.N_points):
            self.lines[j].set_xdata(self.x_pos[0,:self.iSim+1,j])
            self.lines[j].set_ydata(self.x_pos[1,:self.iSim+1,j])
            if self.dim==3:
                self.lines[j].set_3d_properties(zs=self.x_pos[2,:self.iSim+1,j])

            self.endPoints[j].set_xdata(self.x_pos[0,self.iSim+1,j])
            self.endPoints[j].set_ydata(self.x_pos[1,self.iSim+1,j])
            if self.dim==3:
                self.endPoints[j].set_3d_properties(zs=self.x_pos[2,self.iSim+1,j])
        
        # ========= Check collision ----------
        #collisions = obs_check_collision(self.x_pos[:,self.iSim+1,:], obs)
        #collPoints = np.array()

        #print('TODO --- collision observation')
        #collPoints = self.x_pos[:,self.iSim+1,collisions]

        # if collPoints.shape[0] > 0:
        #     plot(collPoints[0,:], collPoints[1,:], 'rx')
        #     print('Collision detected!!!!')
        for o in range(len(self.obs)):# update obstacles if moving
            self.obs[o].update_pos(self.t[self.iSim], self.dt) # Update obstacles

            self.centers[o].set_xdata(self.obs[o].x0[0])
            self.centers[o].set_ydata(self.obs[o].x0[1])
            if self.dim==3:
                self.centers[o].set_3d_properties(zs=obs[o].x0[2])

            if hasattr(self.obs[o], 'center_dyn'):# automatic adaptation of center
                self.cent_dyns[o].set_xdata(self.obs[o].center_dyn[0])
                self.cent_dyns[o].set_ydata(self.obs[o].center_dyn[1])
                if self.dim==3:
                    self.cent_dyns[o].set_3d_properties(zs=self.obs[o].center_dyn[2])


            if self.obs[o].x_end > self.t[self.iSim] or self.iSim<1: # First round or moving
                if self.dim ==2: # only show safety-contour in 2d, otherwise not easily understandable
                    self.contour[o].set_xdata([self.obs[o].x_obs_sf[ii][0] for ii in range(len(self.obs[o].x_obs_sf))])
                    self.contour[o].set_ydata([self.obs[o].x_obs_sf[ii][1] for ii in range(len(self.obs[o].x_obs_sf))])

                if self.dim==2:
                    self.obs_polygon[o].xy = self.obs[o].x_obs
                else:
                    self.obs_polygon[o].xyz = self.obs[o].x_obs
        self.iSim += 1 # update simulation counter
        self.check_convergence() # Check convergence 
        
        # Pause for constant simulation speed
        self.old_time = self.sleep_const(self.old_time)
        self.pause_time = self.old_time

        self.tt1.set_text('{:2.2f} s'.format(round(self.t[self.iSim+1],2) ) )

        return (self.lines + self.obs_polygon + self.contour + self.centers + self.cent_dyns + self.startPoints + self.endPoints + [self.tt1] )

    def check_convergence(self):
        #return
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

