# Plot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap # Create custom color maps
import warnings

# Math packages
from math import sqrt
import cmath
import numpy as np

import sys # Environment variables

base_pwd = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/"

lib_string = base_pwd
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

lib_string = base_pwd + "lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

lib_string = base_pwd + "Analytic/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)
    
from class_obstacle import *
from lib_obstacleAvoidance import obs_check_collision
from lib_contractionAnalysis import *
from dynamicalSystem_lib import *
from lib_obstacleAvoidance import *

simuNumb = 1
saveFig = False

N_heatmap = 101 # Size grid

## Define obstacles --------------------
## Zeros Determinant
# clear all
# # 
if simuNumb == 0:
    # # Place obstacles
    obs = []
    n_obs = 0
    a = [2,2]
    p = [1,1]
    x0 = [8,0.0]
    sf = 1.0
    th_r = 0*pi/180
    x_center = [0.0,0.0]
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
    simulationName = "circleAt8"
#

if simuNumb == 1:
    # # Place obstacles
    obs = []
    n_obs = 0
    a = [1,4]
    p = [1,1]
    x0 = [8,0.0]
    sf = 1.0
    th_r = 60*pi/180
    x_center = [0.0,0.0]
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

    simulationName = "ellipsoidal"


# Place obstacles
if simuNumb == 2:
    obs = []
    n_obs = 0
    a = [1,4]
    p = [1,1]
    x0 = [8,0.0]
    sf = 1.0
    th_r = 0*pi/180
    x_center = [0.0,0.0]
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

    simulationName = "ellipseVertical"

## Simulation Parameters

xAttractor = np.array([0,0])

### Algorithm

N_x = N_heatmap
N_y = N_x 

d = len(obs[0].x0) # Dimensions

rho = 1

fn_handle_objAvoidance = lambda x,xd,obs: obs_modulation_convergence(x,xd,obs)
                          

x_range = [-2,15]
y_range = [-8.8,8.8]
    
xValues = np.linspace(x_range[0],x_range[1],N_x)
yValues = np.linspace(y_range[0],y_range[1],N_y)

d1 = obs[n_obs].x0[0]
if obs[n_obs].x0[1]:
    warnings.warning("Attentioni - x2 \neq 0 ")

tol = 5*1e-3 # Tolerance for numerical zero valuation

# Initialize variables
poles = []
zerosDet = []
zerosTrace = [] 

# Value matrices
denominator = np.zeros((N_x,N_y))
determinantNominator = np.zeros((N_x,N_y))
traceNominator = np.zeros((N_x,N_y))
trace = np.zeros((N_x,N_y))
determinant = np.zeros((N_x,N_y))
eigValues = np.zeros((2,N_x, N_y))

YY, XX = np.mgrid[y_range[0]:y_range[1]:N_y*1j, x_range[0]:x_range[1]:N_x*1j]
xd_init = np.zeros((d, N_x, N_y))
xd_IFD = np.zeros((d, N_x, N_y))


for ix in range(N_x):
    x1 = xValues[ix]
    for iy in range(N_y):
        x2 = yValues[iy]
        x_t = np.array(obs[n_obs].rotMatrix) @ (np.array([x1,x2])-np.array(obs[n_obs].x0))

        Gamma = sum((x_t / np.array(obs[n_obs].a)) ** (2*np.array(obs[n_obs].p) ) )
        nv = (2*np.array(obs[n_obs].p) / np.array(obs[n_obs].a) * (x_t / obs[n_obs].a)**(2*np.array(obs[n_obs].p) - 1)) #normal vector of the tangential hyper-plane

        #x_t = np.array(obs[n_obs].rotMatrix).T @ nv
        
        nv = np.array(obs[n_obs].rotMatrix).T @ nv
        #nv_hat = np.array([x1,x2]) - xAttractor
        
        t1 = nv[1]
        t2 = -nv[0]
        
        D = np.eye(d)+np.diag(([-1,1]*1/abs(Gamma) ** (1/rho) ))
        l_n = D[0,0]
        l_t = D[1,1]
        
#         l_n = 1-1/(Gamma)
#         l_t = 1+1/(Gamma)
        
        # Jacobian Originalo
        determinant[ix, iy], trace[ix,iy] = contraction_det_trace(x1, x2, l_n, l_t, t1, t2, d1)

        sqrtDet = cmath.sqrt(trace[ix,iy]**2/4 - determinant[ix,iy])
        
        eigValues[0,ix,iy] = trace[ix,iy]/2 - sqrtDet.real
        eigValues[1,ix,iy] = trace[ix,iy]/2 + sqrtDet.real

        pos = np.array([XX[ix,iy],YY[ix,iy]])
            
        xd_init[:,ix,iy] = linearAttractor(pos, x0 = xAttractor ) # initial DS

        xd_IFD[:,ix,iy] = IFD(pos, xd_init[:,ix,iy],obs) # modulataed DS with IFD
        
        
# -------------------- Create colormap --------------------

# cdict = {'red':  ((0.0, 0.0, 0.0),
#                   (0.495, 0.8, 1.0),
#                   (0.505, 1.0, 1.0),
#                   (1.0, 0.4, 0.4)),

#          'green': ((0.0, 0.0, 0.0),
#                   (0.495, 0.8, 1.0),
#                    (0.505, 1.0, 0.6),
#                    (1.0, 0.0, 0.0)),

#          'blue':  ((0.0, 0.4, 0.4),
#                    (0.495, 1.0, 1.0),
#                    (0.505, 1.0, 0.8),
#                    (1.0, 0.0, 0.0))
#         }

cdict = {'red':  ((0.0, 0.61, 0.61),
                  (0.495, 1.0, 1.0),
                  (0.505, 1.0, 0.78),
                  (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                  (0.495, 0.78, 1.0),
                   (0.505, 1.0, 1.0),
                   (1.0, 0.61, 0.61)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.495, 0.78, 1.0),
                   (0.505, 1.0, 0.78),
                   (1.0, 0.0, 0.0))
        } 

blue_red = LinearSegmentedColormap('BlueRed', cdict)

# blue = [-lambda0*t2*x1
#         -lambda0*t2*x1]
#green = [0,128,0
#        144,238,144]/255  
green = np.array([[200,255,200],
                  [0,155,0]])/255
    
red = np.array([[155,0,0],
                [255,200,200]])/255
#map_4cols = four_colors(N, green, red) # Create desired coloring

# TODO add margin to colliision matrix
collisions = obs_check_collision(obs, XX, YY)
dx1_noColl = np.squeeze(xd_IFD[0,:,:]) * collisions
dx2_noColl = np.squeeze(xd_IFD[1,:,:]) * collisions

colRange = [-2.2,2.2]
colTicks = [-2,2]

# Vector field
#for in range (0,5):
figs = []
axes = []
axisPlot = (x_range[0],x_range[1],y_range[0],y_range[1])

for ii in range (0,4):
    #plt.figure(ii+1)
    fig_ii, ax_ii = plt.subplots()
    figs.append(fig_ii)
    axes.append(ax_ii)
    
    if ii==0:
        plt.imshow(trace.T, interpolation='nearest', cmap=blue_red,  extent=axisPlot, vmin=colRange[0], vmax=colRange[-1])
        measureName = "trace"
    elif ii==1:
        plt.imshow(determinant.T, interpolation='nearest', cmap=blue_red,  extent=axisPlot, vmin=colRange[0], vmax=colRange[-1])
        measureName = "determinant"
    elif ii==2:
        plt.imshow(np.squeeze(eigValues[0,:,:]).T, interpolation='nearest', cmap=blue_red,  extent=axisPlot, vmin=colRange[0], vmax=colRange[-1])
        measureName = "eigValue0"
    elif ii==3:
        plt.imshow(np.squeeze(eigValues[1,:,:]).T, interpolation='nearest', cmap=blue_red,  extent=axisPlot, vmin=colRange[0], vmax=colRange[-1])
        measureName = "eigValue1"
    # elif ii == 4:
    #     plt.imshow(trace.T, extent=extent)
    #     #imagesc(xValues, yValues, denominator.T) hold on # Create plot
    #     caxis(2*[-1,1]) # set axis range
    #     measureName = "denominator"

    figs[ii].canvas.set_window_title("Figure - " + measureName)o
    cbar = plt.colorbar()
    cbar.set_ticks([colTicks[0], 0, colTicks[-1]])
    cbar.set_ticklabels([colTicks[0], 0, colTicks[-1]])

    # Draw vector field
    res_ifd = axes[ii].streamplot(XX, YY,dx1_noColl, dx2_noColl, color='k')

    for n in range(len(obs)):
        obs[n].draw_ellipsoid() # 50 points resolution
        
    # Draw obstacles
    obs_polygon = []

    #obs_alternative = obs_draw_ellipsoid()
    for n in range(len(obs)):
        x_obs_sf = obs[n].x_obs # todo include in obs_draw_ellipsoid
        obs_polygon.append( plt.Polygon(obs[n].x_obs))
        patchObs = plt.gca().add_patch(obs_polygon[n])
        patchObs.set_facecolor('gray')
        patchObs.set_alpha(0.8)
        
        plt.plot([x_obs_sf[i][0] for i in range(len(x_obs_sf))],
                 [x_obs_sf[i][1] for i in range(len(x_obs_sf))], 'k', linewidth=3)
        
        if hasattr(obs[n], 'center_dyn'):# automatic adaptation of center 
            axes[ii].plot(obs[n].center_dyn[0],obs[n].center_dyn[1], 'k+')
        else:
            axes[ii].plot(obs[n].x0[0],obs[n].x0[1],'k+')
        # Draw obstacles
        
    plt.gca().set_aspect('equal', adjustable='box')

    axes[ii].set_xlim(x_range)
    axes[ii].set_ylim(y_range)
    #axes[ii].set_xlim([(x_range[i]-np.mean(x_range))*1.1+np.mean(x_range) for i in range(2)])
    #axes[ii].set_ylim([(y_range[i]-np.mean(y_range))*1.1+np.mean(y_range) for i in range(2)])

    plt.xlabel(r'$\xi_1$')
    plt.ylabel(r'$\xi_2$')

    if saveFig:
        plt.savefig(base_pwd + "fig/analysisContraction_" + simulationName +  "_" + measureName + ".pdf" )

plt.ion()        
plt.show()
#fprintf("Numerical evaluation took #d seconds. \n", t_numericalZero)
