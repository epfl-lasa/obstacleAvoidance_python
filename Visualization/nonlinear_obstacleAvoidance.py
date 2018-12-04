"""
Visualizatoin Radial Displacement -- Nonlinear Obstacle Avoidance

"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


x_range = [0.3,13]
y_range = [-6,6]

xAttractor=[0,0]

N_points=120
#save Figures=True

obs=[]

a = [2,5.0]
p = [1,1]
x0 = [5.5,0]
th_r = 40/180*pi
sf = 1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
obs[-1].center_dyn = np.array([ 3.87541829,  0.89312174])

# fig, ax = plt.figure()
fig, ax = plt.subplots(figsize=[7,6])

for n in range(len(obs)): 
    obs[n].draw_ellipsoid(numPoints=50) # 50 points resolution

gridResol = 10
N_x = gridResol
N_y = gridResol
YY, XX = np.mgrid[y_range[0]:y_range[1]:N_y*1j, x_range[0]:x_range[1]:N_x*1j]

# Initialize systems
xd_init = np.zeros((2,N_x,N_y))

for ix in range(N_x):
    for iy in range(N_y):
        pos = np.array([XX[ix,iy], YY[ix,iy]])
        xd_init[:,ix,iy] = nonlinear_wavy_DS(pos, x0 = xAttractor ) # initial DS
#res_init = ax.streamplot(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[(0.3,0.3,0.3)])
obs_polygon = []
for n in range(len(obs)):
    x_obs_sf = obs[n].x_obs_sf # todo include in obs_draw_ellipsoid
    obs_polygon.append( plt.Polygon(obs[n].x_obs))

    obs_polygon[n].set_color(np.array([176,124,124])/255)
    obs_polygon[n].set_alpha(0.6)
    plt.gca().add_patch(obs_polygon[n])
        
            #x_obs_sf_list = x_obs_sf[:,:,n].T.tolist()
    plt.plot([x_obs_sf[i][0] for i in range(len(x_obs_sf))],
             [x_obs_sf[i][1] for i in range(len(x_obs_sf))], 'k--')
    
    ax.plot(obs[n].x0[0],obs[n].x0[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)

res_init = ax.streamplot(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[0.8,0.8,0.8])
res_init = ax.quiver(XX, YY, xd_init[0,:,:], xd_init[1,:,:])




plt.axis('equal')
plt.ion()
plt.show()

