'''
@author LukasHuber
@date 2018-12-20

'''

# python3 -- set path

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib

from math import pi

import sys


lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

from lib_modulation import *
from dynamicalSystem_lib import *


plt.close('all')

obstacleShape = 'ellipse'

saveFig=True

# ds_init = linearDS_constVel
ds_init = nonlinear_wavy_DS

dim = 2

def radial_displacement(rel_pos, x_ref, Gamma):
    dim  = rel_pos.shape[0]
    if len(rel_pos.shape) ==1:
        n_samples =1
    else:
        n_samples = rel_pos.shape[1]
    r_norm = LA.norm(rel_pos, axis=0)

    # gamma_func =  np.tile( np.array(Gamma, - 1/Gamma),(dim,1))
    gamma_func =  np.tile( np.array(1-1/Gamma**2),(dim,1))

    return  np.squeeze((np.tile(rel_pos,(1,1)).T*gamma_func+
                        np.swapaxes(np.tile(x_ref, (n_samples, 1)),0,1 ) ))



# def radial_displacement_inverse(rel_pos, x_ref, Gamma):
    # r_norm = LA.norm(rel_pos, axis=0)
    # return 1/(Gamma - 1/Gamma)*r_norm + x_ref
    

if obstacleShape =='star':
    # Custom functions
    def multSin(ang, r1=2, r2=1, nSin=4, dPhi=pi/6):
        return r1+r2*np.cos((dPhi+ang)*nSin)

    def d_multSin(ang, r1=2, r2=1, nSin=4, dPhi=pi/6): # derivatvie with respect to angle

        return -r2*nSin*np.sin((dPhi+ang)*nSin)

    rFunc = multSin
    drFunc = d_multSin

elif obstacleShape=='ellipse':
    ellipse_ax = [2,3.5]
    dPhi_obs = pi/6
    def rFunc(ang, a=[1,2], obs=[], dPhi=0):
        a = ellipse_ax
        dPhi = dPhi_obs
        ang = ang+dPhi
        # TODO include obstacle
        return np.prod(a)/np.sqrt((a[0]*np.sin(ang))**2 + (a[1]*np.cos(ang))**2)
        
    def drFunc(ang, a = [1,2], obs=[], dPhi=0):
        a = ellipse_ax
        dPhi = dPhi_obs
        ang = ang+dPhi
        inerDerivative = 2*(a[0]*a[0]-a[1]*a[1])*np.sin(ang)*np.cos(ang)
        return (-1/2)*np.prod(a)*((a[0]*np.sin(ang))**2 + (a[1]*np.cos(ang))**2)**(-3/2)*inerDerivative
else:
    print('Obstacle shape tyep=<<{}>> not defined.'.format(obstacleShape))

    # Default circle
    def rFunc(ang, r=1, dPhi=0):
        return r
    
    def drFunc(ang, dPhi=0):
        return 0


def obs_avoidance_nonlinear_radialDisplacement(rel_pos, xd, x_ref, Gamma, rad0, ds_init, phi, drFunc, linear=False):
    # TODO: optimize for batch processing (vectorfields) & for single values!
    dim = xd.shape[0]
    n_resol = xd.shape[1]

    m_of_x = np.zeros((dim, n_resol, n_resol))

    xd_init = np.copy(xd)
    xd_mod = np.zeros(xd.shape)

    xInside = []
    yInside = []

    linear = False
    
    # TODO -- remove loop / make generic
    for ix in range(n_resol):
        for iy in range(n_resol):

            if Gamma[ix,iy]<=1:
                xd_mod[:,ix,iy] = np.array([0,0])
                xd_init[:,ix,iy] = np.array([0,0])

                xInside.append(rel_pos[0,ix,iy]+x_ref[0])
                yInside.append(rel_pos[1,ix,iy]+x_ref[1])
                continue

            if linear:
                l0 = 1-1/Gamma[ix,iy]
                l1 = 1+1/Gamma[ix,iy]
            else:
                if Gamma[ix,iy]==1:
                    l0 = 0
                    l1 = 1
                else:
                    l0 = (Gamma[ix,iy]**2-1)/Gamma[ix,iy]**2
                    l1 = Gamma[ix,iy]/(Gamma[ix,iy]-1)

                    normL = LA.norm([l0,l1])

                    l0 = l0/normL
                    l1 = l1/normL

            D = np.diag([l0,l1])

            r_norm =LA.norm(rel_pos[:,ix,iy])
            if r_norm ==0:
                print('WARNING -- r_norm = 0')
                xd[:,ix,iy] = np.array([0,0])
                continue            
            else:
                r = rel_pos[:,ix,iy]/r_norm

            dr_dAng = drFunc(phi[ix,iy])

            # e = np.array([dr_dAng*np.cos(phi[ix,iy]) - rad0[ix,iy]*np.sin(phi[ix,iy]),
                          # dr_dAng*np.sin(phi[ix,iy]) + rad0[ix,iy]*np.cos(phi[ix,iy])])
            e = np.array([dr_dAng*np.cos(phi[ix,iy]) - rad0[ix,iy]*np.sin(phi[ix,iy]),
                          dr_dAng*np.sin(phi[ix,iy]) + rad0[ix,iy]*np.cos(phi[ix,iy])])                          

            e_norm =np.linalg.norm(e)
            if e_norm ==0:
                print('WARNING -- e_norm = 0')
                xd[:,ix,iy] = np.array([0,0])
                continue            
            else:
                e = e/e_norm

            E = np.array([r, e]).T

            # if np.sum(r*e):
                # print('Warning')
                

            if linear:
                xd_init[:,ix,iy] = xd_init[:, ix, iy]
            else:    # nonlinear
                # m_of_x[:,ix,iy] = radial_displacement(np.array([XX[ix,iy], YY[ix,iy]]), x_ref, Gamma[ix,iy])
                m_of_x[:,ix,iy] = radial_displacement(rel_pos[:,ix,iy], x_ref, Gamma[ix,iy])
                xd_init[:,ix,iy] = ds_init(m_of_x[:,ix,iy])

                # TEST
                # ratio = np.zeros(dim)
                # for ii in range(dim):
                    # if m_of_x[ii,ix,iy] == 0:
                        # ratio[ii] = 0
                    # ratio[ii] = xd_init[ii,ix,iy]/m_of_x[ii,ix,iy]

                # if ratio[0] != ratio[1]:
                    # print('Warning')

            xd_mod[:, ix, iy] = E @ D @ LA.inv(E) @ np.squeeze(xd_init[:,ix,iy])

            xd_mag = LA.norm(xd_mod[:, ix, iy])
            if xd_mag:
                xd_mod[:, ix, iy] = xd_mod[:, ix, iy]/xd_mag

    return xd_mod, m_of_x, xd_init


def evaluate_distanceFunction(pos, x_ref, rFunc):
    dim = pos.shape[0]
    n_resol = pos.shape[1]
    
    rel_pos = (pos - np.tile(np.reshape(x_ref, (dim,1,1)), (1,n_resol,n_resol) ) )
    phi = np.arctan2(rel_pos[1,:,:], rel_pos[0,:,:])

    rad = LA.norm(rel_pos, axis=0)

    rad0 = rFunc(phi)
    Gamma = rad/rad0

    return Gamma, rel_pos, rad0, phi

    
# ---------------------------------------------------------------
# Start script
# ---------------------------------------------------------------
print('\n\nScript analysis-metric start.\n\n')

# Choose obstacle function -- change for different obstacle shape

figureSize = (6,4.5)

nResol = 400
angles = np.linspace(-pi,pi,nResol)

#xVals = np.zeros(nResol)
#yVals = np.zeros(nResol)

rads = rFunc(angles)

xVals = np.cos(angles)*rads
yVals = np.sin(angles)*rads

# Create streamplot
N_resol = 5
n_stream = 100

x_range = [-1,13]
y_range = [-5,5]

x_attr = np.array([0,0])
x_ref = np.array([5.3, -0.4])

dy = (y_range[1]-y_range[0])/N_resol
dx = (x_range[1]-x_range[0])/N_resol
YY, XX = np.mgrid[y_range[0]+dy/2:y_range[1]-dy/2:N_resol*1j, x_range[0]+dx/2:x_range[1]-dx/2:N_resol*1j]

# phi = np.arctan2(YY, XX)
pos =  np.vstack(([XX], [YY]))
xd_init = ds_init(pos)

Gamma, rel_pos, rad0, phi  = evaluate_distanceFunction(pos=pos, x_ref=x_ref, rFunc=rFunc)

xd_mod, m_of_x, xd_init = obs_avoidance_nonlinear_radialDisplacement(rel_pos=rel_pos, xd=xd_init, phi=phi, x_ref=x_ref, Gamma=Gamma, rad0=rad0, drFunc=drFunc, ds_init=ds_init, linear=False)

# Streamplot figure
YY_stream, XX_stream = np.mgrid[y_range[0]:y_range[1]:n_stream*1j, x_range[0]:x_range[1]:n_stream*1j]
pos_stream = np.vstack(([XX_stream], [YY_stream]))
xd_init_stream = ds_init(pos_stream)

Gamma_stream, rel_pos, rad0, phi_stream  = evaluate_distanceFunction(pos=pos_stream, x_ref=x_ref, rFunc=rFunc)

xd_mod_stream, m_of_x_stream, xd_init_m = obs_avoidance_nonlinear_radialDisplacement(rel_pos=rel_pos, xd=xd_init_stream, phi=phi_stream, x_ref=x_ref, Gamma=Gamma_stream, rad0=rad0, ds_init=ds_init, drFunc=drFunc, linear=False)

rel_pos = (pos - np.tile(np.reshape(x_ref, (2,1,1)), (1,N_resol,N_resol) ) )

rad = LA.norm(rel_pos, axis=0)

yVals_gamma, xVals_gamma = [], []

for gg in range(7):
    xVals_gamma.append(xVals*(1+gg) + x_ref[0])
    yVals_gamma.append(yVals*(1+gg) + x_ref[1])
    
# Initial plot
fig_init, ax_init = plt.subplots(figsize=figureSize)
obs_polygon =  plt.Polygon(np.vstack((xVals_gamma[0], yVals_gamma[0])).T)
obs_polygon.set_color(np.array([176,124,124])/255)
obs_polygon.set_alpha(0.7)
plt.gca().add_patch(obs_polygon)
plt.plot(xVals_gamma[0], yVals_gamma[0], 'k')

# for gg in range(1,len(xVals_gamma)):
    # plt.plot(xVals_gamma[gg], yVals_gamma[gg], '--', color=[0.5,0.5,0.5])

# normalize
norm = LA.norm(xd_init,axis=0)
xd_init_norm = xd_init / np.tile(LA.norm(xd_init,axis=0) ,(2,1,1))

scale_quiv = 13
ax_init.streamplot(XX_stream, YY_stream, xd_init_stream[0,:,:], xd_init_stream[1,:,:], color=[0.5,0.5,.8])
ax_init.quiver(m_of_x[0,:,:], m_of_x[1,:,:], xd_init_norm[0,:,:], xd_init_norm[1,:,:], color=[0,0,0.5], scale=scale_quiv)
ax_init.quiver(XX, YY, xd_init_norm[0,:,:], xd_init_norm[1,:,:], color=[0.7,0.1,.1], scale=scale_quiv)


pos = np.vstack((XX.reshape(1,-1), YY.reshape(1,-1)))
for ii in range(pos.shape[1]):
    if LA.norm(Gamma.reshape(-1,1)[ii]>=1):
        plt.plot([x_ref[0],pos[0,ii]], [x_ref[1], pos[1,ii]], ':', color=[.5,0.5,0.5])
        plt.plot([m_of_x.reshape(2,-1,1)[0,ii],pos[0,ii]], [m_of_x.reshape(2,-1,1)[1,ii], pos[1,ii]], '--', color=[.1,0.1,0.1], linewidth=2)

plt.axis('equal')
plt.grid(False)

ax_init.set_xlim(x_range)
ax_init.set_ylim(y_range)

ax_init.plot(x_ref[0],x_ref[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)
plt.tick_params(axis='both', which='major',bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

figName = 'radialDisplacement_visualization_initial'
if saveFig:
    plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + figName + '.eps', bbox_inches='tight')


#plt.figure()
#poly = np.vstack((xVals, yVals)).T
fig, ax = plt.subplots(figsize=figureSize)
obs_polygon =  plt.Polygon(np.vstack((xVals_gamma[0], yVals_gamma[0])).T)
obs_polygon.set_color(np.array([176,124,124])/255)
obs_polygon.set_alpha(0.7)
plt.gca().add_patch(obs_polygon)

ax.streamplot(XX_stream, YY_stream, xd_mod_stream[0,:,:], xd_mod_stream[1,:,:], color=[0.5,0.5,.8])
ax.quiver(XX, YY, xd_init_norm[0,:,:], xd_init_norm[1,:,:], color=[0.7,0.1,.1], scale=scale_quiv)
ax.quiver(XX, YY, xd_mod[0,:,:], xd_mod[1,:,:], color=[0,0,0.5], scale=scale_quiv)

# Create vectorfield
plt.plot(xVals_gamma[0], yVals_gamma[0], 'k')

# for gg in range(1,len(xVals_gamma)):
    # plt.plot(xVals_gamma[gg], yVals_gamma[gg], '--', color=[0.5,0.5,0.5])

plt.axis('equal')
plt.grid(False)

ax.plot(x_ref[0],x_ref[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)
# ax.plot(x_attr[0],x_attr[1], 'k*', linewidth=18.0, markersize=18)

ax.set_xlim(x_range)
ax.set_ylim(y_range)

plt.ion()
plt.show()

plt.tick_params(axis='both', which='major',bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

figName = 'radialDisplacement_visualization_modulated'
if saveFig:
    plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + figName + '.eps', bbox_inches='tight')
