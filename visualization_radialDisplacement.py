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


obstacleShape = 'ellipse'

ds_init = constVel
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
    ellipse_ax = [1,4]
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


# ---------------------------------------------------------------
# Start script
# ---------------------------------------------------------------
print('\n\nScript analysis-metric start.\n\n')

# Choose obstacle function -- change for different obstacle shape

figureSize = (8,6)

nResol = 400
angles = np.linspace(-pi,pi,nResol)

#xVals = np.zeros(nResol)
#yVals = np.zeros(nResol)

rads = rFunc(angles)

xVals = np.cos(angles)*rads
yVals = np.sin(angles)*rads

nTang = 14
ind_ang = np.array(np.floor(np.linspace(0,nResol-1, nTang)), dtype=int)

#dr_dAng = drFunc(angles[ind_ang])
#dx = dr_dAng*np.cos(angles[ind_ang]) - rads[ind_ang]*np.sin(angles[ind_ang])
#dy = dr_dAng*np.sin(angles[ind_ang]) + rads[ind_ang]*np.cos(angles[ind_ang])

x_ref = np.array([0,0])
dr_dAng = drFunc(angles)

# Calc all reference directions and tangent directions
r_dir = np.vstack((xVals, yVals)).T - x_ref
e_dir = np.vstack((dr_dAng*np.cos(angles) - rads*np.sin(angles),
                   dr_dAng*np.sin(angles) + rads*np.cos(angles))).T

# Normalize
normR = np.linalg.norm(r_dir, axis=1)
r_dir = r_dir / np.vstack((normR, normR)).T

normE = np.linalg.norm(e_dir, axis=1)
e_dir = e_dir / np.vstack((normE, normE)).T


r_mag = np.zeros(nResol)
e_mag = np.zeros(nResol)

r_mag[0] = 1
e_mag[0] = 1

re = np.zeros(nResol)
er = np.zeros(nResol)


dTheta_mat = np.zeros((2,2,nResol))

convex = np.zeros((nResol))
# Define new length of r_dir, such that diag(E^{-1}(x)@ E(x))==0
# d_rr ==0

# Create streamplot
N_resol = 5
x_range = [-1,11]
y_range = [-5,5]

x_attr = np.array([0,0])
x_ref = np.array([5, 0])

dx = (y_range[1]-y_range[0])/N_resol
dy = (x_range[1]-x_range[0])/N_resol
YY, XX = np.mgrid[y_range[0]+dy/2:y_range[1]-dy/2:N_resol*1j, x_range[0]+dx/2:x_range[1]-dx/2:N_resol*1j]

YY_radial, XX_radial = YY, XX

#fig, ax = plt.subplots()
# YY, XX = np.mgrid[y_range[0]:y_range[1]:(N_resol+2)*1j, x_range[0]:x_range[1]:(N_resol+2)*1j]

# YY = YY[1:-1,:][:,1:-1]
# XX = XX[1:-1,:][:,1:-1]
xd = np.zeros((2,N_resol, N_resol))

phi = np.arctan2(YY, XX)
pos = np.vstack(([XX], [YY]))
xd_init = -(pos - np.tile(np.reshape(x_attr, (2,1,1)), (1,N_resol,N_resol) ) )
xd_norm = np.zeros(xd_init.shape)
m_of_x = np.zeros(xd_init.shape)


rel_pos = (pos - np.tile(np.reshape(x_ref, (2,1,1)), (1,N_resol,N_resol) ) )
phi = np.arctan2(rel_pos[1,:,:], rel_pos[0,:,:])

rad = np.linalg.norm(rel_pos, axis=0)

rad0 = rFunc(phi)
#rad0 = np.linalg.norm(rad0_pos, axis=0)
Gamma = rad/rad0
#rad0_pos = np.zeros((2,N_resol, N_resol))
#rad0 = np.zeros((N_resol, N_resol))
#Gamma = np.zeros((N_resol, N_resol))

e_arr = np.zeros((2, N_resol, N_resol))
r_arr = np.zeros((2, N_resol, N_resol))

linear=False

xInside = []
yInside = []
for ix in range(N_resol):
    for iy in range(N_resol):
        #rad0_pos[:,ix,iy] = rFunc(rel_pos[:,ix,iy])
        #rad0[ix,iy] = np.linalg.norm((rad0_pos[:,ix,iy])) 
        #Gamma[ix,iy] = rad[ix,iy]/rad0[ix,iy]

        if Gamma[ix,iy]<=1:
            xd[:,ix,iy] = np.array([0,0])
            xd[:,ix,iy] = np.array([0,0])

            xInside.append(XX[ix,iy])
            yInside.append(YY[ix,iy])
            continue
            #plt.plot([XX[ix]], [YY[iy]], marker='.')

        if  linear:
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
        
        e = np.array([dr_dAng*np.cos(phi[ix,iy]) - rad[ix,iy]*np.sin(phi[ix,iy]),
                      dr_dAng*np.sin(phi[ix,iy]) + rad[ix,iy]*np.cos(phi[ix,iy])])
        
        e_norm =np.linalg.norm(e)
        if e_norm ==0:
            print('WARNING -- e_norm = 0')
            xd[:,ix,iy] = np.array([0,0])
            continue            
        else:
            e = e/e_norm

        #print('E', E)
        #print('D', D)
        e_arr[:, ix, iy] = e # TODO REMOVE
        r_arr[:, ix, iy] = r
        
        E = np.array([r, e]).T

        if linear:
            xd_init[:,ix,iy] = xd_init[:, ix, iy]
        else:    # nonlinear
            # m_of_x[:,ix,iy] = radial_displacement(np.array([XX[ix,iy], YY[ix,iy]]), x_ref, Gamma[ix,iy])
            m_of_x[:,ix,iy] = radial_displacement(rel_pos[:,ix,iy], x_ref, Gamma[ix,iy])
            xd_init[:,ix,iy] = ds_init(m_of_x[:,ix,iy])
        
        xd[:, ix, iy] = E @ D @ LA.inv(E) @ np.squeeze(xd_init[:,ix,iy])

        xd_mag = LA.norm(xd[:, ix, iy])
        if xd_mag:
            xd_norm[:, ix, iy] = xd[:, ix, iy]/xd_mag


for ii in range(nResol-1):
    #r_mag = 3
    # sovve equation [r_1 hat(e)] [r1;de] = r0
    e_bar = 0.5*(e_dir[ii,:]+e_dir[ii+1,:])
    matr_re = np.vstack((r_dir[ii+1,:], e_bar)).T
    re_mag = np.linalg.inv(matr_re) @ r_dir[ii,:].T * r_mag[ii]
    r_mag[ii+1] = re_mag[0]

    re[ii+1] = re_mag[1]
    
    # Check concavity
    convex[ii] = np.sign(np.cross(e_dir[ii,:], e_dir[ii+1]))
    
    r_bar = 0.5*(r_dir[ii,:]+r_dir[ii+1,:])
    matr_re = np.vstack((e_dir[ii+1,:], r_bar)).T
    re_mag = np.linalg.inv(matr_re) @ e_dir[ii,:].T * e_mag[ii]
    e_mag[ii+1] = re_mag[0]

    er[ii+1] = re_mag[1]
     
    #r_dir[ii,:] = r_dir[ii,:]*r_mag
    #r_dir[ii,:] = r_dir[ii,:]*r_mag


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
ax_init.quiver(m_of_x[0,:,:], m_of_x[1,:,:], xd_init_norm[0,:,:], xd_init_norm[1,:,:], color=[0,0,0.5])
ax_init.quiver(m_of_x[0,:,:], m_of_x[1,:,:], xd_init_norm[0,:,:], xd_init_norm[1,:,:], color=[0,0,0.5])
ax_init.quiver(XX, YY, xd_init_norm[0,:,:], xd_init_norm[1,:,:], color=[0.7,0.1,.1])

pos = np.vstack((XX.reshape(1,-1), YY.reshape(1,-1)))
for ii in range(pos.shape[1]):
    plt.plot([x_ref[0],pos[0,ii]], [x_ref[1], pos[1,ii]], '--', color=[0.4,0.4,0.4])


    
plt.axis('equal')
plt.grid(False)

ax_init.set_xlim(x_range)
ax_init.set_ylim(y_range)

ax_init.plot(x_ref[0],x_ref[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)
ax_init.plot(x_attr[0],x_attr[1], 'k*', linewidth=18.0, markersize=18)

plt.tick_params(axis='both', which='major',bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)


#plt.figure()
#poly = np.vstack((xVals, yVals)).T
fig, ax = plt.subplots(figsize=figureSize)
obs_polygon =  plt.Polygon(np.vstack((xVals_gamma[0], yVals_gamma[0])).T)
obs_polygon.set_color(np.array([176,124,124])/255)
obs_polygon.set_alpha(0.7)
plt.gca().add_patch(obs_polygon)

ax.quiver(XX, YY, xd_init_norm[0,:,:], xd_init_norm[1,:,:], color=[0.7,0.1,.1])
ax.quiver(XX, YY, xd_norm[0,:,:], xd_norm[1,:,:], color=[0,0,0.5])

# Create vectorfield

plt.plot(xVals_gamma[0], yVals_gamma[0], 'k')


# for gg in range(1,len(xVals_gamma)):
    # plt.plot(xVals_gamma[gg], yVals_gamma[gg], '--', color=[0.5,0.5,0.5])

plt.axis('equal')
plt.grid(False)


ax.plot(x_ref[0],x_ref[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)
ax.plot(x_attr[0],x_attr[1], 'k*', linewidth=18.0, markersize=18)

ax.set_xlim(x_range)
ax.set_ylim(y_range)

plt.ion()
plt.show()

figName = 'radialDisplacement_visualization'

plt.tick_params(axis='both', which='major',bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

saveFig=False
if saveFig:
    plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + figName + '.eps', bbox_inches='tight')
