# python3 -- set path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from math import pi

import sys


lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)


from lib_modulation import *
from dynamicalSystem_lib import *


# Custom functions

def multSin(ang, r1=2, r2=1, nSin=4, dPhi=pi/6):
    return r1+r2*np.cos((dPhi+ang)*nSin)

def d_multSin(ang, r1=2, r2=1, nSin=4, dPhi=pi/6): # derivatvie with respect to angle
    return -r2*nSin*np.sin((dPhi+ang)*nSin)

# ---------------------------------------------------------------
# Start script
# ---------------------------------------------------------------

print('\n\nScript analysis-metric start.\n\n')

# Choose obstacle function -- change for different obstacle shape
rFunc = multSin
drFunc = d_multSin


nResol = 1000
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
N_resol = 10
x_range = [-1,11]
y_range = [-5,5]

x_attr = np.array([0,0])
x_ref = np.array([5, 0])

# Create vectorfield
fig, ax = plt.subplots(figsize=(7,6))
#fig, ax = plt.subplots()
YY, XX = np.mgrid[y_range[0]:y_range[1]:N_resol*1j, x_range[0]:x_range[1]:N_resol*1j]

xd = np.zeros((2,N_resol, N_resol))

phi = np.arctan2(YY, XX)
pos = np.vstack(([XX], [YY]))
xd_init = -(pos - np.tile(np.reshape(x_attr, (2,1,1)), (1,N_resol,N_resol) ) )

rel_pos = -(pos - np.tile(np.reshape(x_ref, (2,1,1)), (1,N_resol,N_resol) ) )
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



xInside = []
yInside = []
for ix in range(N_resol):
    for iy in range(N_resol):
        #rad0_pos[:,ix,iy] = rFunc(rel_pos[:,ix,iy])
        #rad0[ix,iy] = np.linalg.norm((rad0_pos[:,ix,iy])) 
        #Gamma[ix,iy] = rad[ix,iy]/rad0[ix,iy]

        if Gamma[ix,iy]<=1:
            xd[:,ix,iy] = np.array([0,0])

            xInside.append(XX[ix,iy])
            yInside.append(YY[ix,iy])
            continue
            #plt.plot([XX[ix]], [YY[iy]], marker='.')

        l0 = 1-1/Gamma[ix,iy]
        l1 = 1+1/Gamma[ix,iy]

        D = np.diag([l0,l1])

        r_norm =np.linalg.norm(rel_pos[:,ix,iy])
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
        xd[:, ix, iy] = E @ D @ np.linalg.inv(E) @ xd_init[:, ix, iy]


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
    

#plt.figure()
#poly = np.vstack((xVals, yVals)).T
obs_polygon =  plt.Polygon(np.vstack((xVals_gamma[0], yVals_gamma[0])).T)
obs_polygon.set_color(np.array([176,124,124])/255)
obs_polygon.set_alpha(0.7)
plt.gca().add_patch(obs_polygon)
plt.plot(xVals_gamma[0], yVals_gamma[0], 'k')


for gg in range(1,len(xVals_gamma)):
    plt.plot(xVals_gamma[gg], yVals_gamma[gg], '--', color=[0.5,0.5,0.5])
# for nn in ns:
#     col = np.array([1,1,1])/np.max(ns)*nn
#     plt.plot([xVals[nn]], [yVals[nn]], color=col, marker='o')

plt.axis('equal')
plt.grid(False)

#plt.quiver(xVals[ind_ang], yVals[ind_ang], dx, dy, color=[0.,0.9,0])
# plt.quiver(xVals[ind_ang], yVals[ind_ang],
#            e_dir[ind_ang,0], e_dir[ind_ang,1], color=[0.,0.9,0])

# plt.quiver(xVals[ind_ang], yVals[ind_ang],
#            r_dir[ind_ang,0], r_dir[ind_ang,1], color=[0.7,0.0,0])

#plt.quiver(XX, YY, r_arr[0,:,:], r_arr[1,:,:], color=[.7, 0.0, 0])
#plt.quiver(XX, YY, e_arr[0,:,:], e_arr[1,:,:], color=[.0, 0.9, 0])           

#ax.quiver(XX, YY, xd[0,:,:], xd[1,:,:])
ax.streamplot(XX, YY, xd[0,:,:], xd[1,:,:], color=[0,0,0.5])
#velMag = np.linalg.norm(np.dstack((xd[0,:,:], xd[1,:,:])), axis=2 )/6*100
#strm = res_ifd = ax.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=velMag, cmap='winter', norm=matplotlib.colors.Normalize(vmin=0, vmax=10.) )


ax.plot(x_ref[0],x_ref[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)
ax.plot(x_attr[0],x_attr[1], 'k*', linewidth=18.0, markersize=18)

ax.set_xlim(x_range)
ax.set_ylim(y_range)

#plt.plot(xInside, yInside, 'ro')

plt.ion()
plt.show()

figName = 'flowAround_multiConcave'

plt.tick_params(axis='both', which='major',bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

saveFig=False
if saveFig:
    plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + figName + '.eps', bbox_inches='tight')

# plt.figure()
# plt.plot(r_dir[:,0], r_dir[:,1])
# plt.plot(r_mag*r_dir[:,0], r_mag*r_dir[:,1])
# plt.axis('equal')
# plt.grid(True)

# plt.figure()
# plt.plot(e_dir[:,0], e_dir[:,1])
# plt.plot(e_mag*e_dir[:,0], e_mag*e_dir[:,1], marker='.')
# ns=[ii*100 for ii in range(9)]

# for nn in ns:
#     col = np.array([1,1,1])/np.max(ns)*nn
#     plt.plot([e_mag[nn]*e_dir[nn,0]], [e_mag[nn]*e_dir[nn,1]], marker='o', color=col)
# plt.axis('equal')
# plt.grid(True)

# dAng = angles[1]-angles[0]

# dr_mag = (r_mag[1:]-r_mag[:-1])/dAng
# de_mag = (e_mag[1:]-e_mag[:-1])/dAng

# plt.figure()
# #plt.plot(angles[:-1],dr_mag)
# #plt.plot(angles[:-1],de_mag)
# plt.plot(angles[:], re/r_mag/dAng)
# plt.plot(angles[:], er/e_mag/dAng)

mag_gamma = [np.linalg.norm(np.vstack((xVals, yVals)), axis=0 )]

fig, ax = plt.subplots(figsize=(7,6))
obs_polygon =  plt.Polygon(np.vstack((np.hstack((angles[0], angles, angles[-1])),
                                      np.hstack((0,mag_gamma[0],0)) )).T )
obs_polygon.set_color(np.array([176,124,124])/255)
obs_polygon.set_alpha(0.7)
plt.gca().add_patch(obs_polygon)
plt.plot(angles, mag_gamma[0], 'k')
for gg in range(1,len(xVals_gamma)):
    plt.plot(angles, mag_gamma[0]*(gg+1), '--', color=[0.5,0.5,0.5])

plt.plot([0,0], [-10,100], color=[0,0.5,0], linewidth=4)
plt.plot([-pi,-pi], [-10,100], color=[0.5,0.,0], linewidth=8)
plt.plot([pi,pi], [-10,100], color=[0.5,0.0,0], linewidth=8)

plt.plot(0, np.abs(x_attr[0]-x_ref[0]), 'k*', linewidth=18.0, markersize=18)

plt.xlim(-pi,pi)
plt.ylim(0,10)

plt.tick_params(axis='both', which='major',bottom=True, top=False, left=False, right=False,  labelleft=False)
plt.xticks([-pi, 0, pi], ('-$\pi$',"0", "$\pi$"), fontsize=20)

figName = 'flowAround_multiConcave_unfold'
saveFig=True
if saveFig:
    plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + figName + '.eps', bbox_inches='tight')


fig, ax = plt.subplots(figsize=(7,6))
obs_polygon =  plt.Polygon(np.vstack((np.hstack((angles[0], angles[0], angles[-1], angles[-1])),
                                      np.hstack((0,1,1,0)) )).T )
obs_polygon.set_color(np.array([176,124,124])/255)
obs_polygon.set_alpha(0.7)
plt.gca().add_patch(obs_polygon)
plt.plot([angles[0],angles[-1]], [1, 1],'k')
for gg in range(1,len(xVals_gamma)):
    plt.plot([angles[0],angles[-1]], [gg+1, gg+1], '--', color=[0.5,0.5,0.5])

plt.plot([0,0], [-10,100], color=[0,0.5,0], linewidth=4)
plt.plot([-pi,-pi], [-10,100], color=[0.5,0.,0], linewidth=8)
plt.plot([pi,pi], [-10,100], color=[0.5,0.0,0], linewidth=8)

attr_pos_lin = [0, np.abs(x_attr[0]-x_ref[0])/mag_gamma[0][np.where(angles>0)[0][0]]]
plt.plot(0, attr_pos_lin[1], 'k*', linewidth=18.0, markersize=18)

N_resol = 20
x_range=[-pi,pi]
y_range=[0,6.5]

YY, XX = np.mgrid[y_range[0]+1:y_range[1]:N_resol*1j, x_range[0]:x_range[1]:N_resol*1j]
pos = np.vstack(([XX], [YY]))

xd_init = -(pos - np.tile(np.reshape(attr_pos_lin, (2,1,1)), (1,N_resol,N_resol) ) )

plt.quiver(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[0,0,0.7])

#plt.plot(xInside, yInside, 'ro')

plt.xlim(x_range)
plt.ylim(y_range)

plt.tick_params(axis='both', which='major',bottom=True, top=False, left=False, right=False,  labelleft=False)
plt.xticks([-pi, 0, pi], ('-$\pi$',"0", "$\pi$"), fontsize=20)

figName = 'flowAround_multiConcave_unfold_flat'
if saveFig:
    plt.savefig('/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/fig/' + figName + '.eps', bbox_inches='tight')





print('\n\nEnd script. \n\n')
