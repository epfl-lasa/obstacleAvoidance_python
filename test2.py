import numpy as np
import matplotlib.pyplot as plt

from math import pi

import sys 
lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

from lib_modulation import *

# n = 100

# ang = np.linspace(0, pi, n)


# plt.figure()
# plt.plot(ang, 1/np.sin(ang), 'b', label='1/sin')
# plt.plot(ang, 1/np.tan(ang), 'r', label='1/tan')
# plt.plot(ang, np.abs(1/np.sin(ang))-np.abs(1/np.tan(ang)), 'g', label='difference')
# plt.grid(true)
# plt.legend()

# plt.xlabel('angel [rad]')
# plt.xlabel('factor []')
# plt.xlim(0,pi)
# plt.ylim(-5.1, 5.1)
# plt.show()



# import sys

# lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_obstacleAvoidance/"
# if not any (lib_string in s for s in sys.path):
#     sys.path.append(lib_string)

# # from lib_obstacleAvoidance/obs_common_section.py import Intersection_matrix
# from obs_common_section import Intersection_matrix

# num = 5
# Intersections = Intersection_matrix(num)

# ii = 0.1
# # for col in range(0,num):
# for col in range(0,num):
#     for row in range(col+1,num):
#         # print('[col, row]=[{},{}]'.format(col,row) )
#         Intersections.set(row, col, ii*1.1)
        
#         ii += 1
#         # print('row', )

# II = np.zeros((num,num))
# for col in range(num):
#     for row in range(num):
#         II[row, col] = Intersections.get(row, col)

# Intersections.get_bool_diag_matrix()
# a = [1,2,3]
# b=1

# # if any a==1:
# if 1 in a:
# # if b ==1:
#     print('got one')


n_resol = 100
dim = 2

x_rel = [0.3,-0.1]

pos = np.zeros((dim, n_resol, n_resol))
surf_pos = np.zeros((dim, n_resol, n_resol))
surf_pos_rel = np.zeros((dim, n_resol, n_resol))
x_vals = np.linspace(-4,4,n_resol)
y_vals = np.linspace(-4,4,n_resol)

for ix in range(n_resol):
    for iy in range(n_resol):
        # pos[:,ix,iy] = np.array([ix,iy])
        pos[:,ix,iy] = np.array([x_vals[ix],y_vals[iy]])

        rad_pos = get_radius_ellipsoid(x_t=pos[:,ix,iy], a=[1,2])
        
        pos_norm = LA.norm(pos[:,ix,iy])
        if pos_norm:
            pos_rel = pos[:,ix,iy]/pos_norm
        else:
            pos_rel = pos[:,ix,iy]
        surf_pos[:,ix,iy] = rad_pos*pos_rel

        # plt.figure()
        # plt.plot(pos[0,:,:], pos[1,:,:], 'b.')
        # plt.plot(surf_pos[0,:,:], surf_pos[1,:,:], 'g.')
        # plt.axis('equal')
        rad_pos = get_radius(vec_cent2ref=x_rel , vec_ref2point=pos[:,ix,iy], a=[1,2])
        
        pos_norm = LA.norm(pos[:,ix,iy])
        if pos_norm:
            pos_rel = pos[:,ix,iy]/pos_norm
        else:
            pos_rel = pos[:,ix,iy]

        surf_pos_rel[:,ix,iy] = rad_pos*pos_rel

        # print('got another')
        # r = get_radius_ellipsoid(x_t=pos[:ix], a=[])


plt.figure()
# plt.plot(pos[0,:,:], pos[1,:,:], 'b.')
plt.plot(surf_pos[0,:,:], surf_pos[1,:,:], 'g.')
plt.plot(surf_pos_rel[0,:,:]+x_rel[0], surf_pos_rel[1,:,:]+x_rel[1], 'r.')
plt.plot(x_rel[0], x_rel[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)
plt.plot(0, 0, 'k.', linewidth=18, markeredgewidth=4, markersize=13)
plt.axis('equal')
plt.grid('on')
plt.show()
