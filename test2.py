import numpy as np
import matplotlib.pyplot as plt

from math import pi

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



a = [1,2,3]
b=1

# if any a==1:
if 1 in a:
# if b ==1:
    print('got one')
