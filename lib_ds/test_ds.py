"""
Test DS lpvDS
"""
import numpy as np
import matplotlib.pyplot as plt

import yaml

import sys # Custom path

lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm/lib_ds/"
if not any (lib_string in s for s in sys.path):
    sys.path.append(lib_string)

# Custom library
from lpvDS import *


########################################################
print("\nStart script.....\n")
########################################################

ds_gmm_dict=yaml.load(open(lib_string+'2D-U-Nav.yml')) 

#print(ds_GMM.keys())
#if False:
DS_lpv = lpvDS(K = ds_gmm_dict['K'],
               M = ds_gmm_dict['M'],
               Priors =ds_gmm_dict['Priors'],
               Mu = ds_gmm_dict['Mu'],
               Sigma = ds_gmm_dict['Sigma'],
               A = ds_gmm_dict['A'],
               attractor = ds_gmm_dict['attractor'])


dim = int(ds_gmm_dict['M'])
x_range = [-2,13]
y_range = [-6,6]

N = 10
figureSize=(7.,6)
streamColor=[0.05,0.05,0.7]


N_x, N_y = N, N

YY, XX = np.mgrid[y_range[0]:y_range[1]:N_y*1j, x_range[0]:x_range[1]:N_x*1j]
#XX, YY = np.array(([[1]])), np.array(([[1]]))


xd_init = np.zeros((dim, N_x,N_y))
xd_mod = np.zeros((dim, N_x,N_y))

for ix in range(N_x):
    for iy in range(N_y):
        
        pos = np.array(([XX[ix,iy], YY[ix,iy]]))
        xd_init[:,ix,iy] = DS_lpv.compute_f(pos, ds_gmm_dict['attractor'])
        #xd_init = DS_lpv.compute_f(pos, ds_gmm_dict['attractor'])
        #print('xd',xd_init)
        #print('looped again')

fig_nli, ax_nli = plt.subplots(figsize=figureSize)

xd1 = np.squeeze(xd_init[0,:,:])
xd2 = np.squeeze(xd_init[0,:,:])
ax_nli.streamplot(XX, YY,
                  np.squeeze(xd_init[0,:,:]),np.squeeze(xd_init[1,:,:]),
                  color=streamColor)

plt.ion()
plt.show()

########################################################
print("\n...... finished script.\n")
########################################################

