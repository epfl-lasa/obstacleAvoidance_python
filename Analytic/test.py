#
#
#
#

from math import sin, cos

import numpy as np
import numpy.linalg as LA

A = np.array([[1,2], [3,4]])

phi = 10/180*pi
R = np.array([[cos(phi), -sin(phi)],[sin(phi), cos(phi) ]])

#A_hat = R @ A  @R.T
A_hat = list(A)n
A_hat[0,:] = 0.6*A[0,:] + 4*A[1,:]


eigA = LA.eig(A)

eigR = LA.eig(R)
eigA_hat = LA.eig(A_hat)

