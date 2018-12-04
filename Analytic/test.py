
#
#
#

from math import sin, cos

import numpy as np
import numpy.linalg as LA

from sympy import *
# from sympy import Matrix

# A = np.array([[1,2], [3,4]])

# phi = 10/180*pi
# R = np.array([[cos(phi), -sin(phi)],[sin(phi), cos(phi) ]])

# #A_hat = R @ A  @R.T
# A_hat = list(A)n
# A_hat[0,:] = 0.6*A[0,:] + 4*A[1,:]


# eigA = LA.eig(A)

# eigR = LA.eig(R)
# eigA_hat = LA.eig(A_hat)

# Define x1, x2 as unknown functions
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
d = Symbol('d')

#
E = Matrix(([a, b], [c, d]))

I = Matrix([[1,0], [0, 1]])
I1 = Matrix([[1,0], [0, 0]])

res1 = simplify(E @ (I+I1) @ E.inv()) 
pprint( res1 )

res2 = simplify(E @ (I1) @ E.inv() + E @ (I) @ E.inv()) 
pprint( res2)

pprint(res2-res1)
