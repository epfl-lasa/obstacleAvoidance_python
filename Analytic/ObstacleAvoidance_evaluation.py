'''
Evaluation of symbolic expression

@author LukasHuber
@date 2018-03-08
'''

import datetime
print('')
print('Started script at {}'.format(datetime.datetime.now()))
print('')

import time
startTime = time.time()

from sympy import *
from sympy import Matrix

# Cooridnates
x1 = Symbol('x1')
x2 = Symbol('x2')

# Position of attractor
d1 = Symbol('d1')
d2 = Symbol('d2')
d2 = 0

# Direction of tangent
t1 = Symbol('t1')
t2 = Symbol('t2')

# Direction of tangent
t1_0 = Symbol('t1_0')
t2_0 = Symbol('t2_0')

# Direction of tangent
l_n = Symbol('l_n')
l_t = Symbol('l_t')




E = Matrix([[x1, t1],[x2, t2]])
D = Matrix([[l_n, 0], [0, l_t]])

M = E @ D @ E.inv()

f_x = Matrix([-x1+d1,-x2+d2])

x_dot = M @ f_x

# Extend to 3d for cross product
x_dot = x_dot.col_join(Matrix([0]))

n = Matrix([x1,x2,0])

crossP = - n.cross(x_dot)
crossP = simplify(crossP[2,0])

#print('-\vec n \times \dot \xi = ', crossP)
print('- \\vec n \\times \\dot \\xi:')
print(crossP)
print('')
#print('1/(', det(E) , ') * ', crossP)

#import pdb; pdb.set_trace() ## DEBUG ##

x_center2 = Matrix([d1,t2_0/t1_0*d1, 0])  # 
 
crossP_2 = x_dot.cross(x_center2)
crossP_2 = simplify(crossP_2[2,0])

print('- \\dot( \\xi) \\times \\xi:')
print(crossP_2)
print('')


#V_0 = phi


endTime = time.time()
print('')
print('Finished script in {} ms.'.format(round(1000*(endTime-startTime),3) ) )
print('')

