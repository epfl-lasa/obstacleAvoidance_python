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

init_printing(use_unicode=True)

# Coordinates
t = Symbol('t')

# Define x1, x2 as unknown functions
#x1 = Symbol('x1')
#x2 = Symbol('x2')
x1, x2 = symbols('x1 x2', cls=Function)

# Position of attractor
d1 = Symbol('d1')
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

# Obstacle in positive plane
# with d1>0, d2>0
# t1>0, t2>0 
E = Matrix([[x1(t)-d1, t1],[x2(t)-d2, t2]])
D = Matrix([[l_n, 0], [0, l_t]])

M = E @ D @ E.inv()

f_x = Matrix([-x1(t),-x2(t)])

x_dot = M @ f_x

# Extend to 3d for cross product
x_dot = x_dot.col_join(Matrix([0]))


n = Matrix([x1(t)-d1,x2(t)-d2,0])

crossP = - n.cross(x_dot)
crossP = simplify(crossP[2,0])

#print('-\vec n \times \dot \xi = ', crossP)
print('- \\vec n \\times \\dot \\xi:')
pprint(crossP)
print('')


x  = Matrix([x1(t),x2(t),0])
dotP = - x_dot.dot(x)

print('')
print('- \\xi^T \\dot \\dot \\xi:')
pprint(dotP)
print('')

x_center2 = Matrix([d1,t2_0/t1_0*d1, 0])  # 
 
crossP_2 = x_dot.cross(x_center2)
crossP_2 = simplify(crossP_2[2,0])

print('')
print('- \\dot( \\xi) \\times \\xi:')
pprint(crossP_2)
print('')

gamma_ = x2(t)/(d1-x1(t))
delta_ = x1(t)/(-d1/t2*t1)

V_0 = 1/2*(gamma_**2 + delta_**2)
# dV_0 = diff(V_0, t)


# dV_0 = dV_0.subs(Derivative(x1(t), t), x_dot[0])
# dV_0 = dV_0.subs(Derivative(x2(t), t), x_dot[1])

# dV_0 = simplify(dV_0)
# dV_O = factor(dV_0)

# print('')
# pprint(Derivative('V_0','t'), use_unicode=True)
# pprint(dV_0)
# print('')



# pos = Symbol('pos') # replaces a strictly negative values
# pos0 = Symbol('pos0') # replaces a negative values 
# neg = Symbol('neg') # replaces a strictly positive values 
# neg0 = Symbol('neg0') # replaces a negative values 

# dV_simp = dV_0.subs((l_n-l_t), neg)
# dV_simp = dV_simp.subs((-d1+x1(t)), neg)
# dV_simp = dV_simp.subs(d1^2, neg)



endTime = time.time()
print('')
print('Finished script in {} ms.'.format(round(1000*(endTime-startTime),3) ) )
print('')

