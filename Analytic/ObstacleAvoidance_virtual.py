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

base_pwd = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgroithm_python/"

# -------------------- Initialization Variable --------------------
# Coordinates
t = Symbol('t')

# Define x1, x2 as unknown functions
x1 = Symbol('x1')
x2 = Symbol('x2')
#x1, x2 = symbols('x1 x2', cls=Function)
k = Symbol('k')
k2 = symbols('k2', cls=Function)

# Position of attractor
d1 = Symbol('d1')
d2 = 0

t1 = Symbol('t1') # Direction of tangent
t2 = Symbol('t2') # Direction of tangent

t1_0 = Symbol('t1_0') # Direction of tangent
t2_0 = Symbol('t2_0') # Direction of tangent

l_t = Symbol('l_t') # Direction of tangent
l_n = Symbol('l_n') # Direction of tangent

#t1_0, t2_0 = symbols('t1_0 t2_0', cls=Function)
#l_t, l_n = symbols('l_t, l_n', cls=Function)

# -------------------- Linear System --------------------
f_x = -1.0*k*Matrix([x1,x2])
#f_x = k2(x1(t),x2(t))*Matrix([-x1(t),-x2(t)])


# -------------------- Obstacel Avoidance Matrices --------------------
# Obstacle in positive plane
# with d1>0, d2>0
# t1 in R, t2>0

# Normal
#n = Matrix([x1(t)-d1,x2(t)-d2,0])
n = Matrix([x1-d1,x2-d2,0])
#E = Matrix([[n[0], t1],[n[1], t2]])
#D = Matrix([[l_n, 0], [0, l_t]])
D = Matrix([[l_n(x1,x2), 0], [0, l_t(x1,x2)]])
#E = Matrix([[n[0], t1(x1(t),x2(t))],[n[1], t2]])
E = Matrix([[n[0], t1(x1,x2)],[n[1], t2(x1,x2)]])                   

# Modulation Matrix
M = E @ D @ E.inv()

# -------------------- Evaluation --------------------
x_dot = simplify(M @ f_x)
# Extend to 3d for cross product
x_dot = x_dot.col_join(Matrix([0]))

# Evaluated at At x2=0
x_dot0 = simplify(x_dot.subs(x2(t), 0))

# Cross product
crossP = - n.cross(x_dot)
crossP = simplify(crossP[2,0])

# Position Vector
x  = Matrix([x1(t),x2(t),0])    
# dotP = - x_dot.dot(x)

M_sym = 0.5*(M + M.T)
traM = simplify(M_sym[0,0] + M_sym[1,1])
detM = simplify(M_sym[0,0]*M_sym[1,1] - M_sym[0,1]*M_sym[1,0])

detM_str = str(detM)
detM_str = str.replace(detM_str, "(x1, x2)", "")
traM_str = str(traM)
traM_str = str.replace(traM_str, "(x1, x2)", "")

print('')
print('trace M')
print(traM_str)
print('')

print('')
print('det M')
print(detM_str)
print('')


# detM_str =
# (1.0*(x2*l_n*t1 + (d1 - x1)*l_t*t2) *(x2*l_t*t1 + (d1 - x1)*l_n*t2)
# -
# 0.25*(x2*l_n*t2 - x2*l_t*t2 + (d1 - x1)*l_n*t1 - (d1 - x1)*l_t*t1)**2)
#
# /
#
# (x2*t1 + (d1 - x1)*t2)**2


detM2 = k**2*(1.0 *
(x2*l_n*t1 + (d1 - x1)*l_t*t2)
*(x2*l_t*t1 + (d1 - x1)*l_n*t2)
-0.25*(
(x2*l_n*t2 - x2*l_t*t2 + (d1 - x1)*l_n*t1 - (d1 - x1)*l_t*t1)**2
))/(
(x2*t1 + (d1 - x1)*t2)**2)

M2 = simplify(detM2)
print("detM", detM2)


# ---------
a_t = Symbol('a_t') 
a_n = Symbol('a_n') 

b_t = Symbol('b_t') 
b_n = Symbol('b_n')

eq = 4*(a_n-b_t)*(a_t-b_n) - (a_n-a_t + b_n - b_t)**2

print('eq', expand(eq))



# -------------------- Divergence / Trace of Jacobian--------------------

# divX = simplify(diff(x_dot[0], x1(t)) +  diff(x_dot[1], x2(t)))

# print('')
# print('div X')
# print(divX)
# print('')



# -------------------- Determinant of Jacobian --------------------

# JacX = Matrix([[diff(x_dot[0],x1), diff(x_dot[0],x2) ],
#                [diff(x_dot[1],x1), diff(x_dot[1],x2) ]])

# tra = simplify(JacX[0,0] + JacX[1,1])
# print('')
# print('tra X')
# print(tra)
# print('')

# # Transpose of JacobianX
# # JacSymmetric = 1/2*(JacX + Matrix( [JacX[0,0],JacX[1,0]],
# #                                  [JacX[0,1],JacX[1,1]] ) ) 

# #JacSym = 1/2*(JacX + Matrix( [[ JacX[0,0],JacX[0,1] ], [ JacX[0,0],JacX[0,1] ] ] ))
# #JacSym = 1/2*(JacX + Matrix( [[ JacX[0,0],JacX[1,0] ], [ JacX[0,1],JacX[1,1] ] ] ))
# JacSym = simplify(1/2*(JacX + JacX.T))

# det = simplify(JacSym[0,0]*JacSym[1,1] - JacSym[0,1]*JacSym[1,0])

# print('')
# print('determinant')
# print( det)
# print('')


# Theta = simplify( M**(-1) )
# F = Theta**(-1) @ JacX @ Theta
# F_sym = simplify(1/2*(F + F.T))

# print('')
# print('F')
# print(F)
# print('')

# detF = simplify((F[0,0]*F[1,1] - F[0,1]*F[1,0]))
# print('')
# print('det F')
# print(detF)
# print('')

# traF = simplify(F[0,0] + F[1,1])
# print('')
# print('tra F')
# print(traF)
# print('')

# -------------------- Zeros / Poles --------------------

# # Tangent of ellipse
# a1 = Symbol('a1')
# a2 = Symbol('a2')

# p1 = 1
# p2 = 1

# # 2D ellipse equation
# t_elli = Matrix([2*p2/a2**2 *x2(t)**(2*p2-1), - 2*p1/a1**2 *x1(t)**(2*p1-1)])

# tra_num = d1*l_n*t2 + d1*l_t*t2 + 2*l_n*t1*x2(t) - 2*l_n*t2*x1(t)
# tra_num = tra_num.subs(t1, t_elli[0])
# tra_num = tra_num.subs(t2, t_elli[1])

# det_num = d1*l_t*t2 + l_n*t1*x2(t) - l_n*t2*x1(t)
# det_num = det_num.subs(t1, t_elli[0])
# det_num = det_num.subs(t2, t_elli[1])

# denom = d1*t2 + t1*x2(t) - t2*x1(t)
# denom = denom.subs(t1, t_elli[0])
# denom = denom.subs(t2, t_elli[1])



# -------------------- Derivative Ellipsoid --------------------
# k = symbols('k', cls=Function)

# dx_lin = k(x1(t),x2(t))*(x1(t)**2+x2(t)**2)**(-1/2)*Matrix([x1(t),x2(t),0])
# div_dx_lin = simplify(diff(dx_lin[0], x1(t)) +  diff(dx_lin[1], x2(t)))


# print('')
# print('div linX')
# pprint(div_dx_lin)
# print('')


# x_center2 = Matrix([d1,t2_0/t1_0*d1, 0])  # 
 
# crossP_2 = x_dot.cross(x_center2)
# crossP_2 = simplify(crossP_2[2,0])

# print('')
# print('- \\dot( \\xi) \\times \\xi:')
# pprint(crossP_2)
# print('')

# gamma_ = x2(t)/(d1-x1(t))
# delta_ = x1(t)/(-d1/t2*t1)

# V_0 = 1/2*(gamma_**2 + delta_**2)
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

printToFunction = False
if printToFunction:
    #det_str = str(det)
    det_str = str(detF)
    det_str = str.replace(det_str, "(t)", "")

    #tra_str = str(tra)
    tra_str = str(traF)
    tra_str = str.replace(tra_str, "(t)", "")

    intend = "    " # default indent python
    with open(base_pwd + "Analytic/" + "lib_contractionAnalysis.py", "w") as text_file:
        #print(f"def determinant\(x1, x2, l_n, l_t, t1, t2, d1\):", file=text_file)
        
        text_file.write("def contraction_det_trace(x1, x2, l_n, l_t, t1, t2, d1): \n")
        text_file.write(intend + "det =" +  det_str + " \n")
        text_file.write(" \n")
        
        text_file.write(intend + "tra =" +  tra_str + " \n")
        
        text_file.write(" \n")
        text_file.write(intend + "return det, tra \n")
        
endTime = time.time()
print('')
print('Finished script in {} ms.'.format(round(1000*(endTime-startTime),3) ) )
print('')

