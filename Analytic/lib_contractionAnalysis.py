def contraction_det_trace(x1, x2, l_n, l_t, t1, t2, d1): 
    det =(-0.25*d1**2*t2**2*(t1*x2 + t2*(d1 - x1))**2*(d1*t2 + t1*x2 - t2*x1)**2*(d1*l_n*t1 - d1*l_t*t1 - l_n*t1*x1 + l_n*t2*x2 + l_t*t1*x1 - l_t*t2*x2)**2 + 1.0*(t2*(t1*(d1 - x1)*(l_n - l_t)*x2 - (l_n*t2*(d1 - x1) + l_t*t1*x2)*x1) - (t1*x2 + t2*(d1 - x1))*(l_n*t2*(d1 - x1) - l_n*t2*x1 + l_t*t1*x2 + t1*(l_n - l_t)*x2))*(1.0*t1*(d1*l_t*t2 + l_n*t1*x2 - l_n*t2*x1)*x2 - (d1*t2 + t1*x2 - t2*x1)*(1.0*d1*l_t*t2 + 2.0*l_n*t1*x2 - 1.0*l_n*t2*x1))*(1.0*d1**2*t2**2 + 2.0*d1*t1*t2*x2 - 2.0*d1*t2**2*x1 + 1.0*t1**2*x2**2 - 2.0*t1*t2*x1*x2 + 1.0*t2**2*x1**2)**2)/((t1*x2 + t2*(d1 - x1))**2*(d1*t2 + t1*x2 - t2*x1)**2*(1.0*d1**2*t2**2 + 2.0*d1*t1*t2*x2 - 2.0*d1*t2**2*x1 + 1.0*t1**2*x2**2 - 2.0*t1*t2*x1*x2 + 1.0*t2**2*x1**2)**2)
    import pdb; pdb.set_trace() ## DEBUG ##

    print('goood')
 
    tra =(-d1*l_n*t2 - d1*l_t*t2 - 2*l_n*t1*x2 + 2*l_n*t2*x1)/(d1*t2 + t1*x2 - t2*x1) 
 
    return det, tra 

