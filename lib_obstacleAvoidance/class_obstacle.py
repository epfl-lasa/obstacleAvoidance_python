import numpy as np
from math import sin, cos, pi
import warnings

class Obstacle:
    """ Class of obstacles """
    # self.N_obs = 0
    def __init__(self, a=[1,1], p=[1,1], x0=[0,0], th_r=0, sf=1, xd=[1,1], sigma=1):
        # Obstacle Counter
        self.a = a
        self.p = p
        self.x0 = x0
        self.th_r = th_r
        self.sf = sf
        #self.sf_a = sf_a
        self.sigma = sigma

        self.xd = xd
        self.w = 0

        self.d = len(x0)

        # Rotation Matrix
        self.compute_R()

        self.x_obs = []
        self.x_obs_sf = []

        #self.center_dyn = self.x0

    
    def update(self, t, dt):
        self.x0 = [0,0]
        self.xd = [0,0]


    def draw_ellipsoid(self, numPoints=50, a_temp = [0,0], draw_sfObs = False):
        if self.d == 2:
            theta = np.linspace(-pi,pi, num=numPoints)
            #numPoints = numPoints
        else:
            theta, phi = np.meshgrid(np.linspace(-pi,pi, num=ns[0]),np.linspace(-pi/2,pi/2,num=ns[1]) ) #
            numPoints = numPoints[0]*numPoints[1]
            theta = theta.T
            phi = phi.T

        #print('a_temp', sum(a_temp))
        #if sum(a_temp) != 0:
        #print('Not saving figure internaly.')

                # For an arbitrary shap, the next two lines are used to find the shape segment
        if hasattr(self,'partition'):
            warnings.warn('Warning - partition no finished implementing')
            for i in range(self.partition.shape[0]):
                ind[i,:] = self.theta>=(self.partition[i,1]) & self.theta<=(self.partition[i,1])
                [i, ind]=max(ind)
        else:
            ind = 0
            
        #a = obs[n].a[:,ind]
        #p = obs[n].p[:,ind]

        # TODO -- add partition index
        if sum(a_temp) == 0:
            a = self.a
        else:
#            import pdb; pdb.set_trace() ## DEBUG ##
            a = a_temp
            
        p = self.p[:]

        R = np.array(self.rotMatrix)

        x_obs = np.zeros((self.d,numPoints))
        
        if self.d == 2:
            print(a)
            x_obs[0,:] = a[0]*np.cos(theta)
            x_obs[1,:] = np.copysign(a[1], theta)*(1 - np.cos(theta)**(2*p[0]))**(1./(2.*p[1]))
        else:
            x_obs[0,:] = a[0,:]*np.cos(phi)*np.cos(theta)
            warnings.warn('Not implemented yet...')
            # TODO --- next line probs wrong. Power of zero...
            #x_obs[1,:,n] = copysign(a(1,:), (theta))*cos(phi)*(1 - 0.^(2.*p(3,:)) - cos(theta)**(2*p(0,:))).^(1./(2.*p(2,:))) 
            #x_obs[2,:,n] = a[3,:]*sign(phi).*(1 - (sign(theta).*cos(phi).*(1 - 0.^(2.*p(3,:)) - cos(theta).^(2.*p(1,:))).^(1./(2.*p(2,:)))).^(2.*p(2,:)) - (cos(phi).*cos(theta)).^(2.*p(1,:))).^(1./(2.*p(3,:)))
        
        # TODO for outside function - only sf is returned, remove x_obs to speed up
        x_obs_sf = np.zeros((self.d,numPoints))
        if hasattr(self, 'sf'):
            if type(self.sf) == int or type(self.sf) == float:
                x_obs_sf = R@(x_obs*self.sf) + np.tile(self.x0,(numPoints,1)).T
            else:
                x_obs_sf = R@(x_obs*np.tile(self.sf,(1,numPoints))) + np.tile(self.x0, (numPoints,1)).T 
        else:
            x_obs_sf = R @ x_obs + np.tile(self.x0,(1,numPoints))
            
        x_obs = R @ x_obs + np.tile(np.array([self.x0]).T,(1,numPoints))

        if sum(a_temp) == 0:
            self.x_obs = x_obs.T.tolist()
            self.x_obs_sf = x_obs_sf
        else:
             return x_obs_sf
        
        #self.x_obs_sf = R @x_obs_sf.T.tolist()
        
    def compute_R(self):
        if self.th_r == 0:
            self.rotMatrix = np.eye(self.d)
            return
        
        # rotating the query point into the obstacle frame of reference
        if self.d == 2 :
            self.rotMatrix = [[cos(self.th_r), -sin(self.th_r)],
                              [sin(self.th_r),  cos(self.th_r)]]
        else:
            print('not implemented yet')
        # elif d == 3
        #     R_x = [ 1, 0, 0 0, np.cos(th_r[0]), np.sin(th_r[0]) 0, -np.sin(th_r[0]), np.cos(th_r[0])]
        #     R_y = [np.cos(th_r(2)), 0, -np.sin(th_r(2)) 0, 1, 0 np.sin(th_r(2)), 0, np.cos(th_r(2))]
        #     R_z = [np.cos(th_r(3)), np.sin(th_r(3)), 0 -np.sin(th_r(3)), np.cos(th_r(3)), 0 0, 0, 1]
        #     R = R_x*R_y*R_z
        # else: #rotation is not yet supported for d > 3
        #     R = np.eye(d)
    
        def obs_check_collision(obs_list, X, Y):
            print('todo - impolement this in class \n')
