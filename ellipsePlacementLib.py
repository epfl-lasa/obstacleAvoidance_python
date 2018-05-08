'''
Ellipse placement library

@author Lukas Huber
@date 2018/02/21

'''

import numpy as np
import matplotlib.pyplot as plt

from math import pi,acos, ceil, cos, sin

class EllipseCreater:
    
    aEllipse1 = np.array(([1.2,0.3]))

    rCirc = 0.3
    nClust = 2

    confRatio = 0.9
    
    def __init__(self):
        self.center1 = np.array(([0.5,0.5]))
        self.center2 = np.array((-0.5, -0.5))

        self.posRobot = np.array(([0,0]))
        self.obs = {}
        self.obs['a'] = self.aEllipse1
        
        return 

    
    def defineSpaceBoundaries(self, x0, dx):
        self.spaceCenter = x0
        
        self.width = dx

        
    def simulatePointClouds(self, N_points = 20):
        self.N_points = 20
        
        phi = np.linspace(2*pi/self.N_points, 2*pi, self.N_points)
        
        cluster1 = np.vstack(self.center1[0]+(self.rCirc*np.cos(phi), self.center1[1]+self.rCirc*np.sin(phi)))
        cluster2 = np.vstack(self.center2[0]+(self.rCirc*np.cos(phi), self.center2[1]+self.rCirc*np.sin(phi)))

        self.cluster = np.hstack((cluster1, cluster2))

        plt.plot(self.cluster[0,:],self.cluster[1,:],'.')
        plt.plot(self.posRobot[0], self.posRobot[1], 'rx')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([-2,2])
        plt.ylim([-2,2])

        
    def estimateEllipse(self):
        # VERY WEAK AND QUICK IMPLEMENTATION. A MORE RIGUROUS ALGORITHM SHOULD BE CHOSEN FOR
        # FURTHER EXPERIMENTATION
        
        # Rotate everything in robot frame of reference

        # TODO: checkPointRange()

        clustSort = np.argsort((self.cluster[1]))

        N = self.cluster.shape[1]

        # Only keep clusters which are in confidence intervall
        n0 = ceil(N*(1-self.confRatio)/2)
        n1 = N-n0
        clustSort = clustSort[n0:n1]

        x0 = np.zeros((2))
        x0[1] = np.mean(self.cluster[1,clustSort])

        yRange = self.cluster[1,clustSort[-1]]-self.cluster[1,clustSort[1]]

        # Compare whether ellipse is closer on the left or right half screen
        medClust = int(clustSort.shape[0]/2)
        x_yPos = np.mean(self.cluster[0,clustSort[0:medClust]] )
        x_yNeg = np.mean(self.cluster[0,clustSort[medClust : -1]])

        # Add exception for a1>a0
        phi = acos(max(yRange,self.aEllipse1[0])/yRange) # Orienation of ellipse
        
        if(x_yPos < x_yNeg):
            phi = - phi

        self.obs['th_r'] = phi

        x0[0] = min(x_yPos,x_yNeg) + abs(cos(phi)*self.aEllipse1[0]) + abs(sin(phi)*self.aEllipse1[1])

        self.obs['x0'] = x0

        
    def checkPointRange():
        
        return 0

    

    def drawEllipse():
        
        return 0



####################################################################################################################

# Start Simulation

####################################################################################################################

Person1 = EllipseCreater()

Person1.simulatePointClouds()

Person1.estimateEllipse()

print('Wanna see da plot?')
plt.show()


