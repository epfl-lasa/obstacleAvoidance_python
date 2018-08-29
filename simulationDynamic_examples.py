#!/usr/bin/python3
import sys

import numpy as np

# ---------- Import Custom libraries ----------
lib_string = "/home/lukas/Code/MachineLearning/ObstacleAvoidanceAlgorithm/lib_obstacleAvoidance/"
if not any(lib_string in s for s in sys.path):
    sys.path.append(lib_string)

from class_obstacle import *
from dynamicSimulation import *

#if __name__ == '__main__':
    #a = AnimatedScatter()
#    a.show()
#plt.ion()
#### Create starting points

N = 3

def samplePointsAtBorder(N, xRange, yRange):
    # Draw points evenly spaced at border
    dx = xRange[1]-xRange[0]
    dy = yRange[1]-yRange[0]

    N_x = ceil(dx/(2*(dx+dy))*(N))+2
    N_y = ceil(dx/(2*(dx+dy))*(N))-0

    x_init = np.vstack((np.linspace(xRange[0], xRange[1], num=N_x),
                        np.ones(N_x)*yRange[0]))

    x_init = np.hstack((x_init,
                        np.vstack((np.linspace(xRange[0], xRange[1], num=N_x),
                                   np.ones(N_x)*yRange[1]))))

    ySpacing = (yRange[1]-yRange[0])/(N_y+1)
    x_init = np.hstack((x_init,
                        np.vstack((np.ones(N_y)*xRange[0],
                                   np.linspace(yRange[0]+ySpacing, yRange[1]-ySpacing, num=N_y) )) ))

    x_init = np.hstack((x_init, 
                        np.vstack((np.ones(N_y)*xRange[1],
                                   np.linspace(yRange[0]+ySpacing,yRange[1]-ySpacing, num=N_y) )) ))

    return x_init

    
simuCase=7
if simuCase==0:
    N = 10
    x_init = np.vstack((np.ones(N)*20,
                        np.linspace(-10,10,num=N) ))
    ### Create obstacle 
    obs = []
    a = [5, 2] 
    p = [1, 1]
    x0 = [10.0, 0]
    th_r = 30/180*pi
    sf = 1.

    #xd=[0, 0]
    w = 0
    x_start = 0
    x_end = 2
    #obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))

    a = [3,2]
    p = [1,1]
    x0 = [7,-6]
    th_r = -40/180*pi
    sf = 1.

    xd=[0.25, 1]
    w = 0
    x_start = 0
    x_end = 10
    
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))
    a = [3,2]
    p = [1,1]
    x0 = [7,-6]
    th_r = -40/180*pi
    sf = 1.

    xd=[0., 0]
    w = 0
    x_start = 0
    x_end = 0
    #obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))
    
    ob2 = Obstacle(
        a= [1,1],
        p= [1,1],
        x0= [10,-8],
        th_r= -40/180*pi,
        sf=1,
        xd=[0, 0],
        x_start=0,
        x_end=0,
        w=0
    )
    #obs.append(ob2)

    ob3 = Obstacle(
        a= [1,1],
        p= [1,1],
        x0= [14,-2],
        th_r= -40/180*pi,
        sf=1,
        xd=[0, 0],
        x_start=0,
        x_end=0,
        w=0
    )
    obs.append(ob3)

    xRange = [ -1,20]
    yRange = [-10,10]
    zRange = [-10,10]
    #obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

    attractorPos = [0,0]

    anim = Animated(x_init, obs, xRange=xRange, yRange=yRange, dt=0.05, N_simuMax=1040, convergenceMargin=0.3, sleepPeriod=0.01,attractorPos=attractorPos )
    
elif simuCase==1:
    N = 10
    x_init = np.vstack((np.ones(N)*1,
                        np.linspace(-1,1,num=N),
                        np.linspace(-1,1,num=N) ))
    ### Create obstacle 
    obs = []

    x0 = [0.5,0.2,0.0]
    a = [0.4,0.1,0.1]
    #a = [4,4,4]
    p = [10,1,1]
    th_r = [0, 0, 30./180*pi]
    sf = 1.

    xd=[0,0,0]
    w = [0,0,0]

    x_start = 0
    x_end = 2
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))

    ### Create obstacle
    x0 = [0.5,-0.2,0]
    a = [0.4,0.1,0.1]
    p = [10,1,1]
    th_r = [0, 0, -30/180*pi]
    sf = 1

    xd=[0,0,0]
    w = [0,0,0]

    x_start = 0
    x_end = 2
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))
    xRange = [-0.2,1.8]
    yRange = [-1,1]
    zRange = [-1,1]


elif simuCase ==2:
    xRange = [-0.7,0.3]
    yRange = [2.3,3.0]
    
    xRange = [-3,3]
    yRange = [-3,3.0]

    N = 10
    #x_init = np.vstack((np.linspace(-.19,-0.16,num=N),
    # np.ones(N)*2.65))

    x_init = np.vstack((np.linspace(-3,-1,num=N),
                        np.ones(N)*0))
                        
    xAttractor = np.array([0,0])

    obs = []
    
    obs.append(Obstacle(a=[1.1, 1],
                        p=[1,1],
                        x0=[0.5,1.5],
                        th_r=-25*pi/180,
                        sf=1.0
    ))
    
    a = [0.2,5]
    p = [1,1]
    x0 = [0.5, 5]
    th_r = -25/180*pi
    sf = 1.0
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

    anim = Animated(x_init, obs, xRange=xRange, yRange=yRange, dt=0.003, N_simuMax=1040, convergenceMargin=0.3, sleepPeriod=0.01)

    
elif simuCase ==3:
    xRange = [-0.7,0.3]
    yRange = [2.3,3.0]
    
    xRange = [-4,4]
    yRange = [-0.1,6.0]

    N = 20
    x_init = np.vstack((np.linspace(-4.5,4.5, num=N),
                        np.ones(N)*5.5))
                       
                        
    xAttractor = np.array([0,0])

    obs = []
    obs.append(Obstacle(
        a = [1.1,1.2],
        p = [1,1],
        x0 = [-1, 1.5],
        th_r = -25/180*pi,
        sf = 1.0
        ))
    
    obs.append(Obstacle(
        a = [1.8,0.4],
        p = [1,1],
        x0 = [0, 4],
        th_r = 20/180*pi,
        sf = 1.0,
        ))
    
    obs.append(Obstacle(
        a=[1.2,0.4],
        p=[1,1],
        x0=[3,3],
        th_r=-30/180*pi,
        sf=1.0 
        ))

    anim = Animated(x_init, obs, xRange=xRange, yRange=yRange, dt=0.02, N_simuMax=1040, convergenceMargin=0.3, sleepPeriod=0.01)

elif simuCase==4:
    xRange = [0,16]
    yRange = [0,9]
    
    N = 4
    #x_init = np.vstack((np.ones(N)*16,
    #                    np.linspace(0,9,num=N) ))b
    
    ### Create obstacle 
    obs = []
    x0 = [3.5,1]
    a = [2.5,0.8]
    p = [1,1]
    th_r = -10
    sf = 1.3

    xd0=[0,0]
    w0 = 0

    x_start = 0
    x_end = 10
    obs.append(Obstacle(a=a, p=p, x0=x01,th_r=th_r, sf=sf, x_start=x_start, x_end=x_end, timeVariant=True))

    def func_w1(t):
        t_interval1 = [0, 2.5, 5, 7, 8, 10]
        w1 = [th_r, -20, -140, -140, -170, -170]
        
        for ii in range(len(t_interval1)-1):
            if t < t_interval1[ii+1]:
                return (w1[ii+1]-w1[ii])/(t_interval1[ii+1]-t_interval1[ii]) * pi/180
        return 0

    def func_xd1(t):
        t_interval1x = [0, 2.5, 5, 7, 8, 10]
        xd1 = [[x01[0], 7, 9, 9, 7, 6],
              [x01[1], 4, 5, 5, 4, -2]]

        for ii in range(len(t_interval1x)-1):
            if t < t_interval1x[ii+1]:
                dt = (t_interval1x[ii+1]-t_interval1x[ii])
                return [(xd1[0][ii+1]-xd1[0][ii])/dt, (xd1[1][ii+1]-xd1[1][ii])/dt]
        return 0

    obs[0].func_w = func_w1
    obs[0].func_xd = func_xd1

    x0 = [12,8]
    a = [2,1.2]
    p = [1,1]
    th_r = 0
    sf = 1.3

    xd0=[0,0]
    w0 = 0

    x_start = 0
    x_end = 10
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, x_start=x_start, x_end=x_end, timeVariant=True))

    def func_w2(t):
        t_interval = [0, 2., 6.5, 7, 10]
        w = [th_r, -60, -60, 30, 30]
        
        for ii in range(len(t_interval)-1):
            if t < t_interval[ii+1]:
                return (w[ii+1]-w[ii])/(t_interval[ii+1]-t_interval[ii]) * pi/180
        return 0

    def func_xd2(t):
        t_interval = [0, 2.0, 5, 6.5, 9, 10]
        xd = [[x0[0], 13, 13, 12, 14, 15], 
              [x0[1], 6, 6, 3, -2, -3 ]]

        for ii in range(len(t_interval)-1):
            if t < t_interval[ii+1]:
                dt = (t_interval[ii+1]-t_interval[ii])
                return [(xd[0][ii+1]-xd[0][ii])/dt, (xd[1][ii+1]-xd[1][ii])/dt]
        return 0

    obs[1].func_w = func_w2
    obs[1].func_xd = func_xd2

    x_init = np.array([[15.5],[0.2]])
    attractorPos = [4,8]

    anim = Animated(x_init, obs, xRange=xRange, yRange=yRange, dt=0.01, N_simuMax=1040, convergenceMargin=0.3, sleepPeriod=0.01,attractorPos=attractorPos )
    print('Animation Finished')

elif simuCase==5:
    
    xRange = [-4,4]
    yRange = [-0.1,6.0]

    N = 20

    dx = xRange[1]-xRange[0]
    dy = yRange[1]-yRange[0]

    N_x = ceil(dx/(2*(dx+dy))*N)
    N_y = ceil(dx/(2*(dx+dy))*N)

    x_init = np.vstack((np.linspace(xRange[0],xRange[1], num=N_x),
                        np.ones(N_x)*yRange[0]) )

    x_init = np.hstack((x_init, 
                        np.vstack((np.linspace(xRange[0],xRange[1], num=N_x),
                                   np.ones(N_x)*yRange[1] )) ))

    x_init = np.hstack((x_init, 
                        np.vstack((np.ones(N_y)*xRange[0],
                                   np.linspace(yRange[0],yRange[1], num=N_y) )) ))

    x_init = np.hstack((x_init, 
                        np.vstack((np.ones(N_y)*xRange[1],
                                   np.linspace(yRange[0],yRange[1], num=N_y) )) ))
    #x_init = np.array( [[-2,-2,-1],
    #                    [2, 3, 3]])
    xAttractor = np.array([0,0])

    obs = []
    obs.append(Obstacle(
        a = [1.1,1.2],
        p = [1,1],
        x0 = [-1, 1.5],
        th_r = -25/180*pi,
        sf = 1
        ))
    
    obs.append(Obstacle(
        a = [1.8,0.4],
        p = [1,1],
        x0 = [0, 4],
        th_r = 20/180*pi,
        sf = 1.0,
        ))
    
    obs.append(Obstacle(
        a=[1.2,0.4],
        p=[1,1],
        x0=[3,3],
        th_r=-30/180*pi,
        sf=1.0 
        ))
    anim = Animated(x_init, obs, xRange=xRange, yRange=yRange, dt=0.02, N_simuMax=1040, convergenceMargin=0.3, sleepPeriod=0.01)

    #dist slow = 0.18
    #anim.ani.save('ani/simue.mpeg', writer="ffmpeg")
    #FFwriter = animation.FFMpegWriter()
    #anim.ani.save('ani/basic_animation.mp4', writer = FFwriter, fps=20)

if simuCase==6:
    xRange = [-0.1,12]
    yRange = [-5,5]

    N=5
    #x_init = samplePointsAtBorder(N, xRange, yRange)
    x_init = np.vstack((np.ones((1,N))*8,
                        np.linspace(-1,1,num=N),))

    xAttractor=[0,0]
    
    obs = []
    a=[0.3, 2.5]
    p=[1,1]
    x0=[2,0]
    th_r=-50/180*pi
    sf=1
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

    # Obstacle 2
    a = [0.4,2.5]
    p = [1,1]
    #x0 = [7,2]
    x0 = [6,0]
    th_r = 50/180*pi
    sf = 1
    obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

if simuCase ==7:
    xAttractor = np.array([0,0])
    centr = [2, 2.5]

    obs = []
    N = 16
    R = 5
    th_r0 = 38/180*pi
    rCent=2.4
    for n in range(N):
        obs.append(Obstacle(
            a = [0.2,3],
            p = [1,1],
            x0 = [R*cos(2*pi/N*n), R*sin(2*pi/N*n)],
            th_r = th_r0 + 2*pi/N*n,
            sf = 1.0))
        
        obs[n].center_dyn=[obs[n].x0[0]-rCent*sin(obs[n].th_r),
                           obs[n].x0[1]+rCent*cos(obs[n].th_r)]
    
    xRange = [-10,10]
    yRange = [-8,8]
    N = 20
    
    
    x_init = samplePointsAtBorder(N, xRange, yRange)
    
    anim = Animated(x_init, obs, xRange=xRange, yRange=yRange, dt=0.01, N_simuMax=1040, convergenceMargin=0.3, sleepPeriod=0.01)



    


saveFigure = True

if saveFigure:
    anim.ani.save('ani/basic_animation.mp4', dpi=100,fps=50)
    print('Saving finished.')
    plt.close('all')
else:
    anim.show()

#if __name__ == '__main__':
#if True:
    #anim = Animated(x_init, obs, xRange=xRange, yRange=yRange, zRange=zRange, dt=0.005, N_simuMax=200000, convergenceMargin=0.3, sleepPeriod=0.01, )
    #

print()
print('---- Script finished ---- ')
print() # THE END
