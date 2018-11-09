"""
Visualizatoin Radial Displacement -- Nonlinear Obstacle Avoidance

"""
xlim = [0.3,13]
ylim = [-6,6]

xAttractor=[0,0]

N_points=120
#saveFigures=True

obs=[]

a = [.80,3.0]
p = [1,1]
x0 = [5.5,-1]
th_r = 40/180*pi
sf = 1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
obs[-1].center_dyn = np.array([ 3.87541829,  0.89312174])

a = [1.0,3.0]
p = [1,1]
x0 = [5.0,2]
th_r = -50/180*pi
sf = 1
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
obs[-1].center_dyn = np.array([ 3.87541829,  0.89312174])

Simulation_vectorFields(xlim, ylim, N_points, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_intersectingObstacles', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_radial, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)



