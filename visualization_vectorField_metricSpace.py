'''
Obstacle Avoidance - Visuatlization Algroith

@author lukashuber
@date 2018-02-15
'''
import matplotlib as plt

import numpy as nup
from math import pi

def visualisation_vectorField(x_range=[0,10],y_range=[0,10], N_y=10, obs=[], sysDyn_init=False, xAttractor = np.array(([0,0])), safeFigure = False):
    fig = []
    ax = []
    for ii in range(3):
        fig_temp, ax_temp = plt.subplots(figsize=(10,8))
        fig.append(fig_temp)
        ax.append(ax_temp)

    print('To continue')
    


