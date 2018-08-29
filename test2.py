import numpy as np
import matplotlib.pyplot as plt

from math import pi

N = 100

ang = np.linspace(0, pi, N)


plt.figure()
plt.plot(ang, 1/np.sin(ang), 'b', label='1/sin')
plt.plot(ang, 1/np.tan(ang), 'r', label='1/tan')
plt.plot(ang, np.abs(1/np.sin(ang))-np.abs(1/np.tan(ang)), 'g', label='Difference')
plt.grid(True)
plt.legend()

plt.xlabel('Angel [rad]')
plt.xlabel('Factor []')
plt.xlim(0,pi)
plt.ylim(-5.1, 5.1)
plt.show()
