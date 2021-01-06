
# # # Kalman filter

#______________________________________#

# # Libraries

import numpy as np
import matplotlib.pyplot as plt

#______________________________________#

# # Creating the noisy signal

x_axis = np.linspace(0, 2*(np.pi), 201)
dt = 1/(len(x_axis) - 1)

f = np.sin(2*x_axis)
fdot = 2*np.cos(2*x_axis)

sigma1, mu1, sigma2, mu2 = 0.1, 0, 0.7, 0
noise1 = sigma1*np.random.randn(201) + mu1
noise2 = sigma1*np.random.randn(201) + mu2

a = f + noise1
b = fdot + noise2

# z is the state-space vector for the variable
z = []
for i in range(0,len(a)):
    z = z + [[b[i], a[i]]]
z = np.asarray(z)

#______________________________________#

# # Setting up the matrices

phi = np.asarray([[np.cos(dt), -np.sin(dt)], [np.sin(dt), np.cos(dt)]])
H = np.asarray([[1, 0], [0, 1]])

R = (sigma2**2)*np.asarray([[1, 0], [0, 1]])
Q = (sigma1**2)*(np.asarray([[1, 0], [0, 1]]))

# Initial guesses 

x_0 = np.asarray([1, 0])
p_0 = np.asarray([[1, 0], [0, 1]])

#______________________________________#

# # Doing the iterative procedure

x, _x,_P, P = [], [x_0], [p_0], []
for i in range(0,len(x_axis)):
    K = _P[i]*(np.transpose(H))*(np.linalg.inv(H*_P[i]*np.transpose(H) + R))
    x = x + [_x[i] + K.dot(z[i] - H.dot(_x[i]))]
    P = P + [(np.eye(2) - K*H)*_P[i]]
    _x = _x + [phi.dot(x[i])]
    _P = _P + [phi*P[i]*np.transpose(phi) + Q]
    
#______________________________________#

# # Plotting the results
    
c, d = [],[]
for i in range(0,len(x_axis)):
    c = c + [x[i][1]]
    d = d + [x[i][0]]

#plt.plot(a)
#plt.plot(b)
plt.plot(c)
#plt.plot(d)
plt.show()
    
#______________________________________#

# Chansoon Lim's comments:

# If model noise is big, Kalman filter will favour the sensor data in the output
# If the sensor data is big, the Kalman filter will favour the model that's embedded in it

# Q: as sampling time increases, model noise increases




