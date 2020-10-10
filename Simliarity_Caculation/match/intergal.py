#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:29:31 2019

@author: leyv
"""
"""
import numpy as np
from scipy import integrate

def half_circle(x):
    return (1 - x ** 2) ** 0.5
    
N = 10000
x = np.linspace(-1, 1, N)
dx = 2. / N;
y = half_circle(x)
area =sum( dx * y)#利用矩形面积法
print (np.trapz(y, x) * 2)#求数值积分
pi_half, err = integrate.quad(half_circle, -1,1) #求积分
print (pi_half * 2)

def half_sphere(x ,y):
    return (1 - x ** 2 - y ** 2) ** 0.5
print (integrate.dblquad(half_sphere,-1,1,lambda x: -half_circle(x),lambda x:half_circle(x)))#求二重积分

from scipy.integrate import odeint

def lorenz(w ,t, p, r, b):
    x ,y, z = w
    return np.array([p * (y -x), x * (r-z)-y, x * y - b * z])
t = np.arange(0 , 40, 0.01)

track1 = odeint(lorenz, (0.0,1.00,0.0),t, args=(10.0, 28.0,3.0))
track2 = odeint(lorenz, (0.0,1.01,0.0),t, args=(10.0, 28.0,3.0))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
#ax = fig.gca(projection = '3d')
ax = Axes3D(fig)
ax.plot(track1[:,0], track1[:,1], track1[:,2])
ax.plot(track2[:,0], track2[:,1], track2[:,2])
plt.show()
"""


