# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:02:08 2019

@author: dingxu
"""

import numpy as np
from numpy.linalg import solve

''''
 2x + 3y = 5
 x   + 3y = 3
'''
data0 = 786
data1 = 340
data2 = 974
data3 = 20

ydata0 = 776
ydata1 = 336
ydata2 = 965
ydata3 = 16

a=np.mat([[data0,data1,1,0],[data1,-data0,0,1],[data2,data3,1,0],[data3,-data2,0,1]])#系数矩阵
b=np.mat([ydata0,ydata1,ydata2,ydata3]).T    #常数项列矩阵
x=solve(a,b)        #方程组的解
delx = x[2]
dely = x[3]
print(x)

