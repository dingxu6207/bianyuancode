# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:14:12 2019

@author: dingxu
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import copy

fitsname1 = 'E:\\shunbianyuan\\ldf_download\\20190921\\'+'M33%20A6.fts'
fitsname2 = 'E:\\shunbianyuan\\ldf_download\\l20190920\\'+'M33%20A6.fts'
 
hduA1 = fits.open(fitsname1)
imagedataA1 = hduA1[0].data
imagedataA1F = copy.deepcopy(imagedataA1)
hang,lie = imagedataA1.shape

hduA2 = fits.open(fitsname2)
imagedataA2 = hduA2[0].data
imagedataA2F = copy.deepcopy(imagedataA2)

###显示图像###
def whadjustimage(img):
    imagedata = img
    mean = np.mean(imagedata)
    sigma = np.std(imagedata)
    mindata = np.min(imagedata)
    maxdata = np.max(imagedata)
    Imin = mean - 3*sigma
    Imax = mean + 3*sigma

    if (Imin < mindata):
        Imin = mindata
    else:
        Imin = Imin

    if (Imax > maxdata):
        Imax = maxdata
    else:
        Imax = Imax
        
    for i in range(hang):
        for j in range(lie):
            if (imagedata[i][j] < Imin):
                imagedata[i][j] = 0
            elif (imagedata[i][j] > Imax):
                imagedata[i][j] = 255
            else:
                imagedata[i][j] = 255*(imagedata[i][j]-Imin)/(Imax-Imin)
    return np.uint8(imagedata)

delx =  -5
dely =  0
theta =  -1.34


img = Image.fromarray(imagedataA1.astype('uint16'))
rotimage = img.rotate(theta)
rotimagedataA1 = np.array(rotimage)

newimage = np.zeros((hang,lie),dtype = np.uint16)
if delx <= 0 and dely <= 0:
    newimage[0:hang+delx,0:lie+dely] = rotimagedataA1[-delx:hang,-dely:lie]
    
if delx >= 0 and dely >= 0:
    newimage[delx:hang,dely:lie] = rotimagedataA1[0:hang-delx,0:lie-dely]  
    
    
minusimage = np.float32(newimage)-np.float32(imagedataA2)

plt.figure(1)    
A3image = whadjustimage(minusimage)
plt.imshow(A3image, cmap='gray')


plt.figure(2) 
imagedataA2 = np.float32(imagedataA1)-np.float32(imagedataA2)   
A2image = whadjustimage(imagedataA2)
plt.imshow(A2image, cmap='gray')


