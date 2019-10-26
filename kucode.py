# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:57:31 2019

@author: dingxu
"""

from photutils import DAOStarFinder,CircularAperture
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


fitsname1 = 'd8549.0088.fits'
fitsname2 = 'd8549.0091.fits'
routename1 = 'E:\\shunbianyuan\\code\\phot\\'
routename2 = 'E:\\shunbianyuan\\code\\phot\\'

fitsname1 = routename1+fitsname1
fitsname2 = routename2+fitsname2

hduA1 = fits.open(fitsname1)
imagedataA1 = hduA1[1].data
hang,lie = imagedataA1.shape
imageA1 = imagedataA1.astype(float)

hduA2 = fits.open(fitsname2)
imagedataA2 = hduA2[1].data
imageA2 = imagedataA2.astype(float)


def whadjustimage(img):
    imagedata = img
    hang,lie = imagedata.shape
    mean = np.mean(imagedata)
    sigma = np.std(imagedata)
    mindata = np.min(imagedata)
    maxdata = np.max(imagedata)
    Imin = mean - 1*sigma
    Imax = mean + 1*sigma

    if (Imin < mindata):
        Imin = mindata
    else:
        Imin = Imin

    if (Imax > maxdata):
        Imax = maxdata
    else:
        Imax = Imax
    return Imin,Imax


def positionflux(image):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    daofind = DAOStarFinder(fwhm = 4.0, threshold = 5.*std)
    sources = daofind(image-median)  
    #for col in sources.colnames:
    #    sources[col].info.format = '%.8g'    
    #print(sources)    
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))    
    positionflux = np.transpose((sources['xcentroid'], sources['ycentroid'],  sources['flux']))
    mylist = positionflux.tolist()
    return positions,mylist



positionsA1,listdataA1 = positionflux(imageA1) 
listdataA1.sort(key=lambda x:x[2],reverse=True)


   
plt.figure(1)
Imin1,Imax1 = whadjustimage(imagedataA1)
plt.imshow(imagedataA1, vmin = Imin1,vmax = Imax1,cmap='gray')

apertures1 = CircularAperture(positionsA1, r = 10.)
apertures1.plot(color='blue', lw=1.5, alpha=0.5)

positionsA2,listdataA2 = positionflux(imageA2)  
plt.figure(2)
Imin2,Imax2 = whadjustimage(imagedataA2)
plt.imshow(imagedataA2, vmin = Imin2,vmax = Imax2,cmap='gray')
apertures2 = CircularAperture(positionsA2, r = 10.)
apertures2.plot(color='blue', lw=1.5, alpha=0.5)






