# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:29:34 2019

@author: dingxu
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

hdu = fits.open('E:\\shunbiayuan\\ldf_download\\bassdata\\d8549.0192.fits')
imgdata = hdu[1].data
hdu[0].header
