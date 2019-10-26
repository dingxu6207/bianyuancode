# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:36:33 2019

@author: dingxu
"""
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils import SExtractorBackground

fitsname = 'NGC%2011.fts'
routename1 = 'E:\\shunbianyuan\\ldf_download\\20190921\\'

fitsname1 = routename1+fitsname

hduA1 = fits.open(fitsname1)
imagedataA1 = hduA1[0].data

sigma_clip = SigmaClip(sigma = 3.0)
bkg = SExtractorBackground(sigma_clip)
bkg_value = bkg.calc_background(imagedataA1)

print(bkg_value)