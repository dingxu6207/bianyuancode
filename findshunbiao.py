# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:50:28 2019

@author: dingxu
"""

import os
from astropy.io import fits
import xlwt

os.chdir('E:\shunbianyuan\ldf_download')
curentpath = os.getcwd()
print(curentpath)
path = curentpath

def findinfo(strfile):
    phdulist = fits.open(strfile)
    RA = phdulist[0].header['RA']
    DEC = phdulist[0].header['DEC']
    return RA,DEC
 
hangcount = 0
sheet = 'B68'
workbook = xlwt.Workbook(encoding='utf-8')
data_sheet = workbook.add_sheet(sheet,cell_overwrite_ok = True)

for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       
       if (strfile[-4:] == '.fts'):
           RA,DEC = findinfo(strfile)
           data_sheet.write(hangcount, 0, strfile)
           data_sheet.write(hangcount, 1, RA)
           data_sheet.write(hangcount, 2, DEC)
           hangcount = hangcount + 1

keepfilename =  'E:/B99.xls'
workbook.save(keepfilename)          