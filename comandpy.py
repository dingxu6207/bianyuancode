# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:26:07 2019

@author: dingxu
"""

import os
 
pathone = 'E:/shunbianyuan/ldf_download/20191009'
pathtwo = 'E:/shunbianyuan/ldf_download/20191013'
length = 100
count = 0
listfits = [0 for i in range(length)]
for filename1 in os.listdir(pathone):
    for filename2 in os.listdir(pathtwo):
        if(filename1 == filename2):           
            listfits[count] = filename2
            count = count+1



for i in range(count+1):
    cmd = 'python cmdminus.py '+listfits[i]+' 1>>log.txt'
    os.system(cmd)
    