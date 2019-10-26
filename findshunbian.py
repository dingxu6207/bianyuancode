# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:50:28 2019

@author: dingxu
"""

import os

os.chdir('E:\shunbianyuan\ldf_download')
curentpath = os.getcwd()
print(curentpath)
path = curentpath
pathone = 'E:/shunbianyuan/ldf_download/20191009'
pathtwo = 'E:/shunbianyuan/ldf_download/20191013'
fitsname = 'PGC%2021073.fts'

f = open("data.txt","w")   #设置文件对象

for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       f.writelines(strfile+'\n')
       if (file == fitsname):           
           print(strfile)

count = 0
listfits = [0 for i in range(100)]
for filename1 in os.listdir(pathone):
    for filename2 in os.listdir(pathtwo):
        if(filename1 == filename2):           
            listfits[count] = filename2
            count = count+1
            
            
print(count)

f.close()