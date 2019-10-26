# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:22:09 2019

@author: dingxu
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from skimage import measure,io
import math
from numpy.linalg import solve
import copy
import time
from PIL import Image
#import sys

#fitsname = sys.argv[1]
fitsname1 = 'd8549.0088.fits'
fitsname2 = 'd8549.0091.fits'
routename1 = 'E:\\shunbianyuan\\code\\phot'
routename2 = 'E:\\shunbianyuan\\code\\phot'

fitsname1 = routename1+fitsname1
fitsname2 = routename2+fitsname2

start = time.time()
hduA1 = fits.open(fitsname1)
imagedataA1 = hduA1[0].data
imagedataA1F = copy.deepcopy(imagedataA1)
hang,lie = imagedataA1F.shape

hduA2 = fits.open(fitsname2)
imagedataA2 = hduA2[0].data
imagedataA2F = copy.deepcopy(imagedataA2)
###显示图像###
def whadjustimage(img):
    imagedata = img
    hang,lie = imagedata.shape
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


###求坐标##
def qiuzuobiao(img):
    resultA = img
    flattenA1 = resultA.flatten()
    n, bins, patches = plt.hist(flattenA1, bins=256, density=1, facecolor='green', alpha=0.75)

    maxn = np.max(n)
    yuzhi = np.where(n == maxn)
    threhold = int(yuzhi[0])+40

    erzhiA1 = (resultA >= threhold)*1.0
    filterA1 = signal.medfilt(erzhiA1,(3,3)) #二维中值滤波

    labels = measure.label(filterA1,connectivity = 2) #8 连通区域标记
    print('regions number:',labels.max()+1) #显示连通区域块数(从 0 开始标

    regionnum = labels.max()+1

    hang,lie = resultA.shape
    plotx = np.zeros(regionnum,dtype = np.uint)
    ploty = np.zeros(regionnum,dtype = np.uint)

    for k in range(regionnum):
        sumx = 0
        sumy = 0
        area = 0
        for i in range(hang):
            for j in range(lie):
                if (labels[i][j] == k):
                    sumx = sumx+i
                    sumy = sumy+j
                    area = area+1
        plotx[k] = int(sumx/area+0.5)
        ploty[k] = int(sumy/area+0.5)
    return plotx,ploty,regionnum
    
def suansanjiaoxing(listS1,listS2,listS3):
    x1 = float(listS1[0])
    y1 = float(listS1[1])
    x2 = float(listS2[0])
    y2 = float(listS2[1])
    x3 = float(listS3[0])
    y3 = float(listS3[1])
    
    datadis1 = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    dS1S2 = math.sqrt(datadis1)
    
    datadis2 = ((x1-x3)*(x1-x3)+(y1-y3)*(y1-y3))
    dS1S3 = math.sqrt(datadis2)
    
    datadis3 = ((x2-x3)*(x2-x3)+(y2-y3)*(y2-y3))
    dS2S3 = math.sqrt(datadis3)
        
    if (dS1S2 < dS1S3 and dS1S3 < dS2S3):
        duan = (dS2S3/dS1S2)
        sumchen = (x1-x3)*(x2-x3) + (y1-y3)*(y2-y3)
    
    if (dS1S2 < dS2S3 and dS2S3 < dS1S3):
        duan = (dS1S3/dS1S2)
        sumchen = (x1-x3)*(x2-x3) + (y1-y3)*(y2-y3)
        
    if (dS2S3 < dS1S3 and dS1S3 < dS1S2):
        duan = (dS1S2/dS2S3)
        sumchen = (x2-x1)*(x3-x1) + (y2-y1)*(y3-y1)    
        
    if (dS2S3 < dS1S2 and dS1S2 < dS1S3):
        duan = (dS1S3/dS2S3)
        sumchen = (x2-x1)*(x3-x1) + (y2-y1)*(y3-y1)    
           
    if (dS1S3 < dS1S2 and dS1S2 < dS2S3):
        duan = dS2S3/dS1S3
        sumchen = (x1-x2)*(x3-x2) + (y1-y2)*(y3-y2)
        
    if (dS1S3 < dS2S3 and dS2S3 < dS1S2):
        duan = dS1S2/dS1S3
        sumchen = (x1-x2)*(x3-x2) + (y1-y2)*(y3-y2)    
        
    return x1,x2,x3,y1,y2,y3,duan,sumchen
    
resultA1 = whadjustimage(imagedataA1F)
A1plotx,A1ploty,A1regionnum = qiuzuobiao(resultA1)  

plt.figure(1)
plt.imshow(resultA1, cmap='gray')  
fluxA1 = np.zeros(A1regionnum,dtype = np.uint)
listdataA1 = [0 for i in range(A1regionnum)]
for i in range(A1regionnum):
    plt.plot(A1ploty[i],A1plotx[i],'*')
    fluxA1[i] = np.sum(resultA1[A1plotx[i]-6:A1plotx[i]+6,A1ploty[i]-6:A1ploty[i]+6]) 
    listdataA1[i] = (A1plotx[i],A1ploty[i],fluxA1[i] )
listdataA1.sort(key=lambda x:x[2],reverse=True)
plt.show()

jiezhi = 100
listsanjiaoA1 =  [0 for i in range(jiezhi)]
for i in range(jiezhi):
    if (i <= 7):
        x1,x2,x3,y1,y2,y3,duan,sumchen = suansanjiaoxing(listdataA1[i],listdataA1[i+1],listdataA1[i+2])
    if (i == 8):
        x1,x2,x3,y1,y2,y3,duan,sumchen = suansanjiaoxing(listdataA1[i],listdataA1[i+1],listdataA1[i-8])
    if (i == 9):
        x1,x2,x3,y1,y2,y3,duan,sumchen = suansanjiaoxing(listdataA1[i],listdataA1[i-9],listdataA1[i-8])
    listsanjiaoA1[i] = (x1,x2,x3,y1,y2,y3,duan,sumchen)
    
    
plt.figure(2)
resultA2 = whadjustimage(imagedataA2F)
A2plotx,A2ploty,A2regionnum = qiuzuobiao(resultA2)
plt.imshow(resultA2, cmap='gray') 
fluxA2 = np.zeros(A2regionnum,dtype = np.uint)
listdataA2 = [0 for i in range(A2regionnum)]
for i in range(A2regionnum):
    plt.plot(A2ploty[i],A2plotx[i],'*') 
    fluxA2[i] = np.sum(resultA2[A2plotx[i]-6:A2plotx[i]+6,A2ploty[i]-6:A2ploty[i]+6])
    listdataA2[i] = (A2plotx[i],A2ploty[i],fluxA2[i] )
listdataA2.sort(key=lambda x:x[2],reverse=True)    
plt.show()

listsanjiaoA2 =  [0 for i in range(jiezhi)]
for i in range(jiezhi):
    if (i <= 7):
        x1,x2,x3,y1,y2,y3,duan,sumchen = suansanjiaoxing(listdataA2[i],listdataA2[i+1],listdataA2[i+2])
    if (i == 8):
        x1,x2,x3,y1,y2,y3,duan,sumchen = suansanjiaoxing(listdataA2[i],listdataA2[i+1],listdataA2[i-8])
    if (i == 9):
        x1,x2,x3,y1,y2,y3,duan,sumchen = suansanjiaoxing(listdataA2[i],listdataA2[i-9],listdataA2[i-8])
    listsanjiaoA2[i] = (x1,x2,x3,y1,y2,y3,duan,sumchen)
    
plt.figure(3)
plt.imshow(resultA1, cmap='gray')
listtempA1 =  [0 for i in range(jiezhi)]
listtempA2 =  [0 for i in range(jiezhi)]
counttemp = 0
bilv = 0.015
dianchengsum = 1500
jiezhistar = 10     
for i in range(jiezhistar):
    for j in range(jiezhistar):
        if (abs(listsanjiaoA1[i][6]-listsanjiaoA2[j][6]) < bilv and abs(listsanjiaoA1[i][7]-listsanjiaoA2[j][7]) < dianchengsum):
            plt.plot(listsanjiaoA1[i][3],listsanjiaoA1[i][0],'*')             
            plt.plot(listsanjiaoA1[i][4],listsanjiaoA1[i][1],'*')
            plt.plot(listsanjiaoA1[i][5],listsanjiaoA1[i][2],'*') 
            listtempA1[counttemp] = listsanjiaoA1[i]
            listtempA2[counttemp] = listsanjiaoA2[j]
            counttemp = counttemp+1
plt.show()


plt.figure(4)
plt.imshow(resultA2, cmap='gray') 
for i in range(jiezhistar):
    for j in range(jiezhistar):
        if (abs(listsanjiaoA1[i][6]-listsanjiaoA2[j][6]) < bilv and abs(listsanjiaoA1[i][7]-listsanjiaoA2[j][7]) < dianchengsum):
            plt.plot(listsanjiaoA2[j][3],listsanjiaoA2[j][0],'*') 
            plt.plot(listsanjiaoA2[j][4],listsanjiaoA2[j][1],'*') 
            plt.plot(listsanjiaoA2[j][5],listsanjiaoA2[j][2],'*')
plt.show()

###求平移和角度###
    
data0 = listtempA1[0][0]  #x0
data1 = listtempA1[0][3]  #y0
data2 = listtempA1[0][1]  #x1
data3 = listtempA1[0][4]  #y1

#ydata2 = listtempA2[0][0] #x0
#ydata3 = listtempA2[0][3] #y0
'''
ydata0 = listtempA2[0][1] #x1
ydata1 = listtempA2[0][4] #y1

ydata2 = listtempA2[0][2] #x2
ydata3 = listtempA2[0][5] #y2
'''
abs01 = (listtempA1[0][0]-listtempA2[0][0]) - (listtempA1[0][1]-listtempA2[0][1])
abs02 = (listtempA1[0][0]-listtempA2[0][0]) - (listtempA1[0][1]-listtempA2[0][2])
abs10 = (listtempA1[0][0]-listtempA2[0][1]) - (listtempA1[0][1]-listtempA2[0][0])
abs12 = (listtempA1[0][0]-listtempA2[0][1]) - (listtempA1[0][1]-listtempA2[0][2])
abs20 = (listtempA1[0][0]-listtempA2[0][2]) - (listtempA1[0][1]-listtempA2[0][0])
abs21 = (listtempA1[0][0]-listtempA2[0][2]) - (listtempA1[0][1]-listtempA2[0][1])

redata = min(abs(abs01),abs(abs02),abs(abs10),abs(abs12),abs(abs20),abs(abs21))
if redata == abs(abs01):
    ydata0 = listtempA2[0][0] #x0
    ydata1 = listtempA2[0][3] #y0
    ydata2 = listtempA2[0][1] #x1
    ydata3 = listtempA2[0][4] #y1


if redata == abs(abs02):
    ydata0 = listtempA2[0][0] #x0
    ydata1 = listtempA2[0][3] #y0
    ydata2 = listtempA2[0][2] #x2
    ydata3 = listtempA2[0][5] #y2
        

if redata == abs(abs10):
    ydata0 = listtempA2[0][1] #x1
    ydata1 = listtempA2[0][4] #y1
    ydata2 = listtempA2[0][0] #x0
    ydata3 = listtempA2[0][3] #y0   
    

if redata == abs(abs12):
    ydata0 = listtempA2[0][1] #x1
    ydata1 = listtempA2[0][4] #y1
    ydata2 = listtempA2[0][2] #x2
    ydata3 = listtempA2[0][5] #y2

    

if redata == abs(abs20):
    ydata0 = listtempA2[0][2] #x2
    ydata1 = listtempA2[0][5] #y2
    ydata2 = listtempA2[0][0] #x0
    ydata3 = listtempA2[0][3] #y0
    

if redata == abs(abs21):
    ydata0 = listtempA2[0][2] #x2
    ydata1 = listtempA2[0][5] #y2
    ydata2 = listtempA2[0][1] #x1
    ydata3 = listtempA2[0][4] #y1    

    
a = np.mat([[data0,data1,1,0],[data1,-data0,0,1],[data2,data3,1,0],[data3,-data2,0,1]])#系数矩阵
b = np.mat([ydata0,ydata1,ydata2,ydata3]).T    #常数项列矩阵
x = solve(a,b)        #方程组的解
if x[2] >= 0:
    delx = int(x[2]+0.5)
else:
    delx = int(x[2]-0.5) 
    
if x[3] >= 0:
    dely = int(x[3]+0.5)
else:
    dely = int(x[3]-0.5)
    
    
if x[1]<1 and x[1]>-1 and x[0]<1 and x[0]>-1:
    theta0 = math.asin((x[1]))*57.3
    theta1 = math.acos((x[0]))*57.3
    theta = min(theta0,theta1)
else:
    theta = 0
print('delx = ', delx)
print('dely = ', dely)
print('theta = ',theta)

img = Image.fromarray(imagedataA1.astype('uint16'))
rotimage = img.rotate(theta)
rotimagedataA1 = np.array(rotimage)

#rotimagedataA1 = transform.rotate(imagedataA1, theta,resize = False)
###图像平移###
newimage = np.zeros((hang,lie),dtype = np.uint16)
if delx <= 0 and dely <= 0:
    newimage[0:hang+delx,0:lie+dely] = rotimagedataA1[-delx:hang,-dely:lie]
    
if delx <= 0 and dely >= 0:
    newimage[0:hang+delx,dely:lie] = rotimagedataA1[-delx:hang,0:lie-dely]
    
if delx >= 0 and dely >= 0:
    newimage[delx:hang,dely:lie] = rotimagedataA1[0:hang-delx,0:lie-dely]   
    
if delx >= 0 and dely <= 0:
    newimage[delx:hang,0:lie+dely] = rotimagedataA1[0:hang-delx,-dely:lie] 


jianimage = np.float32(newimage) - np.float32(imagedataA2)
resultimage = whadjustimage(jianimage)
plt.figure(5)    
plt.imshow(resultimage, cmap='gray') 
lujingname = 'E:/shunbianyuan/diffimage/'+ fitsname + '.jpg'
io.imsave(lujingname,resultimage)

yuancha = np.float32(imagedataA1) - np.float32(imagedataA2)
resultyuancha = whadjustimage(yuancha)
plt.figure(6)    
plt.imshow(resultyuancha, cmap='gray')

end = time.time()
print("运行时间:%.2f秒"%(end-start))