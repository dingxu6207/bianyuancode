# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:22:09 2019
@author: dingxu
修改zhankuan blv dianchensum即可
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from skimage import measure,io
import math
import copy
import time
import sys

#fitsname = 'PGC%2061869.fts'
fitsname = sys.argv[1]
routename1 = 'E:\\shunbianyuan\\ldf_download\\20191013\\'
routename2 = 'E:\\shunbianyuan\\ldf_download\\20191009\\'

fitsname1 = routename1+fitsname
fitsname2 = routename2+fitsname

start = time.time()
hduA1 = fits.open(fitsname1)
imagedataA1 = hduA1[0].data
copyimageA1 = copy.deepcopy(imagedataA1)
medfimagedataA1 = signal.medfilt(copyimageA1,(3,3)) 
quimagedataA1 = medfimagedataA1[400:800,400:1200]
imagedataA1F = copy.deepcopy(quimagedataA1)
hang,lie = imagedataA1F.shape

hduA2 = fits.open(fitsname2)
imagedataA2 = hduA2[0].data
copyimageA2 = copy.deepcopy(imagedataA2)
medfimagedataA2 = signal.medfilt(copyimageA2,(3,3))
quimagedataA2 = medfimagedataA2[400:800,400:1200]
imagedataA2F = copy.deepcopy(quimagedataA2)
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
    threhold = int((yuzhi[0]+40))  #阈值调节，根据星的个数100左右

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
                    subimgthr = float(img[i,j])-float(threhold)
                    subimgthr0 = subimgthr if (subimgthr>0) else 0
                    sumx = sumx+i*subimgthr0
                    sumy = sumy+j*subimgthr0
                    area = area+subimgthr0
        try:
            plotx[k] = round(sumx/(area))
            ploty[k] = round(sumy/(area))
        except ZeroDivisionError:
            plotx[k] = round(sumx/(area+0.0001))
            ploty[k] = round(sumy/(area+0.0001))
    return plotx,ploty,regionnum
    
def suansanjiaoxing(listS1,listS2,listS3):
    duanchu = 0
    sumchen = 0
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
        duanchu = (dS2S3/dS1S2)
        sumchen = (x1-x3)*(x2-x3) + (y1-y3)*(y2-y3)
    
    if (dS1S2 < dS2S3 and dS2S3 < dS1S3):
        duanchu = (dS1S3/dS1S2)
        sumchen = (x1-x3)*(x2-x3) + (y1-y3)*(y2-y3)
        
    if (dS2S3 < dS1S3 and dS1S3 < dS1S2):
        duanchu = (dS1S2/dS2S3)
        sumchen = (x2-x1)*(x3-x1) + (y2-y1)*(y3-y1)    
        
    if (dS2S3 < dS1S2 and dS1S2 < dS1S3):
        duanchu = (dS1S3/dS2S3)
        sumchen = (x2-x1)*(x3-x1) + (y2-y1)*(y3-y1)    
           
    if (dS1S3 < dS1S2 and dS1S2 < dS2S3):
        duanchu = dS2S3/dS1S3
        sumchen = (x1-x2)*(x3-x2) + (y1-y2)*(y3-y2)
        
    if (dS1S3 < dS2S3 and dS2S3 < dS1S2):
        duanchu = dS1S2/dS1S3
        sumchen = (x1-x2)*(x3-x2) + (y1-y2)*(y3-y2)    
        
    return x1,x2,x3,y1,y2,y3,duanchu,sumchen
    
resultA1 = whadjustimage(imagedataA1F)
A1plotx,A1ploty,A1regionnum = qiuzuobiao(resultA1)  

zhankuan = 9
plt.figure(1)
plt.imshow(resultA1, cmap='gray')  
fluxA1 = np.zeros(A1regionnum,dtype = np.uint)
listdataA1 = [0 for i in range(A1regionnum)]
for i in range(A1regionnum):
    plt.plot(A1ploty[i],A1plotx[i],'*')
    fluxA1[i] = np.sum(resultA1[A1plotx[i]-zhankuan:A1plotx[i]+zhankuan,A1ploty[i]-zhankuan:A1ploty[i]+zhankuan]) 
    listdataA1[i] = (A1plotx[i],A1ploty[i],fluxA1[i] )
listdataA1.sort(key=lambda x:x[2],reverse=True)
plt.show()

jiezhiA1 = A1regionnum
listsanjiaoA1 =  [0 for i in range(jiezhiA1)]
for i in range(jiezhiA1):
    if (i <= (jiezhiA1-3)):
        x1,x2,x3,y1,y2,y3,duan,sumchen = suansanjiaoxing(listdataA1[i],listdataA1[i+1],listdataA1[i+2])
    listsanjiaoA1[i] = (x1,x2,x3,y1,y2,y3,duan,sumchen)
   
    
    
plt.figure(2)
resultA2 = whadjustimage(imagedataA2F)
A2plotx,A2ploty,A2regionnum = qiuzuobiao(resultA2)
plt.imshow(resultA2, cmap='gray') 
fluxA2 = np.zeros(A2regionnum,dtype = np.uint)
listdataA2 = [0 for i in range(A2regionnum)]
for i in range(A2regionnum):
    plt.plot(A2ploty[i],A2plotx[i],'*') 
    fluxA2[i] = np.sum(resultA2[A2plotx[i]-zhankuan:A2plotx[i]+zhankuan,A2ploty[i]-zhankuan:A2ploty[i]+zhankuan])
    listdataA2[i] = (A2plotx[i],A2ploty[i],fluxA2[i] )
listdataA2.sort(key=lambda x:x[2],reverse=True)    
plt.show()

jiezhiA2 = A2regionnum
listsanjiaoA2 =  [0 for i in range(jiezhiA2)]
for i in range(jiezhiA2):
    if (i <= (jiezhiA2-3)):
        x1,x2,x3,y1,y2,y3,duan,sumchen = suansanjiaoxing(listdataA2[i],listdataA2[i+1],listdataA2[i+2])
    listsanjiaoA2[i] = (x1,x2,x3,y1,y2,y3,duan,sumchen)
   
    
plt.figure(3)
plt.imshow(resultA1, cmap='gray')
listtempA1 =  [0 for i in range(jiezhiA1)]
listtempA2 =  [0 for i in range(jiezhiA2)]
counttemp = 0
bilv = 0.03
dianchengsum = 1000
#jiezhistar = min(A1regionnum, A2regionnum)
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
def delhanshu(data0,data1,ydata0,ydata1):
    delx = ydata0-data0
    dely = ydata1-data1
        
    print('delx = ', delx)
    print('dely = ', dely)

    return delx,dely


data0 = listtempA1[0][0]  #x0
data1 = listtempA1[0][3]  #y0
data2 = listtempA1[0][1]  #x1
data3 = listtempA1[0][4]  #y1
data4 = listtempA1[0][2]  #x2
data5 = listtempA1[0][5]  #y2

'''
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
    ydata4 = listtempA2[0][2] #x2
    ydata5 = listtempA2[0][5] #y2

if redata == abs(abs02):
    ydata0 = listtempA2[0][0] #x0
    ydata1 = listtempA2[0][3] #y0
    ydata2 = listtempA2[0][2] #x2
    ydata3 = listtempA2[0][5] #y2
    ydata4 = listtempA2[0][1] #x1
    ydata5 = listtempA2[0][4] #y1

if redata == abs(abs10):
    ydata0 = listtempA2[0][1] #x1
    ydata1 = listtempA2[0][4] #y1
    ydata2 = listtempA2[0][0] #x0
    ydata3 = listtempA2[0][3] #y0   
    ydata4 = listtempA2[0][2] #x2
    ydata5 = listtempA2[0][5] #y2

if redata == abs(abs12):
    ydata0 = listtempA2[0][1] #x1
    ydata1 = listtempA2[0][4] #y1
    ydata2 = listtempA2[0][2] #x2
    ydata3 = listtempA2[0][5] #y2
    ydata4 = listtempA2[0][0] #x0
    ydata5 = listtempA2[0][3] #y0
    

if redata == abs(abs20):
    ydata0 = listtempA2[0][2] #x2
    ydata1 = listtempA2[0][5] #y2
    ydata2 = listtempA2[0][0] #x0
    ydata3 = listtempA2[0][3] #y0
    ydata4 = listtempA2[0][1] #x1
    ydata5 = listtempA2[0][4] #y1

if redata == abs(abs21):
    ydata0 = listtempA2[0][2] #x2
    ydata1 = listtempA2[0][5] #y2
    ydata2 = listtempA2[0][1] #x1
    ydata3 = listtempA2[0][4] #y1 
    ydata4 = listtempA2[0][0] #x0
    ydata5 = listtempA2[0][3] #y0

delx1,dely1 =  delhanshu(data0,data1,ydata0,ydata1)   
delx2,dely2 =  delhanshu(data2,data3,ydata2,ydata3) 
delx3,dely3 =  delhanshu(data4,data5,ydata4,ydata5)
delx = round((delx1+delx2+delx3)/3)
dely = round((dely1+dely2+dely3)/3)

###图像平移###
Ahang,Alie = imagedataA1.shape
newimage = np.zeros((Ahang,Alie),dtype = np.uint16)
if delx <= 0 and dely <= 0:
    newimage[0:Ahang+delx,0:Alie+dely] = imagedataA1[-delx:Ahang,-dely:Alie]
    
if delx <= 0 and dely >= 0:
    newimage[0:Ahang+delx,dely:Alie] = imagedataA1[-delx:Ahang,0:Alie-dely]
    
if delx >= 0 and dely >= 0:
    newimage[delx:Ahang,dely:Alie] = imagedataA1[0:Ahang-delx,0:Alie-dely]   
    
if delx >= 0 and dely <= 0:
    newimage[delx:Ahang,0:Alie+dely] = imagedataA1[0:Ahang-delx,-dely:Alie] 


jianimage = np.float32(newimage) - np.float32(imagedataA2)
#absimage = np.abs(jianimage)
resultimage = whadjustimage(jianimage)
plt.figure(5)    
plt.imshow(resultimage,cmap='gray') 
lujingname = 'E:/shunbianyuan/diffimage/'+ fitsname + '.jpg'
io.imsave(lujingname,resultimage)

end = time.time()
print("运行时间:%.2f秒"%(end-start))