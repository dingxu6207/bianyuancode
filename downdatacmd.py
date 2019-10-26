# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 00:17:16 2019

@author: dingxu
"""
import os

listroute = [0 for i in range(8)]
listroute[0] = 'http://psp.china-vo.org/pspdata/2019/10/20191020/00~01%2030~50/'
listroute[1] = 'http://psp.china-vo.org/pspdata/2019/10/20191020/01~02%2030~40/'
listroute[2] = 'http://psp.china-vo.org/pspdata/2019/10/20191020/01~02%2040~50/'
listroute[3] = 'http://psp.china-vo.org/pspdata/2019/10/20191020/02~03%2040~50/'
listroute[4] = 'http://psp.china-vo.org/pspdata/2019/10/20191020/22~23%2030~50/'
listroute[5] = 'http://psp.china-vo.org/pspdata/2019/10/20191020/23~24%2030~50/'
listroute[6] = 'http://psp.china-vo.org/pspdata/2019/10/20191020/M31-18object/'
listroute[7] = 'http://psp.china-vo.org/pspdata/2019/10/20191020/M33-06object/'

for i in range(8):
    cmd = 'python downdata.py'+' '+listroute[i]
    os.system(cmd)