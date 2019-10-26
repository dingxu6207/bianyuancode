# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:34:04 2019

@author: dingxu
"""

import os

listroute = [0 for i in range(1)]
listroute[0] = 'http://das101.china-vo.org/bass/rawdata/20190210/index.html'

for i in range(1):
    cmd = 'python downdata.py '+listroute[i]
    os.system(cmd)