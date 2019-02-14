# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:32:18 2019

@author: Bart
"""

import numpy as np

x = np.arange(0,360, 10)
u = (x+180)%360 - 180
qdr = 170
x = 190
y = (x + 180 - qdr) % 360 - 180