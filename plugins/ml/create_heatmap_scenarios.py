# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:11:31 2019

@author: Bart
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:38:52 2018

@author: Bart
"""

import argparse
from keras.models import model_from_yaml
from custom_keras_layers import ZeroMaskedEntries
import os
from bicnet import BiCNet
import yaml
from keras.utils import CustomObjectScope
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import numpy as np
os.chdir('../../bluesky/tools')
from geo import qdrdist_matrix, qdrpos
import pandas as pd

"""
This script produces heatmaps to gain insight into what an agent has learned
in the scale of the problem.
"""


def normalize(x, low, high):
    """
    Normalizes the input to a zero mean
    :param x:
    :param low:
    :param high:
    :return:
    """

    y = (x-low) / (high-low) * 2 - 1
    return y


def denormalize(x, low, high):
    """
    Denormalized the input to original range
    :param x:
    :param low:
    :param high:
    :return:
    """
    y = (x + (low + high)/(high - low)) * (high - low) / 2
    return y


def rescale(x, xlow, xhigh, low, high):

    x = (x-xlow) / (xhigh-xlow)

    x = x  * (high - low) + low
    return x


def pol2deg(x):
    x=x.strip('E')
    x=x.strip('N')
    x=x.split('.')
    y = float(x[0]) + float(x[1]) / 60 + float(x[2])/3600
    return y


def degto180(angle):
    """Change to domain -180,180 """
    return (angle + 360) % 360 - 180.

def qdrtodeg(angle):
    return(angle + 360) % 360

def degtopi(deg):
    return deg/180*np.pi



ehamlat = 52 + 18/60. + 48/3600.
ehamlon = 4 + 45/60. + 85/3600.

dist = 120*np.ones(360)
heading = np.arange(0,360)

lat, lon = qdrpos(ehamlat * np.ones(360), ehamlon*np.ones(360), heading, dist)



# Create scenario
os.chdir('../../plugins/ml')
for i in range(360):
    f = open('.\scen\heatmapscen{:03d}.scn'.format(i), 'w')
    f.write("00:00:00.00>SWRAD WPT; ZOOM 0.2; SWRAD LABEL; PAN EHKD; FF; DT 1.0; ASAS OFF\n")
    f.write("00:00:00.00>CRE BCS1443, B734, {}, {}, {}, FL200, 200\n".format(lat[i], lon[i], heading[i]+180))
    f.write("00:00:00.00>DEST BCS1443 EHAM RWY06 10\n")
    f.write("00:00:00.00>BCS1443 VNAV OFF\n")
    f.close()


plt.scatter(lat, lon)
plt.scatter(ehamlat, ehamlon)
plt.show()