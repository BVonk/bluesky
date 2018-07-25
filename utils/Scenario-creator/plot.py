# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:20:56 2018

@author: Bart
"""

import sys
sys.path.append('../../bluesky/tools/')
from geo import qdrdist
from aero import vtas2cas

import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import sklearn
from scipy.interpolate import PchipInterpolator
import rdp
import hdbscan
#import seaborn as sns
#import traj_dist.distance as tdist
#import pyproj

def convert_lat(lat):
    sign = 1 if lat[0]=='N' else -1
    lat=lat[1:].split('.')
    lat = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
    return sign*lat

def convert_lon(lon):
    sign = 1 if lon[0]=='E' else -1
    lon=lon[1:].split('.')
    lon = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600
    return sign*lon

def genwp(x1, x2, y1, y2, n):
    wp=[]
    for i in range(n):
        dx = (x2-x1)/(n-1)
        dy = (y2-y1)/(n-1)
        x=np.arange(x1, x2+0.5*dx, dx)[i]
        y=np.arange(y1, y2+0.5*dy, dy)[i]
        wp.append([x,y])
    return wp


fn = '20180705_20180705_0000_2359__EHAM___m3'
df = pd.read_csv('tempData/' + fn + '.csv', sep=';', usecols=['flightid', 'fl1', 'fl2', 'lat1', 'lat2', 'lon1', 'lon2', 'hdg', 'dist'])

############## Preprocessing
# Convert segment format to point format.


## Filter FIR Data
fir = pd.read_csv('../../data/navdata/fir/EHAA.txt', delimiter=' ', names=['lat', 'lon'])
fir.drop_duplicates(inplace=True)


fir['lat1'] = fir.lat.apply(convert_lat)
fir['lon1'] = fir.lon.apply(convert_lon)
fir['lat2'] = fir.lat1.shift(1)
fir.loc[0, 'lat2'] = fir.loc[fir.index[-1], 'lat1']
fir['lon2'] = fir.lon1.shift(1)
fir.loc[0, 'lon2']= fir.loc[fir.index[-1], 'lon1']

plt.plot(fir.lon2, fir.lat2)

flightids = df.flightid.unique()


for i in range(len(flightids)):
    traj = df.loc[df['flightid'] == flightids[i]].copy().reset_index(drop=True)
    plt.plot(traj.lon1, traj.lat1, color='b', linewidth=0.25)

xf=[4.64914, 4.56572]
yf=[52.257, 52.2226]



#final leg
wpf = genwp(xf[0], xf[1], yf[0], yf[1], 4)
wp1 = genwp(4.4145, 4.43909, 52.198, 52.2514, 4)
wp2 = genwp(4.35582, 4.39785, 52.2031, 52.2565, 4)
plt.scatter(*zip(*wpf), color='r')
plt.scatter(*zip(*wp1), color='r')
plt.scatter(*zip(*wp2), color='r')


#plt.plot()
##plt.plot([5.453, 5.46], [52.482, 52.456], color='r')
#plt.plot([4.4145, 4.43909], [52.198, 52.2514], color='r')
#plt.plot([4.35582, 4.39785], [52.2031, 52.2565], color='r')
plt.grid()
#plt.plot(yf,xf, color='r')
plt.scatter(4.645277,52.2536111, color='g')
plt.show()