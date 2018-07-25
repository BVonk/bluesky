# -*- coding: utf-8 -*-
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


fn = '20180705_20180705_0000_2359__EHAM___m3'
df = pd.read_csv('tempData/' + fn + '.csv', sep=';', usecols=['flightid', 'fl1', 'fl2', 'lat1', 'lat2', 'lon1', 'lon2', 'hdg', 'dist'])

############## Preprocessing
# Convert segment format to point format.


flightids = df.flightid.unique()


N = df['flightid'].value_counts().max() + 10

#Compute N:



#Prepare matrix
lat_total = np.empty((len(flightids), N))
lon_total = np.empty((len(flightids), N))
alt_total = np.empty((len(flightids), N))
hdg_total = np.empty((len(flightids), N))
#for i in range(len(flightids)):
#
#    traj = df.loc[df['flightid'] == flightids[i]].copy().reset_index(drop=True)
#    cumdist = np.hstack((np.array([0]), np.array(traj.dist.cumsum())))
#    lat = np.hstack((np.array(traj.lat1), traj.lat2.iloc[-1]))
#    lon = np.hstack((np.array(traj.lon1), traj.lon2.iloc[-1]))
#    alt = np.hstack((np.array(traj.fl1), traj.fl2.iloc[-1]))
#    hdg = np.hstack((np.array(traj.hdg), traj.hdg.iloc[-1]))
#
#
#    # Interpolate with distance as base measure
#    interp_lat = PchipInterpolator(cumdist, lat)
#    interp_lon = PchipInterpolator(cumdist, lon)
#    interp_alt = PchipInterpolator(cumdist, alt)
#    interp_hdg = PchipInterpolator(cumdist, hdg)





#    while traj.shape[0]!=N-1:
#        cumdist = np.hstack((np.array([0]), np.array(traj.dist.cumsum())))
#        idx = np.where(traj.dist==max(traj.dist))[0][0]
#        d = cumdist[idx] + traj.loc[idx, 'dist']/2
#        ilat, ilon, ialt, ihdg = interp_lat(d), interp_lon(d), interp_alt(d), interp_hdg(d)
#        line = traj.loc[idx, :].copy()
#        line.lat1 = ilat
#        line.lon1 = ilon
#        line.fl1  = ialt
#        line = pd.DataFrame(line.values.reshape((1,traj.shape[1])), columns=line.index.values)
#
#
#        traj.loc[idx, 'lat2'] = ilat
#        traj.loc[idx, 'lon2'] = ilon
#        traj.loc[idx, 'fl2' ] = ialt
#        traj.loc[idx, 'dist'], line.dist = traj.loc[idx, 'dist']/2, traj.loc[idx, 'dist']/2
#        traj = pd.concat([traj.iloc[:idx+1, :], line, traj.iloc[idx+1:, :]]).reset_index(drop=True)
#
#    cumdist = np.hstack((np.array([0]), np.array(traj.dist.cumsum())))
#    lat_total[i, :] = np.hstack((np.array(traj.lat1), traj.lat2.iloc[-1]))
#    lon_total[i, :] = np.hstack((np.array(traj.lon1), traj.lon2.iloc[-1]))
#    alt_total[i, :] = np.hstack((np.array(traj.fl1), traj.fl2.iloc[-1]))
#    hdg_total[i, :] = np.hstack((np.array(traj.hdg), traj.hdg.iloc[-1]))
#
## Build db_arr [lat1, lon1, alt1, hdg1, lat2, ...., latN, lonN, latN, hdgN]
#db_arr = np.empty((len(flightids), N*4))
#for i in np.arange(0, N*4, 4):
#    db_arr[:,i]   = lat_total[:,int(i/4)]
#    db_arr[:,i+1] = lon_total[:,int(i/4)]
#    db_arr[:,i+2] = alt_total[:,int(i/4)]
#    db_arr[:,i+3] = hdg_total[:,int(i/4)]
#

db_arr = np.load('tempData/' + fn +'.npy')

#sns.set_context('poster')
#sns.set_style('white')
#sns.set_color_codes()
#plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
cluster_labels = clusterer.fit_predict(db_arr)
#clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#hierarchy = clusterer.cluster_hierarchy_
#alt_labels = hierarchy.get_clusters(0.100, 5)

nclusters = max(cluster_labels)
x = np.where(cluster_labels==-1)[0]
for X in x:
    plt.plot(df.loc[df['flightid'] == flightids[X]].lat1, df.loc[df['flightid'] == flightids[X]].lon1, color='y')

x = np.where(cluster_labels==-1)[0]
for i in np.arange(nclusters+1):
    x = np.where(cluster_labels==i)[0]
    ids = flightids[x]
    for X in x:
        if i==0:
            color='b'
        if i==1:
            color='r'
        if i==2:
            color='g'

        plt.plot(df.loc[df['flightid'] == flightids[X]].lat1, df.loc[df['flightid'] == flightids[X]].lon1, color=color)
# build trajectory matrix

#d = np.arange(0, int(cumdist[-1])+1, 0.1)
#ialt = interp_alt(d)
## Compute distance matrix
#plt.scatter(cumdist, alt)
#plt.plot(d, ialt)
#plt.show()