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
from geo import qdrdist_matrix
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


#path = "C:\Users\Bart\Documents\bluesky\output\20181213_142139-3.0.4"
path = "C:/Users/Bart/Documents/bluesky/output/20181212_222438-3.0.0"
episode = 1000

actormodel = os.path.join(path, 'actor_model{0:05d}.yaml'.format(episode))
actorweights = os.path.join(path, 'actor_model{:05d}.h5'.format(episode))



stream = open(actormodel, 'r')
actor = model_from_yaml(stream, custom_objects={'ZeroMaskedEntries': ZeroMaskedEntries})
stream.close()
actor.load_weights(actorweights)

# Create states, the values to recurring news
# lat, lon, hdg, qdr, dist, spd
minlat, maxlat = 50.75428888888889, 55.
minlon, maxlon = 2., 7.216944444445

minhdg,  maxhdg  = 0, 360
minqdr,  maxqdr  = 0, 360
mindist, maxdist = 0, 110
mincas,  maxcas  = 80, 200

minplotlat,  maxplotlat  = 50.5, 55.5
minplotlon,  maxplotlon  = 1.5, 7.5

n = 1000
ehamlat = 52 + 18/60. + 48/3600.
ehamlon = 4 + 45/60. + 85/3600.
linlat = np.linspace(minplotlat, maxplotlat, n)
#linlat = np.arange(minlat, maxlat+0.25, 0.25)
linlon = np.linspace(minplotlon, maxplotlon, n)
lat, lon = np.meshgrid(linlat, linlon)
cas = 225*np.ones(lat.shape)
qdr, dist = qdrdist_matrix(lat, lon, ehamlat*np.ones(lat.shape), ehamlon*np.ones(lon.shape))
qdr, dist = np.array(qdr), np.array(dist)
hdg = qdrtodeg(qdr.copy())

states = np.array([normalize(lat.ravel(), minlat, maxlat),
                   normalize(lon.ravel(), minlon, maxlon),
                   normalize(hdg.ravel(), minhdg, maxhdg),
                   normalize(qdr.ravel()+180, minqdr, maxqdr),
                   normalize(dist.ravel(), mindist, maxdist),
                   normalize(cas.ravel(), mincas, maxcas)]).transpose()

actions = actor.predict(np.expand_dims(states, axis=1))
df= pd.DataFrame(actions.reshape(n,n), index=linlat, columns = linlon)

EHAA = pd.read_csv('../../data/navdata/fir/EHAA.txt', delimiter=' ', names=['lat', 'lon'], index_col=None)
EHAA = EHAA.applymap(pol2deg)

#Construct quivers
dheading = actions.reshape(n*n, 1) * 80 * np.ones((n*n, 1))
act = qdrtodeg(qdr.reshape(n*n, 1) + dheading)
u = np.cos(degtopi(act))
v = np.sin(degtopi(act))

lonscaled = rescale(lon, lon.min(), lon.max(), 0, n)
latscaled = rescale(lat, lat.min(), lat.max(), 0, n)



# Create the heat map
cmap=sns.cubehelix_palette(light=1, as_cmap=True)
cmap = sns.diverging_palette(240, 10 , center='light', as_cmap=True)
ax = sns.heatmap(data=df, center=0., square=True, cmap=cmap)
ax.scatter(rescale(ehamlon, lon.min(), lon.max(), 0, n),
           rescale(ehamlat, lat.min(), lat.max(), 0, n), color='b', marker='s', facecolors='none')
#ax2 = sns.lineplot(x='lon', y='lat', data=EHAA, sort=False, ax=ax)
ax.plot(rescale(EHAA.lon, lon.min(), lon.max(), 0, n), rescale(EHAA.lat, lat.min(), lat.max(), 0, n), color='b', linewidth = 2)
"""
ax.quiver(latscaled, lonscaled, v,u)
#ax.quiver(rescale(lon, lon.min(), lon.max(), 0, n), rescale(lat, lat.min(), lat.max(), 0, n), v, u)

for i in range(lon.size):
    angle=-hdg.ravel()[i]
    marker1 = mpl.markers.MarkerStyle(marker='^')
    marker1._transform = marker1.get_transform().rotate_deg(angle)
    plt.scatter(latscaled.ravel()[i],lonscaled.ravel()[i], marker=marker1, color='r', )#facecolors='none')
    marker2 = mpl.markers.MarkerStyle(marker='2')
    marker2._transform = marker2.get_transform().rotate_deg(angle)
    plt.scatter(latscaled.ravel()[i],lonscaled.ravel()[i], marker=marker2, color='r', s=100)
"""

#plt.gca().invert_yaxis()
ax.invert_yaxis()
ax.set_aspect(1.25)
locs, labels = plt.xticks(np.linspace(0, n, 11), np.linspace(minplotlon, maxplotlon, 13))
locs, labels = plt.yticks(np.linspace(0, n, 10), np.linspace(minplotlat, maxplotlat, 11))

#ax.yaxis.set_ticks(np.arange(52, 57, 5/10.))
#ax.set_yticklabels(np.arange(52, 57, 5/10.))
#ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')

plt.show()

#stream = open(actormodel, 'r')
