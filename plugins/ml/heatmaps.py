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
import numpy as np
os.chdir('../../bluesky/tools')
from geo import qdrdist_matrix
import pandas as pd

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
#    xlow = np.min(x)
#    xhigh = np.max(x)
    x = (x-xlow) / (xhigh-xlow)

    x = x  * (high - low) + low
    return x




def pol2deg(x):
    x=x.strip('E')
    x=x.strip('N')
    x=x.split('.')
    y = float(x[0]) + float(x[1]) / 60 + float(x[2])/3600
    return y


#path = "C:\Users\Bart\Documents\bluesky\output\20181213_142139-3.0.4"
path = "C:/Users/Bart/Documents/bluesky/output/20181218_105739"
episode = 1000

actormodel = os.path.join(path, 'actor_model{0:05d}.yaml'.format(episode))
actorweights = os.path.join(path, 'actor_model{:05d}.h5'.format(episode))



stream = open(actormodel, 'r')
actor = model_from_yaml(stream, custom_objects={'ZeroMaskedEntries': ZeroMaskedEntries})
stream.close()
actor.load_weights(actorweights)

# Create states, the values to recurring news
# lat, lon, hdg, qdr, dist, spd
minlat,  maxlat  = 50.75428888888889, 55.
minlon,  maxlon  = 2., 7.216944444444445
minhdg,  maxhdg  = 0, 360
minqdr,  maxqdr  = 0, 360
mindist, maxdist = 0, 110
mincas,  maxcas  = 80, 200

ehamlat = 52 + 18/60. + 48/3600.
ehamlon = 4 + 45/60. + 85/3600.
linlat = np.linspace(minlat, maxlat, 100)
linlon = np.linspace(minlon, maxlon, 100)
lat, lon = np.meshgrid(linlat, linlon)
cas = 200*np.ones(lat.shape)
qdr, dist = qdrdist_matrix(lat, lon, ehamlat*np.ones(lat.shape), ehamlon*np.ones(lon.shape))
qdr, dist = np.array(qdr), np.array(dist)
hdg = qdr.copy()

states = np.array([normalize(lat.ravel(), minlat, maxlat),
                   normalize(lon.ravel(), minlon, maxlon),
                   normalize(hdg.ravel(), minhdg, maxhdg),
                   normalize(qdr.ravel()+180, minqdr, maxqdr),
                   normalize(dist.ravel(), mindist, maxdist),
                   normalize(cas.ravel(), mincas, maxcas)]).transpose()

actions = actor.predict(np.expand_dims(states, axis=1))
df= pd.DataFrame(actions.reshape(100,100), index=linlat, columns = linlon)

EHAA = pd.read_csv('../../data/navdata/fir/EHAA.txt', delimiter=' ', names=['lat', 'lon'], index_col=None)
EHAA = EHAA.applymap(pol2deg)


ax = sns.heatmap(data=df, center=0., square=True, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
ax.scatter(rescale(ehamlon, EHAA.lon.min(), EHAA.lon.max(), 0, 100), rescale(ehamlat, EHAA.lat.min(), EHAA.lat.max(), 0, 100))
#ax2 = sns.lineplot(x='lon', y='lat', data=EHAA, sort=False, ax=ax)
ax.plot(rescale(EHAA.lon, EHAA.lon.min(), EHAA.lon.max(), 0, 100), rescale(EHAA.lat, EHAA.lat.min(), EHAA.lat.max(), 0, 100))
plt.gca().invert_yaxis()
plt.show()

#stream = open(actormodel, 'r')
