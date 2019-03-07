# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:09:27 2019

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



def heatmap(path, episode, out):


    actormodel = os.path.join(path, 'actor_model{0:05d}.yaml'.format(episode))
    actorweights = os.path.join(path, 'actor_model{:05d}.h5'.format(episode))

    # Load the keras model architecture from file and load the weights
    stream = open(actormodel, 'r')
    actor = model_from_yaml(stream, custom_objects={'ZeroMaskedEntries': ZeroMaskedEntries})
    stream.close()
    actor.load_weights(actorweights)

    # Range definitions for the plot
    minlat, maxlat = 50.75428888888889, 55.
    minlon, maxlon = 2., 7.216944444445

    minhdg,  maxhdg  = 0, 360
    minqdr,  maxqdr  = 0, 360
    mindist, maxdist = 0, 110
    mincas,  maxcas  = 80, 200

    minplotlat,  maxplotlat  = 50.5, 55.5
    minplotlon,  maxplotlon  = 1.5, 7.5

    ehamlat = 52 + 18/60. + 48/3600.
    ehamlon = 4 + 45/60. + 85/3600.


    sns.set_context("paper")
    fig, ax = plt.subplots(1,3)

    for i in range(3):

    hdgdiff = -10 + 10*i  #offset #15
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

    qdrhdg = hdgdiff * np.ones(dist.shape)
    states = np.array([normalize(lat.ravel(), minlat, maxlat),
                       normalize(lon.ravel(), minlon, maxlon),
                       qdrhdg.ravel(),
                       normalize(dist.ravel(), mindist, maxdist),
                       ]).transpose()

    actions = actor.predict(np.expand_dims(states, axis=1))
    df= pd.DataFrame(actions.reshape(n,n), index=linlat, columns = linlon)
    # EHAA FIR boundaries for drawing
    EHAA = pd.read_csv('../../data/navdata/fir/EHAA.txt', delimiter=' ', names=['lat', 'lon'], index_col=None)
    EHAA = EHAA.applymap(pol2deg)



    fig, ax = plt.subplots()
    sns.set_context("paper")
    # Create the heat map
    cmap = sns.diverging_palette(240, 10 , center='light', as_cmap=True)
    #cmap=sns.cubehelix_palette(light=1, as_cmap=True)
    ax = sns.heatmap(data=df, center=0., square=True, cmap=cmap, vmin=-1, vmax=+1)
    ax.scatter(rescale(ehamlon, lon.min(), lon.max(), 0, n),
               rescale(ehamlat, lat.min(), lat.max(), 0, n), color='b', marker='s', facecolors='none')
    #ax2 = sns.lineplot(x='lon', y='lat', data=EHAA, sort=False, ax=ax)
    ax.plot(rescale(EHAA.lon, lon.min(), lon.max(), 0, n), rescale(EHAA.lat, lat.min(), lat.max(), 0, n), color='b', linewidth = 1.5)



    #ax.quiver(rescale(lon, lon.min(), lon.max(), 0, n), rescale(lat, lat.min(), lat.max(), 0, n), v, u)

    # Create steps for drawing aircraft.

    m = 10
    step = int(n/m)

    steplat = linlat[int(step/2):n-(int(step/2))+1:step]
    steplon = linlon[int(step/2):n-(int(step/2))+1:step]
    steplat, steplon = np.meshgrid(steplat, steplon)

    stepqdr, stepdist = qdrdist_matrix(steplat, steplon, ehamlat*np.ones(steplat.shape), ehamlon*np.ones(steplon.shape))
    stepqdr, stepdist = np.array(stepqdr), np.array(stepdist)
    stephdg = qdrtodeg(qdr.copy())
    stepqdrhdg = hdgdiff * np.ones(stepdist.shape)
    stephdg = stepqdr + stepqdrhdg
    stepstates = np.array([normalize(steplat.ravel(), minlat, maxlat),
                           normalize(steplon.ravel(), minlon, maxlon),
                           stepqdrhdg.ravel(),
                           normalize(stepdist.ravel(), mindist, maxdist),
                           ]).transpose()

    stepactions = actor.predict(np.expand_dims(stepstates, axis=1))
    df= pd.DataFrame(stepactions.reshape(m,m), index=steplat, columns = steplon)

    lonscaled = rescale(steplon, lon.min(), lon.max(), 0, n)
    latscaled = rescale(steplat, lat.min(), lat.max(), 0, n)
    #ax.scatter(latscaled.ravel(), lonscaled.ravel())

    #Construct quivers
    dheading = stepactions.reshape(m*m, 1) * 30
    #dheading = stepactions.reshape(m*m, 1) * 180
    act = qdrtodeg(stepqdr.reshape(m*m, 1) + stepqdrhdg.reshape(m*m, 1) + dheading)
    u = np.cos(degtopi(act))
    v = np.sin(degtopi(act))






    ax.quiver(latscaled, lonscaled, v,u)
    #
    for i in range(steplon.size):
        angle=-stephdg.ravel()[i]
        marker1 = mpl.markers.MarkerStyle(marker='^')
        marker1._transform = marker1.get_transform().rotate_deg(angle)
        plt.scatter(latscaled.ravel()[i],lonscaled.ravel()[i], marker=marker1, color=(0.41015625, 0.828125  , 0.4328125))#facecolors='none')
        marker2 = mpl.markers.MarkerStyle(marker='2')
        marker2._transform = marker2.get_transform().rotate_deg(angle)
        plt.scatter(latscaled.ravel()[i],lonscaled.ravel()[i], marker=marker2, color=(0.41015625, 0.828125  , 0.4328125), s=100)


    #plt.gca().invert_yaxis()
    ax.invert_yaxis()
    ax.set_aspect(1.25)
    locs, labels = plt.xticks(np.linspace(0, n, 11), np.linspace(minplotlon, maxplotlon, 13))
    locs, labels = plt.yticks(np.linspace(0, n, 10), np.linspace(minplotlat, maxplotlat, 11))

    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    #ax.yaxis.set_ticks(np.arange(52, 57, 5/10.))
    #ax.set_yticklabels(np.arange(52, 57, 5/10.))
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

    plt.savefig(out)
#    plt.show()

#stream = open(actormodel, 'r')
    return actions