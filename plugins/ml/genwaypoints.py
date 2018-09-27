# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:19:27 2018

@author: Bart
"""

import os, sys
from os.path import sep
sys.path.append(os.getcwd() + sep + '..' + sep + '..' + sep + 'bluesky' + sep + 'tools' + sep)

from geo import qdrdist, qdrpos
import numpy as np

def generate_waypoints(latd1, lond1, latd2, lond2, n):
    """
    Generates waypoints a set of waypoints between two specified locations
    based on the maximum angle of deviation from the. The The number
    rows of waypoints should be specified in between.
    """
    anglestep = 20

    #Step1: compute distance between wp1 and wp2
#    d = latlondist(latd1, lond1, latd2, lond2) #distance in meters
    qdr, d = qdrdist(latd1, lond1, latd2, lond2)
    print('step1 : d = {}nm, qdr = {} deg'.format(d, qdr))


    # Step2: compute equidistant lines
    distmat = np.asarray([i*d/(n+1) for i in range(1,n+1)])
    latlonmat = np.asarray([qdrpos(latd1, lond1, qdr, d) for d in distmat])


    # Step3: Use the triangle to compute the crosstrack distance
    degtorad = 0.017453292519943295
    angledist1 = np.arctan(anglestep*degtorad) * distmat
    angledist2 = np.flip(angledist1,0)
    angledist = np.minimum(angledist1, angledist2)

    # Step4: Compute waypoint locations
    waypoints = np.empty((5,n,2))
    for i in range(n):
        reflat = latlonmat[i][0]
        reflon = latlonmat[i][1]
        waypoints[0,i,:] = qdrpos(reflat, reflon, qdr+90, 2*angledist[i])
        waypoints[1,i,:] = qdrpos(reflat, reflon, qdr+90, angledist[i])
        waypoints[2,i,:] = np.array([reflat, reflon])
        waypoints[3,i,:] = qdrpos(reflat, reflon, qdr-90, angledist[i])
        waypoints[4,i,:] = qdrpos(reflat, reflon, qdr-90,2* angledist[i])

    # Return matrix of waypoints
    return waypoints


def min_to_deg(deg, mi, sec):
    return deg + mi/60. + sec/3600.


if __name__ == '__main__':
    latd1 = min_to_deg(53, 9, 50)
    lond1 = min_to_deg(6, 40, 0)
    latd2 = min_to_deg(52,30,40.37)
    lond2 = min_to_deg(5,34,8.69)
    n = 3

    wp = generate_waypoints(latd1, lond1, latd2, lond2, n)
    import matplotlib.pyplot as plt
    plt.figure(2)
    plt.scatter(wp[:,:,1], wp[:,:,0])
    plt.scatter(lond1, latd1)
    plt.scatter(lond2, latd2)
    plt.show()
