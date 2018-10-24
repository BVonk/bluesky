from bluesky import traf  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools import geo
from bluesky.tools.aero import nm
import numpy as np


def detect_los(ownship, intruder, RPZ, HPZ):
    '''
    Stripped down version of the conflict detection in ASAS without fixed
    update rate to allow faster running of bluesky when only LOS detection
    is required.
    '''

    # Identity matrix of order ntraf: avoid ownship-ownship detected conflicts
    I = np.eye(ownship.ntraf)

    # Horizontal conflict ------------------------------------------------------

    # qdlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
    qdr, dist = geo.qdrdist_matrix(np.mat(ownship.lat), np.mat(ownship.lon),
                                   np.mat(intruder.lat), np.mat(intruder.lon))

    # Convert back to array to allow element-wise array multiplications later on
    # Convert to meters and add large value to own/own pairs
    qdr = np.array(qdr)
    dist = np.array(dist) * nm + 1e9 * I

    # Vertical crossing of disk (-dh,+dh)
    dalt = ownship.alt.reshape((1, ownship.ntraf)) - \
        intruder.alt.reshape((1, ownship.ntraf)).T  + 1e9 * I

    swlos = (dist < RPZ) * (np.abs(dalt) < HPZ)
    los_pairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swlos))]

    return los_pairs


def calc_turn_wp(delta_qdr, acidx):
    """ Compute the next waypoint required to make a certain turn """
    latA = traf.lat[acidx]
    lonA = traf.lon[acidx]
    turnrad = traf.tas[acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[acidx])) * g0) # [m]

    # Turn left
    if delta_qdr > 0:
        # Compute centre of turning circle
        latR, lonR = qdrpos(latA, lonA, traf.hdg[acidx] + 90, turnrad/nm) # [deg, deg]
        # Rotate vector position vector along turning circle
        latB, lonB = qdrpos(latR, lonR, traf.hdg[acidx] - 90 + abs(delta_qdr), turnrad/nm) # [deg, deg]
    # Turn right
    else:
        # Compute centre of turning circle
        latR, lonR = qdrpos(latA, lonA, traf.hdg[acidx] - 90, turnrad/nm) # [deg, deg]
        # Rotate vector position vector along turning circle
        latB, lonB = qdrpos(latR, lonR, traf.hdg[acidx] + 90 - abs(delta_qdr), turnrad/nm) # [deg, deg]
    return latB, lonB


def abc_formula(a, b, c):
    """Find x using the abc-formula"""
    D = b*b - 4*a*c
    x1 =( -b - D**0.5 )/ (2*a)
    x2 =( -b + D**0.5 )/ (2*a)
    x = x1*x1>0 + x2*x2>0
    return x