# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:35:02 2018

@author: Bart
"""

""" Plugin to resolve conflicts """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools.geo import kwikdist, qdrpos, qdrdist
from bluesky.tools.misc import degto180
from bluesky import traf
from bluesky import navdb
from bluesky.tools.aero import nm, g0

from vierd import ETA

import pickle
import random
import numpy as np
import os
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, RepeatVector, Lambda
from keras.optimizers import Adam, RMSprop
from keras.layers.merge import Add, Multiply, Subtract
import keras.backend as K
import tensorflow as tf


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    global env, agent
    env = Environment()
    agent = Agent()
    config = {
        # The name of your plugin
        'plugin_name':     'THESIS',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 1,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.
        'update':          update,

        # The preupdate function is called before traffic is updated. Use this
        # function to provide settings that need to be used by traffic in the current
        # timestep. Examples are ASAS, which can give autopilot commands to resolve
        # a conflict.
        'preupdate':       preupdate,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         reset
        }



    stackfunctions = {
        # The command name for your function
#        "ENV_STEP": [
#            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
#            'Step the environment',
#            '',
#            env.step,
#
#            # a longer help text of your function.
#            'Print something to the bluesky console based on the flag passed to MYFUN.'],
#
#        "ENV_ACT": [
#            'hdg',
#            'hdg',
#            env.act,
#            'Select a heading'],
#
#        "ENV_RESET": [
#            'Reset the training environment',
#            '',
#            env.reset,
#            'Reset the training environment']
            }

    stack.stack("CRE BART001, B737, 51.862982, 2.830079, 90, fl200, 150")
    stack.stack("BART001 DEST EHAM RWY06")
    stack.stack("BART001 VNAV ON")
    stack.stack("CRE BART002, B737, 51.2, 2.83, 90, FL200, 150")
    stack.stack("BART002 DEST EHAM RWY06")
    stack.stack("BART002 VNAV ON")
    stack.stack("CRE BART003, B737, 50.9, 2.83, 90, FL200, 150")
    stack.stack("BART003 DEST EHAM RWY06")
    stack.stack("BART003 VNAV ON")

    # init_plugin() should always return these two dicts.
    return config, stackfunctions



def update():
    # get_routes()
    # print(traf.ap.route[0].wpname, traf.ap.route[0].iactwp, traf.ap.route[0].wptype)
    agent.act()
    pass


def init():
    load_routes()
    stack.stack("BART001 DEST EHAM RWY06")


def preupdate():
    # Initialize routes
    if sim.simt < 1.5:
        init()
    pass


def reset():
    pass


# wptype 3 == Destination --> So check for that:
def load_routes():
    wpts = np.loadtxt('plugins/ml/routes/testroute.txt')
    i=0
    routes = dict()
    for j in range(wpts.shape[0]):
           for k in range(int(wpts.shape[1]/2)):
               stack.stack('DEFWPT RL{}{}{}, {}, {}, FIX'.format(i, j, k, wpts[j,k*2], wpts[j,k*2+1]))

def get_routes():
    wpts = np.loadtxt('plugins/ml/routes/testroute.txt')
    i = 0
    routes = dict()
    for j in range(wpts.shape[0]):
        a = []
        for k in range(int(wpts.shape[1]/2)):
           a.append('RL{}{}{}'.format(i, j, k))
        routes[j] = a
    # print(routes)

class Environment:
    def __init__(self):
        self.n_aircraft = traf.ntraf
        self.global_obs = np.zeros(10)
        self.ac_dict = {}
        self.prev_traf = 0
        for id in traf.id:
            ac_dict[id]= [0,0,0]

    def step(self):
        pass

    def generate_reward(self):
        pass

    def generate_observation(self):
        pass

    def generate_commands(self):
        pass






class Agent():
    def __init__(self):
        self.ac_dict = {}
        self.speedvalues = [175, 200, 225, 250]
        self.wpactionsize=3
        self.spdactionsize=len(self.speedvalues)


    def __update_aircraft_waypoints(self):
        """
        Checks if aircraft have reached their waypoints. Aircraft having reached their waypoints will have their
        destination as waypoint, which corresponds to wptype == 3
        :return:
        """
        for i in np.arange(traf.ntraf):

            actwp = traf.ap.route[i].iactwp
            if traf.ap.route[i].wptype[actwp] == 3:
                ac_values = self.ac_dict.get(traf.id[i])
                ac_values[0] = 1
                self.ac_dict[traf.id[i]] = s[traf.id[i]]

        # return list of waypoints


    def __update_acdict_entries(self):
        """
        Check which aircraft have to be added to dictionary or deleted when aircraft have landed.
        Sets the aircraft status to [x, [wpindex]. New aircraft are initialized without wpname
        x = 1 when aircraft must select new waypoint, otherwise 0
        """
        for id in traf.id:
            # print(traf.id)
            if id not in self.ac_dict:
                self.ac_dict[id]=[1,[0,0,0]]
        # If aircraft sizes are not equal then aircraft also have to be deleted
        if len(self.ac_dict) < traf.ntraf:
            for key in self.ac_dict.keys():
                if key not in traf.id:
                    self.ac_dict.pop(key)

    """
    Aircraft route index ijk:
        i = main route index
        j = section of main route
        k = segment choice in section
    """



    def get_mask(self):
        """
        Creates a mask for valid actions an aircraft can select. Masks the aircraft in the dictionary set that require a new waypoint to be selected with 1.
        :return:
        """
        actionmask = np.zeros((self.wpactionsize, traf.ntraf))
        for i in np.arange(traf.ntraf):
            value = self.ac_dict.get(traf.id[i])
            if value[0]==0:
                actionmask[value[1][2], i]=1
            else:
                actionmask[:,i]=1
        return actionmask


    def act(self):
        # Select actions for each aircraft
        self.__update_acdict_entries()
        print(self.ac_dict)
        # __update_aircraft_waypoints()

        # Infer action selection
        wp_action = np.random.random((self.wpactionsize,traf.ntraf))
        spd_action = np.random.random((self.spdactionsize,traf.ntraf))

        # Retrieve mask for action selection
        mask = self.get_mask()
        print(mask)

        # Select action
        wp_action_masked = wp_action * mask
        print(wp_action_masked)

        wp_ind = np.argmax(wp_action, axis=0) # wp_action_masked
        spd_ind = np.argmax(spd_action, axis=0)
        speed_actions = [self.speedvalues[i] for i in spd_ind]

        # Produce action command
        # Get the number of segments in main route index and construct
        maxsegments = 4 # Dummy to replace

        # Action is already selected. Just generate the commands ;)

        for i in np.arange(traf.ntraf):
            # Set wpindex
            ac_name = traf.id[i]
            wp_values = self.ac_dict.get(ac_name)
            # Only add wpts to route if required
            actwp = traf.ap.route[i].iactwp
            if traf.ap.route[i].wptype[actwp] == 3: # actwp == 3 corresponds to destination
                wp_name = wp_values[1]
                wp_name[2] = wp_ind[i]
                self.ac_dict[traf.id[i]] = [0, wp_name]
                stack.stack("{} ADDWPT RL{}".format(ac_name,  ''.join(map(str, wp_name))))
                wp_name[1] = wp_name[1] + 1

            # Create speed command
            stack.stack("{} SPD {}".format(ac_name, speed_actions[i]))

        # Wp command stack.stack(callsign ADDWPT wpname)
        # Spd command
















