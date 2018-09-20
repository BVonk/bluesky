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
from plugins.ml.actor import ActorNetwork
from plugins.ml.critic import CriticNetwork
from plugins.ml.ReplayMemory import ReplayMemory


from plugins.vierd import ETA

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
    # Train in the update function
    # if agent.replay_memory.num_experiences > agent.batch_size:
    #     print('replay {}'.format(agent.replay_memory.num_experiences))
    #     agent.train()
    pass



def init():
    load_routes()
    stack.stack("BART001 DEST EHAM RWY06")


def preupdate():
    # Initialize routes
    if sim.simt < 1.5:
        init()
        new_state = env.init()
        stack.stack("CRE BART004, B737, 51.862982, 2.830079, 90, fl200, 150")
        stack.stack("BART004 DEST EHAM RWY06")
        stack.stack("BART004 VNAV ON")

    else:
        # Construct new state
        state, reward, new_state, done = env.step()
        agent.update_replay_memory(state, reward, done, new_state)
        if agent.replay_memory.num_experiences > agent.batch_size:
            agent.train()

    agent.act(new_state)


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
        self.ac_dict = {}
        self.prev_traf = 0
        self.observation = []
        self.done = False
        for id in traf.id:
            self.ac_dict[id]= [0,0,0]

    def init(self):
        self.generate_observation()
        return self.observation

    def step(self):
        prev_observation = self.observation
        self.generate_observation()
        # Check termination conditions
        self.check_termination()
        self.generate_reward()
        return prev_observation, self.reward, self.observation, self.done

    def generate_reward(self):
        global_reward = 1
        local_reward = np.ones((traf.ntraf,1))
        self.reward = local_reward + global_reward

    def generate_observation(self):
        # Produce a running average off all variables in the field with regard to the initial state convergence that is being tackled in the problem.
        self.observation = np.array([traf.lat, traf.lon, traf.tas, traf.cas, traf.alt]).transpose()

    def check_termination(self):
        self.done = False

    def generate_commands(self):
        pass





class Agent:
    def __init__(self):
        self.ac_dict = {}
        self.speed_values = [175, 200, 225, 250]
        self.wp_action_size=3
        self.spd_action_size=len(self.speed_values)
        self.state_size=5
        #TODO: Set the correct action size to correspond with the desired action output and neural network architecture
        self.action_size=1 #self.wp_action_size + self.spd_action_size
        self.max_agents = 80
        self.action = []
        self.memory_size = 10000


        self.batch_size = 4
        self.tau = 0.9
        self.gamma = 0.99
        self.critic_learning_rate = 0.001
        self.actor_learning_rate = 0.0001
        self.loss = 0
        self.train_indicator = True



        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        K.set_session(self.sess)

        self.actor = ActorNetwork(self.sess, self.state_size, self.action_size, self.batch_size, self.tau, self.actor_learning_rate)
        self.critic = CriticNetwork(self.sess, self.state_size, self.action_size, self.batch_size, self.tau, self.critic_learning_rate)
        self.replay_memory = ReplayMemory(self.memory_size)

        #Now load the weight
        try:
            self.actor.model.load_weights("actormodel.h5")
            self.critic.model.load_weights("criticmodel.h5")
            self.actor.target_model.load_weights("actormodel.h5")
            self.critic.target_model.load_weights("criticmodel.h5")
            print("Weights load successfully")
        except:
            print("Cannot find the weights")


    def pad_zeros(self, array, max_timesteps):
        if array.shape[0] == max_timesteps:
            return array
        else:
            result = np.zeros((max_timesteps, array.shape[1]))
            result[:array.shape[0], :] = array
            return result

    def train(self):
        batch = self.replay_memory.getBatch(self.batch_size)
        for seq in batch:
            print(seq[0].shape)

        # In order to create sequences with equal length for batch processing sequences are padded with zeros to the
        # maximum sequence length in the batch. Keras can handle the zero padded sequences by ignoring the zero
        # calculations
        sequence_length = [seq[0].shape[0] for seq in batch]
        max_t = max(sequence_length)

        states = np.asarray([self.pad_zeros(seq[0], max_t) for seq in batch])
        actions = np.asarray([self.pad_zeros(seq[1], max_t) for seq in batch])
        rewards = np.asarray([self.pad_zeros(seq[2], max_t) for seq in batch])
        new_states = np.asarray([self.pad_zeros(seq[3], max_t) for seq in batch])
        dones = np.asarray([seq[4] for seq in batch])
        y_t = actions.copy()

        # print(states.shape, states)
        # print('batch', batch)
        # print('new_states - ', new_states.shape)
        # print('predicted', self.actor.target_model.predict(new_states))
        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])
        # print(target_q_values.shape)

        #Compute the target values
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                # print(rewards[k].shape, target_q_values[k].shape, y_t[k].shape)
                y_t[k] = rewards[k] + self.gamma * target_q_values[k]

        if self.train_indicator:
            print(states.shape, actions.shape, y_t.shape)
            # self.loss += self.critic.model.train_on_batch([states, actions], y_t)
            actions_for_grad = self.actor.model.predict(states)
            grads = self.critic.gradients(states, actions_for_grad)
            self.critic.train(states, actions_for_grad, y_t)
            self.actor.train(states, grads)
            self.actor.update_target_network()
            self.critic.update_target_network()
            print("training epoch succesful")

    def update_replay_memory(self, state, reward, done, new_state):
        self.replay_memory.add(state, self.action, reward, new_state, done)

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
                self.ac_dict[traf.id[i]] = [traf.id[i]]

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
        actionmask = np.zeros((self.wp_action_size, traf.ntraf))
        for i in np.arange(traf.ntraf):
            value = self.ac_dict.get(traf.id[i])
            if value[0]==0:
                actionmask[value[1][2], i]=1
            else:
                actionmask[:,i]=1
        return actionmask


    def act(self, state):
        # Select actions for each aircraft
        self.__update_acdict_entries()
        # print(self.ac_dict)
        # __update_aircraft_waypoints()

        # Infer action selection
        self.action = np.random.random((traf.ntraf, 1))
        wp_action = np.random.random((self.wp_action_size,traf.ntraf))
        spd_action = np.random.random((self.spd_action_size,traf.ntraf))

        # Retrieve mask for action selection
        mask = self.get_mask()
        # print(mask)

        # Select action
        wp_action_masked = wp_action * mask
        # print(wp_action_masked)

        wp_ind = np.argmax(wp_action, axis=0) # wp_action_masked
        spd_ind = np.argmax(spd_action, axis=0)
        speed_actions = [self.speed_values[i] for i in spd_ind]

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









