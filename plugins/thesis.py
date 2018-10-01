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
from plugins.ml.normalizer import Normalizer


from plugins.vierd import ETA

import pickle
import random
import numpy as np
import os
from collections import deque
from keras.models import Sequential, Model, model_from_yaml
from keras.layers import Dense, Dropout, Input, RepeatVector, Lambda
from keras.optimizers import Adam, RMSprop
from keras.layers.merge import Add, Multiply, Subtract
import keras.backend as K
import tensorflow as tf


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    global env, agent, update_interval
    env = Environment()
    agent = Agent()
    update_interval = 10
    config = {
        # The name of your plugin
        'plugin_name':     'THESIS',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': update_interval,

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

    env.reset()

    # init_plugin() should always return these two dicts.
    return config, stackfunctions



def update():
    # Train in the update function
    # if agent.replay_memory.num_experiences > agent.batch_size:
    #     print('replay {}'.format(agent.replay_memory.num_experiences))
    #     agent.train()
    pass



def init():
    # load_routes()
    pass


def preupdate():
    # Initialize routes
    if sim.simt < update_interval*1.5:
        # init()
        new_state = env.init()
        # stack.stack("CRE BART004, B737, 51.862982, 2.830079, 90, fl200, 150")
        # stack.stack("BART004 DEST EHAM RWY06")
        # stack.stack("BART004 VNAV ON")

    else:
        # Construct new state
        state, reward, new_state, done = env.step()
        # print(agent.replay_memory.count(), state)
        print('reward')
        agent.update_cumreward(reward)
        agent.update_replay_memory(state, reward, done, new_state)
        if agent.replay_memory.num_experiences > agent.batch_size:
            agent.train()
        # agent.write_summaries(reward)

    agent.act(new_state)
    if env.done:
        # Reset environment states and agent states
        env.reset()
        agent.reset()


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
        self.state_size = 5
        self.ac_dict = {}
        self.prev_traf = 0
        self.observation = np.zeros((1,self.state_size))
        self.done = False
        self.state_normalizer = Normalizer(self.state_size)
        for id in traf.id:
            self.ac_dict[id]= [0,0,0]

    def init(self):
        self.observation = self.generate_observation()
        return self.observation

    def step(self):
        prev_observation = self.observation
        self.observation = self.generate_observation()
        # Check termination conditions
        self.check_reached()
        self.generate_reward()
        return prev_observation, self.reward, self.observation, self.done

    def generate_reward(self):
        global_reward = 1
        local_reward = np.ones((traf.ntraf,1))
        self.reward = local_reward + global_reward
        if self.done:
            self.reward = 100 * np.ones((traf.ntraf,1))
        else:
            self.reward = -1 * np.ones((traf.ntraf,1))


    def generate_observation(self):
        # Produce a running average off all variables in the field with regard to the initial state convergence that is being tackled in the problem.
        obs = np.array([traf.lat, traf.lon, traf.tas, traf.cas, traf.alt]).transpose()
        self.state_normalizer.observe(obs)
        obs = self.state_normalizer.normalize(obs)
        return obs

        # def reached(self, qdr, dist, flyby):
        #     # Calculate distance before waypoint where to start the turn
        #     # Turn radius:      R = V2 tan phi / g
        #     # Distance to turn: wpturn = R * tan (1/2 delhdg) but max 4 times radius
        #     # using default bank angle per flight phase
        #
        #
        #
        #     # Avoid circling by checking for flying away
        #     away = np.abs(degto180(bs.traf.trk % 360. - qdr % 360.)) > 90.  # difference large than 90
        #
        #     # Ratio between distance close enough to switch to next wp when flying away
        #     # When within pro1 nm and flying away: switch also
        #     proxfact = 1.01  # Turnradius scales this contant , factor => [turnrad]
        #     incircle = dist < turnrad * proxfact
        #     circling = away * incircle  # [True/False] passed wp,used for flyover as well
        #     nonflyby = (flyby == 1) * away * (np.array(dist) < 1000)
        #     #        print("NONFLYBY ", nonflyby)
        #     # Check whether shift based dist is required, set closer than WP turn distance
        #     swreached = np.where(bs.traf.swlnav * ((dist < self.turndist) + circling + nonflyby))[0]
        #
        #     # Return True/1.0 for a/c where we have reached waypoint
        #     return swreached

    def check_reached(self):
        qdr, _ = qdrdist(traf.lat, traf.lon,
                                traf.actwp.lat, traf.actwp.lon)  # [deg][nm])

        # check which aircraft have reached their destination by checkwaypoint type = 3 (destination) and the relative
        # heading to this waypoint is exceeding 150
        dest = np.asarray([traf.ap.route[i].wptype[traf.ap.route[i].iactwp] for i in range(traf.ntraf)]) == 3
        away = np.abs(degto180(traf.trk % 360. - qdr % 360.)) > 150.
        reached = np.where(away * dest)[0]
        # reached = self.Reached(qdr, dist, traf.actwp.flyby)
        # if reached!=[]:
        #     print(reached)
        if reached==[0]:
            # TODO: Extend for multi-aircraft
            self.done = True
        else:
            self.done = False

    def generate_commands(self):
        pass

    def reset(self):
        self.done = False
        stack.stack('open ./scenario/bart/BiCNet.SCN')
        load_routes()



class Agent:
    def __init__(self):
        self.ac_dict = {}
        self.speed_values = [175, 200, 225, 250]
        self.wp_action_size = 3
        self.spd_action_size = len(self.speed_values)
        self.state_size = 5
        #TODO: Set the correct action size to correspond with the desired action output and neural network architecture
        self.action_size = self.wp_action_size * self.spd_action_size
        self.max_agents = None
        self.action = []
        self.memory_size = 10000
        self.i = 0



        self.batch_size = 32
        self.tau = 0.9
        self.gamma = 0.99
        self.critic_learning_rate = 0.001
        self.actor_learning_rate = 0.0001
        self.loss = 0
        self.cumreward=0
        self.train_indicator = True



        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        # self.sess = tf.Session()
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

        # Set up summary Ops
        self.summary_ops, self.summary_vars = self.build_summaries()

        self.sess.run(tf.global_variables_initializer())
        summary_dir = './output/tf_summaries/'
        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

    def pad_zeros(self, array, max_timesteps):
        if array.shape[0] == max_timesteps:
            return array
        else:
            result = np.zeros((max_timesteps, array.shape[-1], self.action_size))
            result[:array.shape[0], :] = array
            return result

    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)
        # tf.summary.histogram("Histogram", self.actor.weights)
        # episode_ave_max_q = tf.Variable(0.)
        # tf.summary.scalar("Qmax Value", episode_ave_max_q)
        summary_vars = [episode_reward]# , episode_ave_max_q]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    def write_summaries(self, reward):
        summary_str = self.sess.run(self.summary_ops, feed_dict={
            self.summary_vars[0]: np.sum(reward),
                    })
        self.writer.add_summary(summary_str, self.i)
        self.writer.flush()
        self.i += 1

    def save_model(self, model, name):
        #yaml saves the model architecture
        model_yaml = model.to_yaml()
        with open("{}.yaml".format(name), 'w') as yaml_file:
            yaml_file.write(model_yaml)
        #Serialize weights to HDF5 format
        model.save_weights("{}.h5".format(name))

    def train(self):
        batch = self.replay_memory.getBatch(self.batch_size)
        for seq in batch:
            pass
            # print(seq)
            # print(seq[0].shape)

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
        mask = np.asarray([self.pad_zeros(seq[5], max_t) for seq in batch])
        y_t = rewards.copy()

        # print(states.shape, states)
        # print('batch', batch)
        # print('new_states - ', new_states.shape)
        # print('predicted', self.actor.target_model.predict(new_states))
        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict([new_states, mask])])
        # print(target_q_values.shape)

        #Compute the target values
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                # print(rewards[k].shape, target_q_values[k].shape, y_t[k].shape)
                y_t[k] = rewards[k] + self.gamma * target_q_values[k]

        print('y_t', y_t.shape)
        if self.train_indicator:
            # print(states.shape, actions.shape, y_t.shape)
            # self.loss += self.critic.model.train_on_batch([states, actions], y_t)
            actions_for_grad = self.actor.model.predict([states, mask])
            grads = self.critic.gradients(states, actions_for_grad)
            self.critic.train(states, actions, y_t)
            self.actor.train(states, mask, grads)
            self.actor.update_target_network()
            self.critic.update_target_network()
            # print("training epoch succesful")


    def update_replay_memory(self, state, reward, done, new_state):
        max_t = state.shape[0]
        print('update_replay', state.shape, reward.shape, self.action.reshape(max_t, self.action_size).shape, self.mask.reshape(max_t, self.action_size).shape)
        self.replay_memory.add(state, self.action.reshape(max_t, self.action_size), reward, new_state, done, self.mask.reshape(max_t, self.action_size))

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
        actionmask = np.zeros((traf.ntraf, self.spd_action_size, self.wp_action_size))
        for i in np.arange(traf.ntraf):
            value = self.ac_dict.get(traf.id[i])
            # If an aircraft does not need a new action [value[0] = 0] so the current action is maintained
            if value[0]==0:
                print('value', value[1][2])
                actionmask[i, value[1][2], :]=1
            else:
                actionmask[i,:,:]=1

        return actionmask.reshape(1, traf.ntraf, self.action_size)


    def act(self, state):
        # Select actions for each aircraft
        self.__update_acdict_entries()
        # print(self.ac_dict)
        # __update_aircraft_waypoints()
        print(self.ac_dict)
        # Infer action selection
        # self.action = np.random.random((traf.ntraf, 1))
        # actions are grouped per as follows

        #    Waypoints
        # s [[0, 1, 2, 3],
        # p [4, 5, 6, 7]
        # d [8, 9, 10, 11],
        #   [12, 13, 14, 15]]



        wp_action = np.random.random((self.wp_action_size,traf.ntraf))
        spd_action = np.random.random((self.spd_action_size,traf.ntraf))

        # Retrieve mask for action selection
        self.mask = self.get_mask()

        self.action = self.actor.predict([np.reshape(state, (1, traf.ntraf, self.state_size)), self.mask])
        print('action_shape', self.action.shape, self.action)
        print('mask', self.mask)
        print(self.action[0,0,:])
        action = np.zeros((traf.ntraf, 1))
        for i in range(traf.ntraf):
            action[i] = np.random.choice(a=np.arange(self.action_size), p=self.action[0,i,:])

        # print(mask)

        # Select action
        # print(wp_action_masked)
        # print(wp_action_masked)


        # bool_mask = tf.placeholder(dtype=tf.bool, shape=)
        # h2 = tf.boolean_mask(h2, bool_mask)
        # h2 = tf.nn.softmax(h2, name)
        print('action', action)
        wp_ind = np.argmax(wp_action, axis=1) # wp_action_masked
        wp_ind = action % self.wp_action_size
        spd_ind = np.argmax(spd_action, axis=1)
        spd_ind = np.floor(action / self.wp_action_size).astype(int)
        print('spd_ind', spd_ind)
        speed_actions = [self.speed_values[i[0]] for i in spd_ind]

        # Produce action command
        # Get the number of segments in main route index and construct
        maxsegments = 4 #TODO: Dummy to replace

        # Action is already selected. Just generate the commands ;)

        for i in np.arange(traf.ntraf):
            # Set wpindex
            ac_name = traf.id[i]
            wp_values = self.ac_dict.get(ac_name)
            # Only add wpts to route if required
            actwp = traf.ap.route[i].iactwp
            if traf.ap.route[i].wptype[actwp] == 3: # actwp == 3 corresponds to destination
                wp_name = wp_values[1]
                wp_name[2] = int(wp_ind[i])
                self.ac_dict[traf.id[i]] = [0, wp_name]
                stack.stack("{} ADDWPT RL{}".format(ac_name,  ''.join(map(str, wp_name))))
                wp_name[1] = wp_name[1] + 1

            # Create speed command
            stack.stack("{} SPD {}".format(ac_name, speed_actions[i]))

        # Wp command stack.stack(callsign ADDWPT wpname)
        # Spd command
    def update_cumreward(self, reward):
        self.cumreward += reward

    def reset(self):
        self.write_summaries(self.cumreward)
        self.cumreward=0
        self.ac_dict={}








