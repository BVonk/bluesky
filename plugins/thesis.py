# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:35:02 2018

@author: Bart
"""

""" Plugin to resolve conflicts """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools.geo import kwikdist, qdrpos, qdrdist, qdrdist_matrix
from bluesky.tools.misc import degto180
from bluesky import traf
from bluesky import navdb
from bluesky.tools.aero import nm, g0
from plugins.ml.actor import ActorNetwork
from plugins.ml.critic import CriticNetwork
from plugins.ml.ReplayMemory import ReplayMemory
from plugins.ml.normalizer import Normalizer
from plugins.ml.OU import OrnsteinUhlenbeckActionNoise



from plugins.vierd import ETA

import pickle
import random
import numpy as np
import time
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
    global env, agent, update_interval, routes, log_dir
    # Create logging folder
    log_dir = 'output/'+(time.strftime('%Y%m%d_%H%M%S')+'/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = Environment()
    agent = Agent()
    update_interval = 10
    routes = dict()


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
        new_state = env.init()


    else:
        # Construct new state
        state, reward, new_state, done = env.step()
        agent.update_cumreward(reward)
        agent.update_replay_memory(state, reward, done, new_state)
        if agent.replay_memory.num_experiences > agent.batch_size:
            agent.train()
        # agent.write_summaries(reward)

    agent.act_continuous(new_state)
    if env.done:
        # Reset environment states and agent states
        if env.episode % 50==0:
            agent.save_models(env.episode)
        env.reset()
        agent.reset()


    pass


def reset():
    pass


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


def norm(data, axis=0):
    ma = np.max(data, axis=axis)
    mi = np.min(data, axis=axis)
    return (data-mi)/(ma-mi)


class Environment:
    def __init__(self):
        self.n_aircraft = traf.ntraf
        self.state_size = 5
        self.ac_dict = {}
        self.prev_traf = 0
        self.observation = np.zeros((1,self.state_size))
        self.done = False
        self.state_normalizer = Normalizer(self.state_size)
        self.wp_db = self.load_waypoints()
        self.episode = 1

        for id in traf.id:
            self.ac_dict[id]= [0,0,0]

    def init(self):
        self.observation = self.generate_observation_continuous()
        return self.observation

    def step(self):
        prev_observation = self.observation
        self.observation = self.generate_observation_continuous()
        # Check termination conditions
        self.check_reached()
        self.generate_reward()
        return prev_observation, self.reward, self.observation, self.done

    def generate_reward(self):
        global_reward = 1
        local_reward = np.ones((traf.ntraf,1))
        self.reward = local_reward + global_reward
        if self.done:
            self.reward = 60 * np.ones((traf.ntraf,1))
        else:
            self.reward = -1 * np.ones((traf.ntraf,1))


    def generate_observation_continuous(self):
        destidx = navdb.getaptidx('EHAM')
        lat, lon = navdb.aptlat[destidx], navdb.aptlon[destidx]
        qdr, dist = qdrdist_matrix(traf.lat, traf.lon, lat*np.ones(traf.lat.shape), lon*np.ones(traf.lon.shape))
        obs = np.array([traf.lat, traf.lon, traf.hdg, qdr, dist]).transpose()
        return obs


    def generate_observation_discrete(self):
        # Produce a running average off all variables in the field with regard to the initial state convergence that is being tackled in the problem.
        obs = np.array([traf.lat, traf.lon, traf.tas, traf.cas, traf.alt]).transpose()
        #self.state_normalizer.observe(obs)
        #obs = self.state_normalizer.normalize(obs)

        #Get the normalized coordinates of the last and current waypoint.
        coords = np.empty((traf.ntraf, 4))
        for i in range(traf.ntraf):
            ac = agent.ac_dict.get(traf.id[i])

            if ac is None:
                coords[i, 0:2] = np.array([0,0])
                coords[i, 2:4] = np.array([0,0])
            else:
                lastwp = ac.lastwp
                curwp = ac.curwp
                if curwp == '':
                    curwp = 'dummy'
                if lastwp == '':
                    lastwp = 'dummy'

                coords[i, 0:2] = np.asarray(self.wp_db.get(lastwp))
                coords[i, 2:4] = np.asarray(self.wp_db.get(curwp))
        obs = np.hstack((obs, coords))
        return obs

    def check_reached(self):
        qdr, dist = qdrdist(traf.lat, traf.lon,
                                traf.actwp.lat, traf.actwp.lon)  # [deg][nm])

        # check which aircraft have reached their destination by checkwaypoint type = 3 (destination) and the relative
        # heading to this waypoint is exceeding 150
        dest = np.asarray([traf.ap.route[i].wptype[traf.ap.route[i].iactwp] for i in range(traf.ntraf)]) == 3
        away = np.abs(degto180(traf.trk % 360. - qdr % 360.)) > 150.
        dist = dist<2
        reached = np.where(away * dest * dist)[0]

        if reached==[0]:
            # TODO: Extend for multi-aircraft
            self.done = True
        else:
            self.done = False

    def generate_commands(self):
        pass

    def load_waypoints(self):
        wpts = np.loadtxt('plugins/ml/routes/testroute.txt')
        rows = wpts.shape[0]
        cols = wpts.shape[1]
        wpts = wpts.reshape((int(rows * cols / 2), 2))
        destidx = navdb.getaptidx('EHAM')
        lat, lon = navdb.aptlat[destidx], navdb.aptlon[destidx]
        wpts = np.vstack((wpts, np.array([lat,lon])))
        wpts = norm(wpts, axis=0)
        eham = wpts[-1,:]
        wpts = wpts[:-1,:].reshape((rows, cols))
        wp_db = dict()
        i = 0
        for j in range(wpts.shape[0]):
            for k in range(int(wpts.shape[1] / 2)):
                #               stack.stack('DEFWPT RL{}{}{}, {}, {}, FIX'.format(i, j, k, wpts[j,k*2], wpts[j,k*2+1]))
                wp_db['RL{}{}{}'.format(i, j, k)] = [wpts[j, k * 2], wpts[j, k * 2 + 1]]
        wp_db['EHAM'] = [eham[0], eham[1]]
        wp_db['dummy']=[0,0]
        return wp_db

    def reset(self):
        self.done = False
        self.episode += 1
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
        self.action_size = 1 #self.wp_action_size * self.spd_action_size
        self.max_agents = None
        self.action = []
        self.memory_size = 10000
        self.i = 0

        self.OU = OrnsteinUhlenbeckActionNoise(mu=np.array([0]))

        self.batch_size = 32
        self.tau = 0.9
        self.gamma = 0.99
        self.critic_learning_rate = 0.001
        self.actor_learning_rate = 0.001
        self.loss = 0
        self.cumreward=0
        self.train_indicator = True



        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
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
        summary_dir = log_dir
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
        model.save_weights("{}.h5".format(name))\

    def save_models(self, episode):
        self.save_model(self.actor.model, log_dir+'actor_model{0:05d}'.format(episode))
        self.save_model(self.actor.target_model, log_dir+'target_actor_model{0:05d}'.format(episode))
        self.save_model(self.critic.model, log_dir+'critic_model{0:05d}'.format(episode))
        self.save_model(self.critic.target_model, log_dir+'target_critic_model{0:05d}'.format(episode))

    def train(self):
        batch = self.replay_memory.getBatch(self.batch_size)
        for seq in batch:
            pass

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
        # mask = np.asarray([self.pad_zeros(seq[5], max_t) for seq in batch])
        y_t = rewards.copy()

        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

        #Compute the target values
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.gamma * target_q_values[k]

        if self.train_indicator:
            actions_for_grad = self.actor.model.predict(states)
            grads = self.critic.gradients(states, actions_for_grad)
            self.critic.train(states, actions, y_t)
            self.actor.train(states, grads)
            self.actor.update_target_network()
            self.critic.update_target_network()



    def update_replay_memory(self, state, reward, done, new_state):
        max_t = state.shape[0]
        # print('update_replay', state.shape, reward.shape, self.action.reshape(max_t, self.action_size).shape, self.mask.reshape(max_t, self.action_size).shape)
        self.replay_memory.add(state, self.action.reshape(max_t, self.action_size), reward, new_state, done)

    def __update_aircraft_waypoints(self):
        """
        Checks if aircraft have reached their waypoints. Aircraft having reached their waypoints will have their
        destination as waypoint, which corresponds to wptype == 3
        :return:
        """
        for i in np.arange(traf.ntraf):

            actwp = traf.ap.route[i].iactwp
            ac = self.ac_dict.get(traf.id[i])
            if traf.ap.route[i].wptype[actwp] == 3 and actwp<ac.route_length:
                ac.set_wp_flag(1)
                self.ac_dict[traf.id[i]] = ac
            elif traf.ap.route[i].wptype[actwp] == 3 and actwp >= ac.route_length:
                ac.set_dest_flag(1)
        # return list of waypoints


    def __update_acdict_entries(self):
        """
        Check which aircraft have to be added to dictionary or deleted when aircraft have landed.
        Sets the aircraft status to [x, [wpindex]. New aircraft are initialized without wpname
        x = 1 when aircraft must select new waypoint, otherwise 0
        """
        for id in traf.id:
            if id not in self.ac_dict:
                self.ac_dict[id]=Aircraft(id)
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
            aircraft = self.ac_dict.get(traf.id[i])
            # If an aircraft does not need a new action only speed changes while flying towards the current waypoint are
            # allowed. Otherwise all actions are allowed.
            if aircraft.dest_flag == 1:
                actionmask[i,:,0] = 1
            elif aircraft.wpflag == 0:
                actionmask[i, :, aircraft.k] = 1
            else:
                actionmask[i,:,:]=1

        return actionmask.reshape(1, traf.ntraf, self.action_size)

    def act_continuous(self, state):
        # self.__update_acdict_entries()
        self.action = self.actor.predict([np.reshape(state, (1, traf.ntraf, self.state_size))]) + self.OU()
        print('actionstate', self.action, state)
        #Apply gaussian here to sample from to get action values
        mu = 0
        sigma = 0.5
        y_offset = 0.107982 + 6.69737e-08
        action = bell_curve(self.action[0], mu=mu, sigma=sigma, y_offset=y_offset, scaled=True)

        # Due to exploratio noise action could drop below 0
        action[np.where(action<0)[0]]=0

        dist_limit = 5 #nm
        dist = state[:, 4].transpose()
        dist = dist.reshape(action.shape)
        mul_factor = 90*np.ones(dist.shape)

        print(mul_factor.shape, dist.shape)
        wheredist = np.where(dist<dist_limit)[0]
        mul_factor[wheredist] = dist[wheredist] / dist_limit * 90
        minus = np.where(self.action<0)[0]
        plus = np.where(self.action>=0)[0]
        dheading = np.zeros(action.shape)
        print(dheading.shape, minus.shape, mul_factor.shape, action.shape)
        dheading[minus] = (action[minus] - 1) * mul_factor[minus]
        dheading[plus] = np.abs(action[plus]-1) * mul_factor[plus]
        print('selfaction', self.action)
        print('dheading', dheading)
        action[minus] = -1*action[minus]
        qdr = state[:,3].transpose()
        print('qdr', qdr)
        heading =  qdr + dheading
        print('heading action', heading)
        for i in np.arange(traf.ntraf):
            stack.stack('HDG {} {}'.format(traf.id[i], heading[0][i]))




    def act_discrete(self, state):
        # Infer action selection
        # actions are grouped per as follows
        #    Waypoints
        # s [[0, 1, 2, 3],
        # p [4, 5, 6, 7]
        # d [8, 9, 10, 11],
        #   [12, 13, 14, 15]]
        self.__update_acdict_entries()
        self.__update_aircraft_waypoints()

        # Retrieve mask for action selection
        self.mask = self.get_mask() # self.mask = np.array([[[1,1,1,0,0,0,0,0,0,0,0,0]]])
        self.action = self.actor.predict([np.reshape(state, (1, traf.ntraf, self.state_size)), self.mask])
        self.action = self.actor.predict([np.reshape(state, (1, traf.ntraf, self.state_size))])
        # print('mask', self.mask)
        # print(self.action[0,0,:])
        action = np.zeros((traf.ntraf, 1))
        for i in range(traf.ntraf):
            action[i] = np.random.choice(a=np.arange(self.action_size), p=self.action[0,i,:])

        wp_ind = action % self.wp_action_size
        spd_ind = np.floor(action / self.wp_action_size).astype(int)
        print('spd_ind', spd_ind)
        speed_actions = [self.speed_values[i[0]] for i in spd_ind]

        # Produce action command
        # Get the number of segments in main route index and construct
        maxsegments = 4 #TODO: Dummy to replace

        # Action is already selected. Just generate the commands ;)

        for i in np.arange(traf.ntraf):
            # Set wpindex
            ac = self.ac_dict.get(traf.id[i])
            # Only add wpts to route if required
            actwp = traf.ap.route[i].iactwp

            if ac.wpflag==1: # actwp == 3 corresponds to destination
                wp_name = [ac.i, ac.j, ac.k]
                wp_name[2] = int(wp_ind[i])
                wpt = 'RL{}'.format(''.join(map(str, wp_name)))
                stack.stack("{} ADDWPT {}".format(ac.id,  wpt))
                ac.set_k(int(wp_ind[i]))
                ac.increment_j()
                ac.set_wp_flag(0)
                ac.set_wp(wpt)
                self.ac_dict[traf.id[i]] = ac

            elif actwp>=ac.route_length and ac.curwp!='EHAM':
                dest='EHAM'
                ac.set_wp(dest)

            # Create speed command
            stack.stack("{} SPD {}".format(ac.id, speed_actions[i]))


    def update_cumreward(self, reward):
        self.cumreward += reward

    def reset(self):
        self.write_summaries(self.cumreward)
        self.cumreward=0
        self.ac_dict={}

class Aircraft():
    def __init__(self, id):
        self.lastwp = ''
        self.curwp  = ''
        self.i = 0
        self.j = 0
        self.k = 0
        self.wpflag = 1
        self.dest_flag = 0
        self.id = id
        self.route_length = 3

    def set_wp(self, wp):
        self.lastwp = self.curwp
        self.curwp = wp

    def set_wp_flag(self, x):
        self.wpflag = x

    def set_k(self, x):
        self.k = x

    def set_dest_flag(self, x):
        self.dest_flag = x

    def increment_j(self):
        self.j += 1

def bell_curve(x, mu=0., sigma=1., y_offset=0, scaled=1):
    # Computes Gaussian and scales the peak to 1.
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-np.power(x - mu, 2) / (2 * sigma ** 2)) + y_offset
    if scaled:
        y_scale = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-np.power(0, 2) / (2 * sigma ** 2)) + y_offset
        y = y/y_scale
    return y