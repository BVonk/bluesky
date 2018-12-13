# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:35:02 2018

@author: Bart
"""

""" Plugin to resolve conflicts """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim, traf, navdb  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools.geo import kwikdist, qdrpos, qdrdist, qdrdist_matrix
from bluesky.tools.misc import degto180
from bluesky.tools.aero import nm, g0
import bluesky.settings as CONF

from plugins.ml.actor import ActorNetwork, ActorNetwork_shared_obs
from plugins.ml.critic import CriticNetwork, CriticNetwork_shared_obs
from plugins.ml.ReplayMemory import ReplayMemory
from plugins.ml.normalizer import Normalizer
from plugins.ml.OU import OrnsteinUhlenbeckActionNoise
from plugins.help_functions import detect_los, normalize, print_intermediate_layer_output
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
from shutil import copyfile
from copy import deepcopy
# from keras.backend import manual_variable_initialization
# manual_variable_initialization(True)

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    global env, agent, update_interval, routes, log_dir

    if os.path.isdir(CONF.log_dir):
        log_dir = CONF.log_dir
    else:
        log_dir = 'output/'+(time.strftime('%Y%m%d_%H%M%S')+'/')

    # Create logging folder for Tensorflow summaries and saving network weights
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        os.makedirs(log_dir+'training/')
        os.makedirs(log_dir+'test/')

    # The settings file is copied as a record of the settings used for simulation
    copyfile('settings.cfg', log_dir+'settings.cfg')
    copyfile('scenario/Bart/' + CONF.scenario, log_dir+CONF.scenario)

    agent = Agent(CONF.state_size, CONF.action_size, CONF.tau, CONF.gamma, CONF.critic_lr,
                  CONF.actor_lr, CONF.memory_size, CONF.max_agents, CONF.batch_size, CONF.train_bool, log_dir,
                  CONF.load_ep, CONF.sigma_OU, CONF.theta_OU, CONF.dt_OU, CONF.model_type)

    # env = Environment(CONF)
    env = Environment(CONF.state_size, CONF.scenario, CONF.model_type)
    # routes = dict()

    # TODO: Find system performance indicator
    # TODO: Implement minimum wake separation, minimum wake separation is defined from trailing experiments
    # TODO: Set destination for every aircraft (FAF) for designated runway
    # TODO: Use BlueSKy logging / plotting
    # TODO: Fix circling target problem, this problem occurs at higher speeds, currently solved by reducing speed to 200 m/s near the runway. Second option is adding speed changes (instantaneous?)
    # TODO: Check all intermediate layers of the critic jwz.
    # TODO: Replace LSTM with RNN for easier shit
    # TODO: Initialize the initial RNN values with 0 for the 'hidden state'
    # TODO: Create multiple scenarios for curriculum learning
    # TODO: Check the initial
    # TODO: Find some global optimization parameter
    # TODO: Annealing of the environment noise
    # TODO: Use warm-up episode

    config = {
        # The name of your plugin
        'plugin_name':     'THESIS',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': CONF.update_interval,

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
    # print('settings {}'.format(CONF.conf))
    # First run the environment must be initialized to get the first state ready before model inference can take place.
    if sim.simt < CONF.update_interval*1.5:
        new_state = env.init()


    elif traf.ntraf!=0:
        # Construct new state
        state, reward, new_state, done = env.step()
        agent.update_cum_reward(reward)
        agent.update_replay_memory(state, reward, done, new_state)

        # if agent.replay_memory.num_experiences > agent.batch_size:
        # print('memory_size', agent.replay_memory.num_experiences)
        if agent.replay_memory.num_experiences > 1000 or agent.replay_memory.num_experiences == agent.memory_size:
            # agent.train_no_batch()
            agent.train()
        # agent.write_summaries(reward)
        # Now get observation without the deleted aircraft for acting. Otherwise an error occurs because the deleted
        # aircraft no longer exists in the stack.
        new_state = env.get_observation()

    collision = env.check_collision()

    if traf.ntraf!=0 and not collision:
        action = agent.act(new_state)
        env.action_command(action)


    # Check if all aircraft in simulation landed and there are no more scenario commands left
    if (env.get_done() and len(stack.get_scendata()[0])==0) or collision or env.step_num>170:
        # Reset environment states and agent states
        if env.episode % 50==0:
            agent.save_models(env.episode)
        env.reset()
        agent.reset()


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
    def __init__(self, state_size, scenario, model_type, shared_state_size=0):
        self.n_aircraft = traf.ntraf
        if type(state_size)==list:
            self.state_size = state_size[0]
            self.shared_state_size = state_size[1]
        else:
            self.state_size = state_size
            self.shared_state_size = 0

        self.ac_dict = {}
        self.prev_traf = 0
        self.step_num = 0
        self.observation = np.zeros((1,self.state_size))
        self.done = np.array([False])
        self.state_normalizer = Normalizer(self.state_size)
        # self.wp_db = self.load_waypoints()
        self.episode = int(CONF.load_ep)
        print('episode', self.episode)
        self.scn = scenario
        self.idx = []
        self.los_pairs = []
        self.dist_scale = 110
        observation_dict = {'bicnet_normal': self.generate_observation_continuous,
                            'bicnet_shared': self.generate_observation_continuous_shared}
        action_dict = {'bicnet_normal': self.action_command,
                       'bicnet_shared': self.action_command}
        self.act = action_dict.get(model_type)
        self.generate_observation = observation_dict.get(model_type)

        for id in traf.id:
            self.ac_dict[id]= [0,0,0]


    def init(self):
        self.observation = self.generate_observation() #[self.generate_observation_continuous(), self.generate_shared_observation()]
        return self.observation


    def step(self):
        self.step_num += 1
        self.prev_observation = self.observation
        self.observation = self.generate_observation()
        # Check termination conditions
        # self.los_pairs = detect_los(traf, traf, traf.asas.R, traf.asas.dh)
        # Add in 9999999 for vertical protection zone to ignore vertical separation
        self.los_pairs = detect_los(traf, traf, traf.asas.R, 9999999)
        self.check_reached()
        self.generate_reward()
        # print('rew', self.reward)
        done = True if self.done.all() == True else False

        done_idx = np.where(self.done == True)[0]
        for idx in done_idx:
            stack.stack("DEL {}".format(traf.id[idx]))

        # There is a mismatch between the aircraft size in the observation returned for the replay memory and the
        # observation required to select actions when an aircraft is deleted. Therefore two separate observations must
        # be used.
        replay_observation = self.observation

        if type(self.observation)==list:
            self.observation[0] = np.delete(self.observation[0], done_idx, 0)
            if self.observation[0].shape[0]==0:
                self.observation[1] = np.delete(self.observation[1], np.arange(self.observation[1].shape[0]), 0)
            else:
                mask = np.ones(self.observation[1].shape, dtype=np.bool)
                for idx in done_idx:
                    mask[idx, :, :] = 0
                    if idx == 0 and mask.shape[1]==0:
                        mask[:,:,:] = 0
                    elif idx == 0:
                        mask[idx:, idx, :] = 0
                    elif idx == mask.shape[1]:
                        mask[:idx, idx - 1, :] = 0
                    else:
                        mask[:idx, idx - 1, :] = 0
                        mask[idx:, idx, :] = 0

                self.observation[1] = self.observation[1][mask].reshape((traf.ntraf - len(done_idx), traf.ntraf - len(done_idx) - 1, self.shared_state_size))
        else:
            self.observation = np.delete(self.observation, done_idx, 0)
        self.idx = np.delete(traf.id, done_idx)

        return self.prev_observation, self.reward, replay_observation, done

    def generate_reward(self):
        """ Generate reward scalar for each aircraft"""
        global_reward = -0.05
        reached_reward = self.done * 10
        los_reward = np.zeros(reached_reward.shape)
        forward_reward = np.zeros(reached_reward.shape)
        forward_reward[np.where(self.observation[:,4]<self.prev_observation[:,4])[0]] = 0.02
        forward_reward = 0.2 * (np.abs(self.observation[:,3]*180 - degto180(traf.hdg)) % 180) / 90
        # print('obs', self.observation[:,3], traf.hdg, self.observation[:,3]*180)


        if len(self.los_pairs) > 0:
            ac_list = [ac[0] for ac in self.los_pairs]
            traf_list = list(traf.id)
            idx = [traf_list.index(x) for x in ac_list]
            for i in idx:
                los_reward[i] = los_reward[i] - 25
        self.reward = np.asarray(reached_reward + global_reward + los_reward + forward_reward).reshape((reached_reward.shape[0],1))
        # print('reward', self.reward, 'r_rew', reached_reward, 'glob', global_reward, 'los', los_reward , 'forw',  forward_reward)

    def generate_observation_continuous(self):
        """ Generate observation of size N_aircraft x state_size"""
        destidx = navdb.getaptidx('EHAM')

        minlat, maxlat = 50.75428888888889, 55.
        minlon, maxlon = 2., 7.216944444444445

        lat, lon = navdb.aptlat[destidx], navdb.aptlon[destidx]
        qdr, dist = qdrdist_matrix(traf.lat, traf.lon, lat*np.ones(traf.lat.shape), lon*np.ones(traf.lon.shape))
        qdr, dist = np.asarray(qdr)[0], np.asarray(dist)[0]
        obs = np.array([traf.lat, traf.lon, traf.hdg, qdr, dist, traf.cas]).transpose()
        # Normalize input data to the range [-1, ,1]
        obs = np.array([normalize(traf.lat, minlat, maxlat),
                        normalize(traf.lon, minlon, maxlon),
                        normalize(traf.hdg, 0, 360),
                        normalize(qdr+180, 0, 360),
                        normalize(dist, 0, self.dist_scale),
                        normalize(traf.cas, 80, 200)]).transpose()
        return obs

    def generate_shared_observation(self):
        """
        Generate the shared observation sequences for the aircraft
        States are distance, heading, bearing, latitude longitude, (speed)

        The observation are Ntraf sequences of size (Ntraf - 1 , x)
        """
        minlat, maxlat = 50.75428888888889, 55.
        minlon, maxlon = 2., 7.216944444444445
        # Generate distance matrix
        dist, qdr = qdrdist_matrix(np.mat(traf.lat), np.mat(traf.lon), np.mat(traf.lat), np.mat(traf.lon))
        dist, qdr = np.array(dist), np.array(qdr)
        shared_obs = np.zeros((traf.ntraf, traf.ntraf - 1, self.shared_state_size))
        for i in range(traf.ntraf):
            # shared_obs = np.zeros((traf.ntraf - 1, self.shared_state_size))
            shared_obs[i, :, 0] = np.delete(normalize(dist[i,:], 0, 200), i, 0)
            shared_obs[i, :, 1] = np.delete(normalize(qdr[i,:]+180, 0, 360), i, 0)  # divide by 180
            shared_obs[i, :, 2] = np.delete(normalize(traf.lat, minlat, maxlat), i)
            shared_obs[i, :, 3] = np.delete(normalize(traf.lon, minlon, maxlon), i)
            shared_obs[i, :, 4] = np.delete(traf.hdg/360., i)     # divide by 180
            shared_obs[i, :, 5] = np.delete(normalize(traf.cas, 75, 200), i)

        if traf.ntraf==1:
            shared_obs = np.zeros((traf.ntraf+1, traf.ntraf, self.shared_state_size))
        # print('shard_obs', shared_obs.shape)
        return shared_obs

    def generate_observation_continuous_shared(self):
        return [self.generate_observation_continuous(), self.generate_shared_observation()]


    def generate_observation_discrete(self):
        # Produce a running average off all variables in the field with regard to the initial state convergence that is being tackled in the problem.
        obs = np.array([traf.lat, traf.lon, traf.tas, traf.cas, traf.alt]).transpose()
        # self.state_normalizer.observe(obs)
        # obs = self.state_normalizer.normalize(obs)

        # Get the normalized coordinates of the last and current waypoint.
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

    def action_command(self, action):
        for i in range(len(self.idx)):
            stack.stack('HDG {} {}'.format(self.idx[i], action[i]))

        if len(self.idx)!=0:
            obs = self.observation[0] if type(self.observation) == list else self.observation
            if (len(obs.shape)==2):
                obs = np.expand_dims(obs, axis=0)
            # dist = obs[0][:, :, 4]*self.dist_scale
            dist = obs[:, :, 4] * self.dist_scale
            dist_lim = 10
            dist_idx = np.where(np.abs(dist-dist_lim/2)<dist_lim/2)[1]


            for idx in dist_idx:
                stack.stack('SPD {} 200'.format(self.idx[idx]))


    def check_collision(self):
        if len(self.los_pairs)!=0:
            return True
        else:
            return False

    def check_reached(self):
        qdr, dist = qdrdist(traf.lat, traf.lon,
                                traf.actwp.lat, traf.actwp.lon)  # [deg][nm])

        # check which aircraft have reached their destination by checkwaypoint type = 3 (destination) and the relative
        # heading to this waypoint is exceeding 150
        dest = np.asarray([traf.ap.route[i].wptype[traf.ap.route[i].iactwp] for i in range(traf.ntraf)]) == 3
        away = np.abs(degto180(traf.trk % 360. - qdr % 360.)) > 100.
        d = dist<2
        reached = np.where(away * dest * d)[0]
        # print('track', traf.trk, qdr, dist)

        # What is the desired behaviour. When an aircraft has reached, it will be flagged as reached.
        # Therefore the aircraft should get a corresponding reward as well for training purposes in the replay memory
        # Next the corresponding aircraft should be deleted.
        # If no more aircraft are present, next episode should start.
        # print('reached', reached)
        done = np.zeros((traf.ntraf), dtype=bool)
        done[reached] = True
        self.done = done

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
        self.done = np.array([False])
        self.los_pairs = []
        self.done = []
        self.episode += 1
        self.prev_traf = 0
        self.observation = np.zeros((1,self.state_size))
        self.done = np.array([False])
        self.idx = []
        self.los_pairs = []
        self.step_num = 0

        stack.stack('open ./scenario/bart/{}'.format(self.scn))
        # load_routes()

    def get_done(self):
        if self.done.all() == True:
            return True
        else:
            return False

    def get_observation(self):
        return self.observation


class Agent:
    def __init__(self, state_size, action_size, tau=0.9, gamma=0.99, critic_lr=0.001, actor_lr=0.001, memory_size=10000, max_agents=10, batch_size = 32, training=True, load_dir='', load_ep=0, sigma=0.15, theta=.5, dt=0.1, model_type='bicnet_normal'):
        # Config parameters

        self.train_indicator = training
        self.action_size = action_size
        if type(state_size)==list:
            self.state_size = state_size[0]
            # print('selfstate', self.state_size)
            self.shared_obs_size = state_size[1]
        else:
            self.state_size = state_size
            self.shared_obs_size = 0
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.critic_learning_rate = critic_lr
        self.actor_learning_rate = actor_lr
        self.loss = 0
        self.cum_reward=0
        self.memory_size = memory_size
        self.max_agents = max_agents
        self.OU = OrnsteinUhlenbeckActionNoise(np.zeros(self.max_agents), sigma=sigma, theta=theta, dt=dt)

        self.load_dir = load_dir
        self.load_ep  = str(load_ep).zfill(5)

        self.ac_dict = {}
        # self.speed_values = [175, 200, 225, 250]
        # self.wp_action_size = 3
        # self.spd_action_size = len(self.speed_values)

        # self.wp_action_size * self.spd_action_size

        self.action = []
        self.summary_counter = 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)


        actor_dict = {'bicnet_normal': ActorNetwork,
                      'bicnet_shared': ActorNetwork_shared_obs}
        critic_dict = {'bicnet_normal': CriticNetwork,
                       'bicnet_shared': CriticNetwork_shared_obs}
        batch_dict = {'bicnet_normal': self.preprocess_batch,
                      'bicnet_shared': self.preprocess_batch_shared}
        act_dict = {'bicnet_normal': self.act_continuous,
                    'bicnet_shared': self.act_continuous}


        create_actor = actor_dict.get(model_type)
        create_critic = critic_dict.get(model_type)
        self.get_batch = batch_dict.get(model_type)
        self.act = act_dict.get(model_type)

        self.actor = create_actor(self.sess, state_size, self.action_size,
                                  self.max_agents, self.batch_size, self.tau, self.actor_learning_rate)
        self.critic = create_critic(self.sess, state_size, self.action_size,
                                    self.max_agents, self.batch_size, self.tau, self.critic_learning_rate)
        self.replay_memory = ReplayMemory(self.memory_size)

        # Set up summary Ops
        self.summary_ops, self.summary_vars, self.summary_ops_test, self.summary_vars_test = self.build_summaries()

        self.sess.run(tf.global_variables_initializer())
        if self.train_indicator:
            summary_dir = log_dir + 'training/'
        else:
            summary_dir = log_dir + 'test/'
        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        #Now load the weight
        try:
            self.actor.load_weights(self.load_dir + "actor_model" + "{0:05d}".format(int(self.load_ep)) + ".h5")
            self.critic.load_weights(self.load_dir + "critic_model" + "{0:05d}".format(int(self.load_ep)) + ".h5")
            self.actor.load_target_weights(self.load_dir + "target_actor_model" + "{0:05d}".format(int(self.load_ep)) + ".h5")
            self.critic.load_target_weights(self.load_dir + "target_critic_model" + "{0:05d}".format(int(self.load_ep)) + ".h5")
            self.summary_counter = int(self.load_ep)
            print("Weights load successfully")
        except:
            print("Cannot find the weights")

    def pad_zeros(self, array, max_timesteps):
        """Pad 0's for local observation sequences"""
        # if array.shape[0] == max_timesteps:
        #     return array
        # else:
        # result = np.zeros((max_timesteps, array.shape[-1], self.action_size))
        result = np.zeros((max_timesteps, array.shape[-1]))
        if len(array.shape)==3:
            array = array.reshape((array.shape[1], array.shape[2]))
        result[0:array.shape[0], :] = array
        # print('resultshape', result.shape)
        return result

    def pad_nines(self, array, max_timesteps):
        """Pad -999 for shared observations sequences"""
        # print("array shape", array.shape)
        zeros = -999.0 * np.ones((max_timesteps, max_timesteps - 1, array.shape[-1]))
        if len(array.shape)==4:
            array=array.reshape((array.shape[1:]))
        zeros[0:array.shape[0], 0:array.shape[1], :] = array
        # print('shapeshape', array.shape)
        return zeros.reshape(max_timesteps*(max_timesteps-1), array.shape[2])

    def build_summaries(self):
        episode_reward = tf.Variable(0., name="Episode_reward")
        # loss = tf.Variable(0., name="critic_loss")
        # tf.summary.scalar("Critic_loss", loss)
        tf.summary.scalar("Reward", episode_reward)
        summary_vars = [episode_reward]# , episode_ave_max_q]
        summary_ops = tf.summary.merge_all()

        test_reward = tf.Variable(0.)
        summary_ops_test = tf.summary.scalar("Test_reward", test_reward)
        summary_vars_test = [test_reward]
        # summary_ops_test = tf.summary.tensor_summary('test', test_reward)
        return summary_ops, summary_vars, summary_ops_test, summary_vars_test

    def write_train_summaries(self, reward):
        summary_str = self.sess.run(self.summary_ops, feed_dict={
            self.summary_vars[0]: np.sum(reward),
                    })
        self.writer.add_summary(summary_str, self.summary_counter)
        self.writer.flush()
        self.summary_counter += 1

    def write_test_summaries(self, reward):
        summary_str = self.sess.run(self.summary_ops_test, feed_dict={
            self.summary_vars_test[0]: np.sum(reward),
        })
        self.writer.add_summary(summary_str, self.summary_counter)
        self.writer.flush()
        self.summary_counter += 1

    def save_model(self, model, name):
        #yaml saves the model architecture
        model_yaml = model.to_yaml()
        with open("{}.yaml".format(name), 'w') as yaml_file:
            yaml_file.write(model_yaml)
        #Serialize weights to HDF5 format
        model.save_weights("{}.h5".format(name))

    def save_models(self, episode):
        self.save_model(self.actor.model, log_dir+'actor_model{0:05d}'.format(episode))
        self.save_model(self.actor.target_model, log_dir+'target_actor_model{0:05d}'.format(episode))
        self.save_model(self.critic.model, log_dir+'critic_model{0:05d}'.format(episode))
        self.save_model(self.critic.target_model, log_dir+'target_critic_model{0:05d}'.format(episode))

    def preprocess_batch(self):
        """
        Unpack the batch into format that is workable for tensorflow models for a normal bicnet model
        :return:
        """
        batch = self.replay_memory.getBatch(self.batch_size)
        max_t = []
        for i in range(self.batch_size):
            # print('batch', batch[i])
            max_t.append(batch[i][2].size) # Check the size of the reward

        max_t = max(max_t)

        # print('max_t', max_t)
        states = np.asarray([self.pad_zeros(seq[0], max_t) for seq in batch])
        actions = np.asarray([self.pad_zeros(seq[1], max_t) for seq in batch])
        rewards = np.asarray([self.pad_zeros(seq[2], max_t) for seq in batch])
        new_states = np.asarray([self.pad_zeros(seq[3], max_t) for seq in batch])
        dones = np.asarray([seq[4] for seq in batch])

        return states, actions, rewards, new_states, dones

    def preprocess_batch_shared(self):
        """
        Unpack the batch into format that is workable for tensorflow models for a nshared observation bicnet model
        :return:
        """
        batch = self.replay_memory.getBatch(self.batch_size)
        max_t = []
        for i in range(self.batch_size):
            max_t.append(batch[i][2].size)

        max_t = max(max_t)
        states = [np.asarray([self.pad_zeros(seq[0][0], max_t) for seq in batch])]
        states.append(np.asarray([self.pad_nines(seq[0][1], max_t) for seq in batch]))
        actions = np.asarray([self.pad_zeros(seq[1], max_t) for seq in batch])
        rewards = np.asarray([self.pad_zeros(seq[2], max_t) for seq in batch])
        new_states = [np.asarray([self.pad_zeros(seq[3][0], max_t) for seq in batch])]
        new_states.append(np.asarray([self.pad_nines(seq[3][1], max_t) for seq in batch]))
        dones = np.asarray([seq[4] for seq in batch])

        return states, actions, rewards, new_states, dones

    def train_no_batch(self):
        """
        Experimental function to check batches without any significant reason. Averaging the gradients will be done
        after the batch. This should not hurt performance too much, because no GPU is used anyway.
        :return:
        """
        if self.train_indicator:
            batch = self.replay_memory.getBatch(self.batch_size)
            actor_gradients = []
            critic_gradients = []

            # First obtain all the gradients from the critic before training the critic.
            states = [np.expand_dims(seq[0], axis=0) for seq in batch]
            actions = [np.expand_dims(seq[1], axis=0) for seq in batch]
            rewards = [seq[2] for seq in batch]
            new_states = [np.expand_dims(seq[3], axis=0) for seq in batch]
            dones = [seq[4] for seq in batch]

            # for sample in batch:
            #     states, actions, rewards, new_states, dones = sample

            target_q_values = self.critic.predict_target_separate(new_states, self.actor.predict_target_separate(new_states))
            y_t = deepcopy(target_q_values)
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k].reshape((rewards[k].shape[0], 1))
                else:
                    y_t[k] = rewards[k].reshape((rewards[k].shape[0], 1)) + self.gamma * target_q_values[k]



            actions_for_grads = self.actor.predict_separate(states)
            grads = self.critic.gradients_separate(states, actions_for_grads)


            self.actor.train_separate(states, grads)
            self.critic.train_separate(states, actions, y_t)

            self.actor.update_target_network()
            self.critic.update_target_network()

    def train(self):
        if self.train_indicator:
            # TODO: Check that the gradient calculator ignores the zero input sequences.
            batch = self.replay_memory.getBatch(self.batch_size)
            # print("batch", batch)
            # np.save("batch_array", batch)
            # In order to create sequences with equal length for batch processing sequences are padded with zeros to the
            # maximum sequence length in the batch. Keras can handle the zero padded sequences by ignoring the zero
            # calculations
            states, actions, rewards, new_states, dones = self.get_batch()
            # target_q_values = self.critic.target_model.predict(new_states, self.actor.target_model.predict(new_states))
            # target_q_values = self.critic.target_model.predict([new_states[0], new_states[1], self.actor.target_model.predict(new_states)])
            target_q_values = self.critic.predict_target(new_states, self.actor.predict_target(new_states))
            y_t = target_q_values.copy()
            #Compute the target values
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k].reshape((rewards.shape[1], 1))
                else:
                    y_t[k] = rewards[k].reshape((rewards.shape[1], 1)) + self.gamma * target_q_values[k]

            # prediction = self.critic.model.predict([states, actions])

            # print('pred', prediction.shape, prediction)
            # print('y_t', y_t.shape, y_t)



            # loss = self.critic.train(states, actions, y_t)
            actions_for_grad = self.actor.model.predict(states)
            grads = self.critic.gradients(states, actions_for_grad)
            # Mask gradients?
            # print('grads', len(grads[0]), grads[0])
            loss = self.critic.train(states, actions, y_t)
            self.actor.train(states, grads)
            self.actor.update_target_network()
            self.critic.update_target_network()


    def update_replay_memory(self, state, reward, done, new_state):
        # print('state_shape', state[1].shape)
        self.replay_memory.add(state, self.action.reshape(self.action.shape[1], self.action_size), reward, new_state, done)


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
        state_copy = state.copy()
        # No action should be taken if there are no aircraft, otherwise the program crashes.
        # n_aircraft = int(np.prod(obs.shape) / self.state_size)
        if type(state)==list:
            n_aircraft = int(np.prod(state[0].shape) / self.state_size)


            if len(state_copy[0].shape)==2:
                state_copy[0] = np.expand_dims(state_copy[0], axis=0)
            if len(state_copy[1].shape)==2:
                state_copy[1] = np.expand_dims(state_copy[1], axis=0)
            obs = state_copy[0]
            state_copy[1] = state[1].reshape(1, n_aircraft * (n_aircraft - 1), state[1].shape[-1])
        else:
            n_aircraft = int(np.prod(state.shape)/self.state_size)
            if len(state_copy.shape)==2:
                state_copy = np.expand_dims(state_copy, axis=0)

            obs = state_copy
        if n_aircraft==0:
            return

        # state[0] = state[0].reshape(1, state[0].shape[0], state[0].shape[1])
        # state[0] = self.pad_zeros(state[0], self.max_agents).reshape((1, self.max_agents, self.state_size))
        # state[1] = self.pad_nines(state[1], self.max_agents).reshape((1, self.max_agents, self.max_agents-1, self.shared_obs_size))
        # state_copy = state.copy()
        # state_copy[0] = state[0].reshape(1, state[0].shape[0], state[0].shape[1])

        # if n_aircraft==1:
        #     state_copy[1] = state[1].reshape(1, n_aircraft, state[1].shape[-1])
        # else:
        #     state_copy[1] = state[1].reshape(1, n_aircraft*(n_aircraft-1), state[1].shape[-1])
        # state_copy[1] = state[1].reshape(1, n_aircraft * (n_aircraft - 1), state[1].shape[-1])


        self.action = self.actor.predict(state_copy)
        # state[0] = state[0].reshape(1, state[0].shape[0], state[0].shape[1])

        # data = state
        # model = self.target.model
        # print_intermediate_layer_output(model, data, 'merged_mask')
        # print_intermediate_layer_output(model, data, 'pre_brnn')
        # print_intermediate_layer_output(model, data, 'brnn')
        # print_intermediate_layer_output(model, data, 'post_brnn')


        # data = [state[0], state[1], self.action]
        # model = self.critic.model
        # print_intermediate_layer_output(model, data, 'input_actions')
        # print_intermediate_layer_output(model, data, 'max_pool')
        # print_intermediate_layer_output(model, data, 'concatenate_inputs')
        # print_intermediate_layer_output(model, data, 'input_mask')
        # print_intermediate_layer_output(model, data, 'pre_brnn')
        # print_intermediate_layer_output(model, data, 'brnn')
        # print_intermediate_layer_output(model, data, 'post_brnn')

        # Exploration noise is added only when no test episode is running
        # print('OU', self.OU())
        noise = self.OU()[0:n_aircraft].reshape(self.action.shape)
        # print(not self.summary_counter % CONF.test_freq, not self.train_indicator)
        print('action', self.action, '' , noise, 'noise')
        if not self.summary_counter % CONF.test_freq == 0 or not self.train_indicator:
            # Add exploration noise and clip to range [-1, 1] for action space

            self.action = self.action + noise
            # print('noise', noise, self.action)

            self.action = np.maximum(-1*np.ones(self.action.shape), self.action)
            self.action = np.minimum(np.ones(self.action.shape), self.action)
            # print('action', self.action)

        # Keras masked inputs do not output 0, but rather output the previous output without modifying it. This gives
        # issues when using the action outputs as inputs for the critic and trying to mask it. The nonzero masked output
        # from the actor therefore is not probably masked in the critic due to the nonzero values. Therefore the masked
        # actor output is manually reset to 0 here. It is better to use tensorflow for masked output in recurrent
        # networks, because Tensorflow does output 0 for masked inputs.
        # self.action[:, n_aircraft:, :] = 0

        # data = state
        # model = self.target.model
        # print_intermediate_layer_output(model, data, 'merged_mask')
        # print_intermediate_layer_output(model, data, 'pre_brnn')
        # print_intermediate_layer_output(model, data, 'brnn')
        # print_intermediate_layer_output(model, data, 'post_brnn')


        # data = [state[0], state[1], self.action]
        # model = self.critic.model
        # print_intermediate_layer_output(model, data, 'input_actions')
        # print_intermediate_layer_output(model, data, 'max_pool')
        # print_intermediate_layer_output(model, data, 'concatenate_inputs')
        # print_intermediate_layer_output(model, data, 'input_mask')
        # print_intermediate_layer_output(model, data, 'pre_brnn')
        # print_intermediate_layer_output(model, data, 'brnn')
        # print_intermediate_layer_output(model, data, 'post_brnn')


        # Apply Bell curve here to sample from to get action values
        # mu = 0
        # sigma = 0.5
        # y_offset = 0.107982 + 6.69737e-08
        # action = bell_curve(self.action[0], mu=mu, sigma=sigma, y_offset=y_offset, scaled=True)
        dist_limit = 5 # nm
        # dist = state[0][:,:, 4] * 110
        dist = (obs[:,:,4]+1)/2 * 110 # denormalize
        # dist = dist.reshape(action.shape)
        mul_factor = 80*np.ones(dist.shape)

        wheredist = np.where(dist<dist_limit)[0]
        mul_factor[wheredist] = dist[wheredist] / dist_limit * 80
        minus = np.where(self.action.reshape(self.action.shape[1])<0)[0]
        plus = np.where(self.action.reshape(self.action.shape[1])>=0)[0]
        # dheading = np.zeros(action.shape)
        # dheading[minus] = (action[minus] - 1) * mul_factor[minus]
        # dheading[plus] = np.abs(action[plus]-1) * mul_factor[plus]
        # action[minus] = -1*action[minus]

        # qdr = state[0][:,:,3].transpose()*360-180 # Denormalize
        qdr = obs[:,:,3].transpose() * 180 #- 180
        dheading = self.action[0] * mul_factor

        # dheading = 90*np.ones(dheading.shape)
        # print('heading', dheading, self.action)
        heading =  qdr + dheading
        print(qdr, dheading, heading, mul_factor, wheredist, dist)
        # print('action', self.action.ravel())
        return heading.ravel()[0:n_aircraft]


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

        action = np.zeros((traf.ntraf, 1))
        for i in range(traf.ntraf):
            action[i] = np.random.choice(a=np.arange(self.action_size), p=self.action[0,i,:])

        wp_ind = action % self.wp_action_size
        spd_ind = np.floor(action / self.wp_action_size).astype(int)

        speed_actions = [self.speed_values[i[0]] for i in spd_ind]

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

    def update_cum_reward(self, reward):
        # print(reward)
        self.cum_reward -= 1
        # self.cum_reward = 4 - len(reward)

        pass
        # self.cum_reward += reward

    def reset(self):
        if self.summary_counter%CONF.test_freq == 0:
            self.write_test_summaries(self.cum_reward)
        else:
            self.write_train_summaries(self.cum_reward)
        print("Episode {}, Score {}".format(self.summary_counter, self.cum_reward))
        if self.summary_counter % CONF.test_freq == 0:
            print("Starting test run at Episode {}".format(self.summary_counter+1))
        # else:
            # print("Start training Episode {}".format(self.summary_counter))

        self.cum_reward = 0
        # self.ac_dict={}


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