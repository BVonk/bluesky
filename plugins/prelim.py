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
    load_routes()
    config = {
        # The name of your plugin
        'plugin_name':     'PRELIM',

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


    # init_plugin() should always return these two dicts.
    return config, stackfunctions



def update():
    # train() if train_phase else test()
    # test()
    load_routes()

def train():
    eventmanager.update()


    for i in eventmanager.events:
        if env.actnum == 0:
            agent.sta = ETA(agent.acidx) + random.random() * 100
        next_state, reward, done, prev_state = env.step()
        prev_state = np.reshape(prev_state, [1, agent.state_size])
        next_state = np.reshape(next_state, [1, agent.state_size])
        if env.actnum>0:
            agent.remember(prev_state, agent.action, reward, next_state, done)
        if len(agent.memory) > agent.batch_size:
            agent.train()
        if not done:
            agent.act(next_state)

def test():
    eventmanager.update()

#    print('ETA {} wp {}'.format(ETA(agent.acidx), traf.ap.route[0].wpname[traf.ap.route[0].iactwp]))
    for i in eventmanager.events:
        if env.actnum == 0:
            agent.sta = ETA(agent.acidx) + random.random() * 100
#            print('STA ', agent.sta)
        next_state, reward, done, prev_state = env.step()

        if not done:
            agent.act_test(next_state)

        f = open(agent.testname, 'a')
        f.write("{},{},{},{},{},{},{},{},{},{}\n".format(env.ep, env.actnum, env.reward, env.state[0], env.state[1], env.state[2], agent.sta, sim.simt, traf.lat[env.acidx], traf.lon[env.acidx]))
        f.close()

        if env.ep>25:
            sim.stop()


def preupdate():
#    if len(traf.id) !=0:
#        agent.act(env.state)
#        env.act(agent.action)
    pass


def reset():
    pass


def load_routes():

    wpts = np.loadtxt('plugins/ml/routes/sugol.txt')
    i=0
    for j in range(wpts.shape[0]):
           for k in range(int(wpts.shape[1]/2)):
               stack.stack('DEFWPT RL{}{}{}, {}, {}, FIX'.format(i, j, k, wpts[j,k*2+1], wpts[j,k*2]))


    # stack.stack('DEFWPT SIM{}{}{}, {}, {}, FIX'.format(i, j, k, lat, lon))



class Env:
    def __init__(self):
        self.reward = 0
        self.done = False
        self.done_penalty = False
        self.prev_state = np.ones(state_size)
        self.state = np.ones(state_size)
        self.actnum = 0
        self.ep = 0
        self.fname = './output/log.csv'

        if not os.path.isfile(self.fname):
            f = open(self.fname, 'w')
            f.write("episode;step;reward;dist;t;hdg;sta;simt;lat;lon;epsilon\n")
            f.close()

        self.reset()


    def step(self):
        self.actnum += 1
        self.prev_state = self.state

        # Update state
        qdr, dist = qdrdist(traf.lat[self.acidx], traf.lon[self.acidx],
                            traf.ap.route[self.acidx].wplat[-2],
                            traf.ap.route[self.acidx].wplon[-2])
        t = agent.sta - ETA(agent.acidx)
        # print('STA {}, ETA {}, t {}'.format(agent.sta, ETA(agent.acidx), t))
        hdg_rel = degto180(qdr - traf.hdg[agent.acidx])
#        self.state = np.array([dist, t, hdg_ref,
#                               traf.tas[agent.acidx]])
        self.state = np.array([dist, t, hdg_rel/180.])

        # Check episode termination
        if dist<1 or t<-100:
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon -= agent.epsilon_decay
            self.done = True
            if dist >= 1:
                self.done_penalty = True
            else:
                self.done_penalty = False

        reward = self.gen_reward()
        # print('State {}'.format(self.state))
        # print('Reward {}, epsilon {}'.format(reward, agent.epsilon))
        if train_phase:
            self.log()

        if self.done:
            env.reset()

        return self.state, reward, self.done, self.prev_state


    def gen_reward(self):
        dist = self.state[0]
        t = self.state[1]
        dt = self.state[1]-self.prev_state[1]
        hdg = self.state[2]
        hdg_ref = 60.
        reward_penalty = 0




        a_dist = -0.22
        a_tpos = -0.2
        a_tneg = -0.1
        a_hdg = -0.07

        dist_rew = 3 + a_dist * dist


        if t>0 and dt>0:
            dt_rew = -1
        elif t>0 and dt<=0:
            dt_rew = 1
        elif t<=0 and dt>0:
            dt_rew = -1
        elif t<=0 and dt <=0:
            dt_rew = 1

        t_rew = 0

        if self.done and self.done_penalty:
            t_rew = -100.
        elif self.done and not self.done_penalty:
            t_rew = 10 + a_tpos * abs(t)
        #     hdg_rew = a_hdg * abs(degto180(hdg_ref - hdg))
        #
        # else:
        #     hdg_rew = 0



        self.reward = dist_rew + t_rew + dt_rew
        return self.reward


    def reset(self):
        if self.ep%25 == 0 and self.ep!=0 and train_phase:
            agent.save("./output/model{0:05}".format(self.ep))
            print("Saving model after {} episodes".format(self.ep))


        stack.stack('open ./scenario/bart/APP_SUGOL_RWY06.SCN')
        self.actnum = 0
        self.ep += 1
        self.done=False
        print("Episode ", self.ep)


    def log(self):
        dist = self.state[0]
        t = self.state[1]
        hdg = self.state[2]
        hdg_ref = 60.
        hdg_diff = (degto180(hdg_ref - hdg))
        f = open(self.fname, 'a')
        f.write("{};{};{};{};{};{};{};{};{};{};{}\n".format(self.ep, self.actnum, self.reward, dist, t, hdg_diff, agent.sta, sim.simt, traf.lat[self.acidx], traf.lon[self.acidx], agent.epsilon))
        f.close()


    def act(self, action):
        # Set new heading reference of the aircraft
        stack.stack(traf.id[self.acidx] + ' HDG ' + str(action))






















