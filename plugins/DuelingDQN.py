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
    # Addtional initilisation code
    # Pan to waypoint with fixed zoom level and create a random aircraft.
    # Configuration parameters
    global env, agent, eventmanager, state_size, train_phase, model_fname
    state_size = 3
    action_size = 3
    train_phase = True
    model_fname = ''#'output/model00375'
    env = Env()

    sess = tf.Session()

    K.set_session(sess)
    agent = DuelingDQNAgent(state_size, action_size)
    print("Agent initialized")
    eventmanager = Eventmanager()

    config = {
        # The name of your plugin
        'plugin_name':     'DUELDQN',

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




### Other functions of your plugin
#def conflict_resolution():
#    print(traf.asas.conflist_now)
#    for conf in traf.asas.conflist_now:
#        intruder = conf.split(' ')[-1]
#        stack.stack(intruder + ' DEL')
#        stack.stack('ECHO Deleted intruder '+intruder)

def update():
    train() if train_phase else test()
#    test()

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


class Env:
    def __init__(self):
        self.acidx = 0
        self.reward = 0
        self.done = False
        self.done_penalty = False
        self.state = np.ones((1, state_size))
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
        prev_state = self.state

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
        if dist<1 or agent.sta-sim.simt<-60 or t<-100:
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

        return self.state, reward, self.done, prev_state


    def gen_reward(self):
        dist = self.state[0]
        t = self.state[1]
        hdg = self.state[2]
        hdg_ref = 60.
        reward_penalty = 0




        a_dist = -0.22
        a_tpos = -0.05
        a_tneg = -0.1
        a_hdg = -0.07

        dist_rew = 3 + a_dist * dist

        if t>0:
            t_rew = 5 + a_tpos * t
        else:
            t_rew = a_tneg * abs(t)

        if self.done and self.done_penalty:
            reward_penalty = -10000.
        #     hdg_rew = a_hdg * abs(degto180(hdg_ref - hdg))
        #
        # else:
        #     hdg_rew = 0



        self.reward = dist_rew + t_rew + reward_penalty# + hdg_rew
        return self.reward


    def reset(self):
        if self.ep%25 == 0 and self.ep!=0 and train_phase:
            agent.save("./output/model{0:05}".format(self.ep))
            print("Saving model after {} episodes".format(self.ep))


        stack.stack('open ./scenario/4d.SCN')
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



class ActorCritic:
    def __init__(self, state_size, action_size, sess):
        self.acidx = 0
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau   = .125
        self.actions = [self.act1, self.act4, self.act5]
        self.sta = 0
        self.action = 0
        self.batch_size = 32
        self.trainsteps = 0
        self.c = 1000


        # Actor model
        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size])

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights,
                                        -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # Critic model
        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)

        self.sess.run(tf.initialize_all_variables())


    def create_actor_model(self):
        state_input = Input(shape=(self.state_size,))
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.action_size, activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.state_size,))
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=(self.action_size,))
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        model.summary()
        return state_input, action_input, model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def _train_actor(self, samples):
        for sample in samples:
            state, action, reward, next_state, _ = sample
            predicted_action = self.actor_model.predict(state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                    self.critic_state_input: state,
                    self.critic_action_input: predicted_action
                    })[0]

            self.sess.run(self.optimize, feed_dicht={
                    self.actor_state_input: state,
                    self.actor_critic_grad: grads})

    def _train_critic(self, samples):
        for sample in samples:
            state, action, reward, next_state, done = sample
            print("state shape ", state.shape,next_state)
            if not done:
                target_action = self.target_actor_model.predict(next_state)
                future_reward = self.target_critic_model.predict(
                        [next_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([state, np.array(action)], reward, verbose=0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        self._train_critic(samples)
        self._train_actor(samples)
        self.trainsteps += 1

        if self.train_steps%self.c == 0:
            self.update_target()

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()



    def act1(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.

        dqdr = 15
        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]
        turnrad = traf.tas[self.acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[self.acidx])) * g0) # [m]

        #Turn right so add bearing
#        qdr = traf.qdr[self.acidx] + 90

        latR, lonR = qdrpos(latA, lonA, traf.hdg[self.acidx] + 90, turnrad/nm) # [deg, deg]
        # Rotate vector
        latB, lonB = qdrpos(latR, lonR, traf.hdg[self.acidx] - 90 + dqdr, turnrad/nm) # [deg, deg]
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act2(self):
        pass


    def act3(self):
        pass


    def act4(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.

        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]

        latB, lonB = qdrpos(latA, lonA, traf.hdg[self.acidx], 0.25)
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act5(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.
        dqdr = 15
        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]
        turnrad = traf.tas[self.acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[self.acidx])) * g0) # [m]

        #Turn right so add bearing
#        qdr = traf.qdr[self.acidx] + 90

        latR, lonR = qdrpos(latA, lonA, traf.hdg[self.acidx] - 90, turnrad/nm) # [deg, deg]
        # Rotate vector
        latB, lonB = qdrpos(latR, lonR, traf.hdg[self.acidx] + 90 - dqdr, turnrad/nm) # [deg, deg]
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            agent.action = random.choice(np.arange(0, self.action_size))

        else:
            act_values = self.model.predict(env.state.reshape((1,agent.state_size)))
            print('episode {}, Qvalues {}'.format(env.ep, act_values[0]))
            self.action = np.argmax(act_values[0])

        # Pick the action with the highest Q-value
        self.actions[self.action]()


class DuelingDQNAgent:
    def __init__(self, state_size, action_size):
        # self.fname = 'output/run_0.6/model00600'
        self.acidx = 0
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.98    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = (1-self.epsilon_min)/250.
        self.learning_rate = 0.001
        self.clipvalue = 0.1
        self.batch_size = 32
        self.done = False
        self.sta = 0
        self.action = 0
        self.actions = [self.act1, self.act4, self.act5]
        self.replaysteps = 0
        self.model = self._build_model()
        self.targetmodel = self._build_model()

        if train_phase and not model_fname=='':
            self.load(model_fname)
#
        elif not train_phase:
            self.load(model_fname)
            self.testname = "output/test{}.csv".format(model_fname[-5:])
            f = open(self.testname, 'w')
            f.write("episode,step,reward,state1,state2,state3,STA,t,lat,lon\n")
            f.close()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        state_input = Input(shape=(self.state_size,))
        A1 = Dense(128, activation='relu')(state_input)
        A2 = Dense(128, activation='relu')(A1)
        A3 = Dense(self.action_size, activation='linear')(A2)
        Amean = K.mean(A2, keepdims=True)
        # A_avg = tf.subtract(A2, Amean)
        # Amean = Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
        #                                      output_shape=(self.action_size,))(A2)
        A4 = Lambda(lambda a: tf.subtract(a, K.mean(a)), output_shape=(self.action_size,))(A3)
        # Value network
        V1 = Dense(128, activation='relu')(state_input)
        V2 = Dense(128, activation='relu')(V1)
        V3 = Dense(1, activation='linear')(V2)
        V2repeat = RepeatVector(3)(V3)

        output = Add()([V3, A4])
        model = Model(input=state_input, output=output)
        adam = Adam(lr=self.learning_rate, clipvalue = self.clipvalue)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def calc_turn_wp(self, delta_qdr):
        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]
        turnrad = traf.tas[self.acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[self.acidx])) * g0) # [m]

        #Turn right so add bearing
#        qdr = traf.qdr[self.acidx] + 90

        # Centre of turning circle
        latR, lonR = qdrpos(latA, lonA, traf.hdg[self.acidx] + 90, turnrad/nm) # [deg, deg]
        # Rotate vector
        latB, lonB = qdrpos(latR, lonR, traf.hdg[self.acidx] - 90 + delta_qdr, turnrad/nm) # [deg, deg]


    def act1(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.

        dqdr = 15
        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]
        turnrad = traf.tas[self.acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[self.acidx])) * g0) # [m]

        #Turn right so add bearing
#        qdr = traf.qdr[self.acidx] + 90

        latR, lonR = qdrpos(latA, lonA, traf.hdg[self.acidx] + 90, turnrad/nm) # [deg, deg]
        # Rotate vector
        latB, lonB = qdrpos(latR, lonR, traf.hdg[self.acidx] - 90 + dqdr, turnrad/nm) # [deg, deg]
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act2(self):
        pass


    def act3(self):
        pass


    def act4(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.

        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]

        latB, lonB = qdrpos(latA, lonA, traf.hdg[self.acidx], 0.25)
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act5(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.
        dqdr = 15
        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]
        turnrad = traf.tas[self.acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[self.acidx])) * g0) # [m]

        #Turn right so add bearing
#        qdr = traf.qdr[self.acidx] + 90

        latR, lonR = qdrpos(latA, lonA, traf.hdg[self.acidx] - 90, turnrad/nm) # [deg, deg]
        # Rotate vector
        latB, lonB = qdrpos(latR, lonR, traf.hdg[self.acidx] + 90 - dqdr, turnrad/nm) # [deg, deg]
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act(self, state):
        act_values = self.model.predict(env.state.reshape((1, agent.state_size)))
        print('episode {}, state {}, Qvalues {}'.format(env.ep, env.state, act_values[0]))
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            agent.action = random.choice(np.arange(0, self.action_size))

        else:

            self.action = np.argmax(act_values[0])

        # Pick the action with the highest Q-value
        self.actions[self.action]()

    def act_test(self, state):
        act_values = self.model.predict(env.state.reshape((1,agent.state_size)))
        self.action = np.argmax(act_values[0])
        print('episode {}, state {}, Qvalues {}'.format(env.ep, env.state, act_values[0]))
        self.actions[self.action]()

    def update_target_weights(self):
        weights = self.model.get_weights()
        target_model = self.targetmodel.set_weights(weights)

    def train(self):
        c = 1000
        batch_size = 32
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:

            # If done, make the target reward
            target = reward

            if not done:
                # Predict the future discounted reward
                target = reward + self.gamma * np.amax(self.targetmodel.predict(next_state.reshape((1,agent.state_size)))[0])
                # print("target ", target)

            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that target_f
            target_f = self.model.predict(state.reshape((1,self.state_size)))
            # print("target_f ", target_f)
            target_f[0][action] = target
            # print("target_f ", target_f)

            # Train the Neural Net with the state and target_f
#            print('target', target)
#            print(st)
            self.model.fit(state.reshape((1,agent.state_size)), target_f, epochs=1, verbose=0)

        if self.replaysteps%c == 0:
            self.update_target_weights()


#        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay

        self.replaysteps += 1


    def load(self, name):
        print("Loading weights from {}".format(name))
        self.model.load_weights(name + '.hdf5')

        if train_phase:
            env.ep = int(model_fname[-5:])
            self.epsilon = max(self.epsilon_min, self.epsilon - env.ep * self.epsilon_decay)
            self.targetmodel.load_weights(name + 'target.hdf5')
            self.memory = pickle.load(open(name + '.p','rb'))


    def save(self, name):
        self.model.save_weights(name + '.hdf5')
        self.targetmodel.save_weights(name + 'target.hdf5')
        pickle.dump(self.memory, open(name + '.p', 'wb'))



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.fname = 'output/run_0.6/model00600'
        self.acidx = 0
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.98    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9954
        self.learning_rate = 0.0005
        self.batch_size = 32
        self.done = False
        self.sta = 0
        self.action = 0
        self.actions = [self.act1, self.act4, self.act5]
        self.replaysteps = 0
        self.model = self._build_model()

        if train_phase and not self.fname=='':
            self.load(self.fname)
#
        elif not train_phase:
            self.load(self.fname)
            self.testname = "output/test{}.csv".format(self.fname[-5:])
            f = open(self.testname, 'w')
            f.write("episode,step,reward,state1,state2,state3,STA,t,lat,lon\n")
            f.close()

        self.targetmodel = self.model
        print(self.model.summary())

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate)
                      )
        print(model.summary())
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def calc_turn_wp(self, delta_qdr):
        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]
        turnrad = traf.tas[self.acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[self.acidx])) * g0) # [m]

        #Turn right so add bearing
#        qdr = traf.qdr[self.acidx] + 90

        # Centre of turning circle
        latR, lonR = qdrpos(latA, lonA, traf.hdg[self.acidx] + 90, turnrad/nm) # [deg, deg]
        # Rotate vector
        latB, lonB = qdrpos(latR, lonR, traf.hdg[self.acidx] - 90 + delta_qdr, turnrad/nm) # [deg, deg]


    def act1(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.

        dqdr = 15
        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]
        turnrad = traf.tas[self.acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[self.acidx])) * g0) # [m]

        #Turn right so add bearing
#        qdr = traf.qdr[self.acidx] + 90

        latR, lonR = qdrpos(latA, lonA, traf.hdg[self.acidx] + 90, turnrad/nm) # [deg, deg]
        # Rotate vector
        latB, lonB = qdrpos(latR, lonR, traf.hdg[self.acidx] - 90 + dqdr, turnrad/nm) # [deg, deg]
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act2(self):
        pass


    def act3(self):
        pass


    def act4(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.

        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]

        latB, lonB = qdrpos(latA, lonA, traf.hdg[self.acidx], 0.25)
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act5(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.
        dqdr = 15
        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]
        turnrad = traf.tas[self.acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[self.acidx])) * g0) # [m]

        #Turn right so add bearing
#        qdr = traf.qdr[self.acidx] + 90

        latR, lonR = qdrpos(latA, lonA, traf.hdg[self.acidx] - 90, turnrad/nm) # [deg, deg]
        # Rotate vector
        latB, lonB = qdrpos(latR, lonR, traf.hdg[self.acidx] + 90 - dqdr, turnrad/nm) # [deg, deg]
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            agent.action = random.choice(np.arange(0, self.action_size))

        else:
            act_values = self.model.predict(env.state.reshape((1,agent.state_size)))
            print('episode {}, Qvalues {}'.format(env.ep, act_values[0]))
            self.action = np.argmax(act_values[0])

        # Pick the action with the highest Q-value
        self.actions[self.action]()

    def act_test(self, state):
        act_values = self.model.predict(env.state.reshape((1,agent.state_size)))
        self.action = np.argmax(act_values[0])
        print('Qvalues {}'.format(act_values[0]))
        self.actions[self.action]()

    def train(self):
        batch_size = 32
        c = 1000
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:

            # If done, make the target reward
            target = reward

            if not done:
                # Predict the future discounted reward
                target = reward + self.gamma * np.amax(self.targetmodel.predict(next_state.reshape((1,agent.state_size)))[0])

            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that target_f
            target_f = self.model.predict(state.reshape((1,self.state_size)))
            target_f[0][action] = target

            # Train the Neural Net with the state and target_f
#            print('target', target)
#            print(st)
            self.model.fit(state.reshape((1,agent.state_size)), target_f, epochs=1, verbose=0)

        if self.replaysteps%c == 0:
            self.targetmodel = self.model


#        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay

        self.replaysteps += 1


    def load(self, name):
        print("Loading weights from {}".format(name))
        self.model.load_weights(name + '.hdf5')
        if train_phase:
            env.ep = int(self.fname[-5:])
            self.epsilon = max(self.epsilon_min, self.epsilon - env.ep * 0.9/1000.)
            self.memory = pickle.load(open(name + '.p','rb'))


    def save(self, name):
        self.model.save_weights(name + '.hdf5')
        pickle.dump(self.memory, open(name + '.p', 'wb'))


class Eventmanager():
    def __init__(self):
        self.eventidx = []


    def update(self):
        self.events = []
        for acidx in range(traf.ntraf):
            if traf.ap.route[acidx].iactwp == len(traf.ap.route[acidx].wptype) - 2:
                self.events.append(acidx)
