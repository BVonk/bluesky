import numpy as np
from plugins.ml.bicnet import BiCNet
# from bicnet import BiCNet
import keras.backend as K
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        K.set_session(sess)

        self.model, self.Q_values, self.actions, \
        self.states = BiCNet.build_critic(None, state_size, action_size, 8, 8, LEARNING_RATE, 'critic')
        self.target_model, self.target_out, self.target_actions, \
        self.target_states = BiCNet.build_critic(None, state_size, action_size, 8, 8, LEARNING_RATE, 'critic_target')
        self.action_grads = tf.gradients(self.Q_values, self.actions)
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.states: states,
            self.actions: actions
        })[0]

    def train(self, states, actions, y):
        loss = self.model.train_on_batch([states, actions], y)
        return loss

    def update_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in np.arange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)


class CriticNetwork_shared_obs(object):
    def __init__(self, sess, state_size, shared_state_size, action_size, MAX_AIRCRAFT, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        K.set_session(sess)

        self.model, self.Q_values, self.actions, self.states, \
        self.shared_states = BiCNet.build_critic_shared_obs(MAX_AIRCRAFT, state_size, shared_state_size, action_size, 32, 32, LEARNING_RATE, 'critic')
        self.target_model, self.target_out, self.target_actions, self.target_states, \
        self.target_shared_states = BiCNet.build_critic_shared_obs(MAX_AIRCRAFT, state_size, shared_state_size, action_size, 32, 32, LEARNING_RATE, 'critic_target')
        self.action_grads = tf.gradients(self.Q_values, self.actions)
        self.sess.run(tf.global_variables_initializer())
        print(self.model.summary())
    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.states: states[0],
            self.shared_states: states[1],
            self.actions: actions
        })[0]

    def train(self, states, actions, y):
        self.model.train_on_batch([states[0], states[1], actions], y)

    def update_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in np.arange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

if __name__ == '__main__':
    sess = tf.Session()
    critic = CriticNetwork(sess, 10, 5, 20, 32, 0.9, 0.0001)
