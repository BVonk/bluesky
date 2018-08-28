import numpy as np
from plugins.ml.bicnet import BiCNet

import keras.backend as K
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, max_agents, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = BiCNet.build_critic(None, state_size, action_size, 64, 64, LEARNING_RATE)
        self.target_model, self.target_action, self.target_state = BiCNet.build_critic(None, state_size, action_size, 64, 64, LEARNING_RATE)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]


    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in np.arange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)


if __name__ == '__main__':
    sess = tf.Session()
    critic = CriticNetwork(sess, 10, 5, 20, 32, 0.9, 0.0001)
