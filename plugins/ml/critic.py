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

        #Now create the model
        self.model, self.Q_values, self.actions, self.states = BiCNet.build_critic(None, state_size, action_size, 8, 8, LEARNING_RATE,
                                                                  'critic')
        self.target_model, self.target_out, self.target_actions, self.target_states = BiCNet.build_critic(None, state_size, action_size,
                                                                                       8, 8, LEARNING_RATE,
                                                                                       'critic_target')
        # Gradient of the critic output w.r.t. the actions.

        # self.actor_grads = tf.gradients(self.actions, )






        # gradient = tf.gradients(self.inputs)
        # states = tf.placeholder(tf.float32, [BATCH_SIZE, None, state_size], name='jemoeder')
        # actions = tf.placeholder(tf.float32, [BATCH_SIZE, None, action_size], name='jemoeder2')

        # Network target y_i
        # target_values = tf.placeholder(tf.float32, [None, None],'jemoeder3')

        self.action_grads = tf.gradients(self.Q_values, self.actions)

        self.sess.run(tf.global_variables_initializer())


        # raw_gradients = tf.multiply(target_values - self.Q_values,  tf.gradients(self.Q_values, self.model.weights))
        # critic_grads = tf.div(tf.reduce_sum(raw_gradients), BATCH_SIZE)
        # tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)






    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.states: states,
            self.actions: actions
        })[0]

    def train(self, states, actions, y):
        self.model.train_on_batch([states, actions], y)

    # def predict(self, inputs):
    #     return self.sess.run(self.Q_values, feed_dict={
    #         self.inputs: inputs
    #     })
    #
    # def predict_target(self, inputs):
    #     return self.sess.run(self.Q_values, feed_dict={
    #         self.target_inputs: inputs
    #     })

    def update_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in np.arange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)


if __name__ == '__main__':
    sess = tf.Session()
    critic = CriticNetwork(sess, 10, 5, 20, 32, 0.9, 0.0001)
