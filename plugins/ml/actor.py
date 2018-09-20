import numpy as np
from plugins.ml.bicnet import BiCNet
import tensorflow as tf
import keras.backend as K

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = BiCNet.build_actor(None, state_size, action_size, 64, 64, 'actor')
        self.target_model, self.target_weights, self.target_state = BiCNet.build_actor(None, state_size, action_size, 64, 64, 'actor_target')
        self.action_gradient = tf.placeholder(tf.float32,[None, None, action_size])
        # Negative action gradients are used for gradient ascent.
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def update_target_network(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in np.arange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)



if __name__ == '__main__':
    sess = tf.Session()
    actor = ActorNetwork(sess, 10, 5, 20, 32, 0.9, 0.0001)
