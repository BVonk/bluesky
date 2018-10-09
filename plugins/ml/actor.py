import numpy as np
from plugins.ml.bicnet import BiCNet
# from bicnet import BiCNet
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
        self.model, self.weights, self.actions,  self.state, self.mask = BiCNet.build_actor(None, state_size, action_size, 16, 16, 'actor')
        print(self.model.summary())
        self.target_model, self.target_weights, self.target_actions, self.target_state, self.target_mask = BiCNet.build_actor(None, state_size, action_size, 16, 16, 'actor_target')
        self.action_gradient = tf.placeholder(tf.float32,[None, None, action_size])
        # Negative action gradients are used for gradient ascent.
        self.unnormalized_actor_gradients = tf.gradients(self.model.output, self.weights, -self.action_gradient)

        # NUM_AGENTS=5
        # with tf.name_scope("actor_gradients"):
        #     grads = []
        #     for i in range(NUM_AGENTS):
        #         for j in range(NUM_AGENTS):
        #             grads.append(tf.gradients(self.actions[:, j], self.weights, -self.action_gradient[j][:, i]))
        #     grads = np.array(grads)
        #     self.unnormalized_actor_gradients = [tf.reduce_sum(list(grads[:, i]), axis=0) for i in
        #                                          range(len(self.weights))]
        #     self.actor_gradients = list(map(lambda x: tf.div(x, self.BATCH_SIZE), self.unnormalized_actor_gradients))

        self.actor_gradients = list(map(lambda x: tf.div(x, self.BATCH_SIZE), self.unnormalized_actor_gradients))

        grads = zip(self.actor_gradients, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

        # print(self.state.shape)

    def train(self, states, mask, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.mask: mask,
            self.action_gradient: action_grads
        })

    def update_target_network(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in np.arange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def predict(self, inputs):
        return self.model.predict(inputs)

if __name__ == '__main__':
    sess = tf.Session()
    actor = ActorNetwork(sess, 10, 5, 20, 32, 0.9, 0.0001)
