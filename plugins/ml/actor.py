import numpy as np
from plugins.ml.bicnet import BiCNet
# from bicnet import BiCNet
import tensorflow as tf
import keras.backend as K

class ActorNetwork(object):
    # def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
    def __init__(self, sess, state_size, action_size, max_aircraft, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model, self.weights, self.actions,  self.state = BiCNet.build_actor(max_aircraft, state_size, action_size, 32, 32, 'actor')
        # print(self.model.summary())
        self.target_model, self.target_weights, self.target_actions, self.target_state = BiCNet.build_actor(max_aircraft, state_size, action_size, 32, 32, 'actor_target')
        self.action_gradient = tf.placeholder(tf.float32,[None, None, action_size])
        # Negative action gradients are used for gradient ascent.

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        # define variables to save the gradients in each batch


        self.unnormalized_actor_gradients = tf.gradients(ys = self.model.output, xs = self.weights, grad_ys = -self.action_gradient) # -action_gradient to use gradient ascend. instead of gradient descend.
        self.gradients = tf.gradients(ys=self.model.output, xs=self.weights,
                                                         grad_ys=-self.action_gradient)  # -action_gradient to use gradient ascend. instead of gradient descend.
        self.normalized_gradients = list(map(lambda x: tf.div(x, self.BATCH_SIZE), self.gradients))

        self.accumulated_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()),
                                             trainable=False) for tv in
                                 self.model.trainable_weights]

        # define operation to reset the accumulated gradients to zero
        self.reset_gradients = [gradient.assign(tf.zeros_like(gradient)) for gradient in
                           self.accumulated_gradients]

        self.evaluate_batch = [accumulated_gradient.assign_add(gradient/self.BATCH_SIZE) for accumulated_gradient, gradient in zip(self.accumulated_gradients, self.gradients)]

        self.apply_gradients = self.optimizer.apply_gradients(zip(self.accumulated_gradients, self.model.trainable_weights))

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.normalized_gradients, self.weights))



        self.sess.run(tf.global_variables_initializer())

        # print(self.state.shape)

    def get_grads(self, states, action_grads):
        grads = self.sess.run(self.unnormalized_gradients, feed_dict={
                    self.state: states,
                    self.action_gradient: action_grads
        })
        return grads

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            # self.mask: mask,
            self.action_gradient: action_grads
        })

    def train_separate(self, x, action_grads):

        for i in range(self.BATCH_SIZE):
            self.sess.run(self.accumulated_gradients, feed_dict={
                self.state: x[i],
                self.action_gradient: action_grads[i]
            })

        self.sess.run(self.apply_gradients)
        self.sess.run(self.reset_gradients)


    def predict(self, inputs):
        return self.model.predict(inputs)


    def predict_target(self, inputs):
        return self.target_model.predict(inputs)


    def predict_separate(self, inputs):
        out = []
        for i in range(self.BATCH_SIZE):
            out.append(self.model.predict(inputs[i]))
        return out


    def predict_target_separate(self, inputs):
        out = []
        for i in range(self.BATCH_SIZE):
            out.append(self.target_model.predict(inputs[i]))
        return out


    def update_target_network(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in np.arange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)


class ActorNetwork_shared_obs(object):
    def __init__(self, sess, state_size, action_size, MAX_AIRCRAFT, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model, self.weights, self.actions,  self.state, self.shared_state = BiCNet.build_actor_shared_obs(MAX_AIRCRAFT, state_size[0], state_size[1], action_size, 16, 16, 'actor')
        print(self.model.summary())
        self.target_model, self.target_weights, self.target_actions, self.target_state, self.target_shared_state = BiCNet.build_actor_shared_obs(MAX_AIRCRAFT, state_size[0], state_size[1], action_size, 16, 16, 'actor_target')
        self.action_gradient = tf.placeholder(tf.float32,[None, None, action_size])
        # Negative action gradients are used for gradient ascent.
        self.unnormalized_actor_gradients = tf.gradients(self.model.output, self.weights, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.BATCH_SIZE), self.unnormalized_actor_gradients))

        grads = zip(self.actor_gradients, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

        # print(self.state.shape)

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states[0],
            self.shared_state: states[1],
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

    def predict_target(self, inputs):
        return self.target_model.predict(inputs)


if __name__ == '__main__':
    sess = tf.Session()
    actor = ActorNetwork(sess, 10, 5, 20, 32, 0.9, 0.0001)
