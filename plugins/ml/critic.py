import numpy as np
from plugins.ml.bicnet import BiCNet
# from bicnet import BiCNet
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

        self.model, self.Q_values, self.actions, \
        self.states = BiCNet.build_critic(max_agents, state_size, action_size, 128, 128, LEARNING_RATE, 'critic')
        self.target_model, self.target_out, self.target_actions, \
        self.target_states = BiCNet.build_critic(max_agents, state_size, action_size, 128, 128, LEARNING_RATE, 'critic_target')
        self.action_grads = tf.gradients(self.Q_values, self.actions)

        self.ys = tf.placeholder(shape = (None, None, None), dtype=tf.float32, name='ys')
        cost = tf.square(self.Q_values-self.ys)
        self.unnormalized_gradients = tf.gradients(cost, self.model.trainable_weights)
        self.normalized_gradients = list(map(lambda x: tf.div(x, self.BATCH_SIZE), self.unnormalized_gradients))
        # self.grads = zip(self.gradients, self.model.trainable_weights)
        # self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(self.grads)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.optimize = self.optimizer.apply_gradients(zip(self.normalized_gradients, self.model.trainable_weights))

        # define variables to save the gradients in each batch
        self.accumulated_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()),
                                             trainable=False) for tv in
                                 self.model.trainable_weights]

        # define operation to reset the accumulated gradients to zero
        self.reset_gradients = [gradient.assign(tf.zeros_like(gradient)) for gradient in
                           self.accumulated_gradients]

        self.evaluate_batch = [accumulated_gradient.assign_add(gradient/self.BATCH_SIZE) for accumulated_gradient, gradient in zip(self.accumulated_gradients, self.unnormalized_gradients)]

        self.apply_gradients = self.optimizer.apply_gradients(zip(self.accumulated_gradients, self.model.trainable_weights))

        self.sess.run(tf.global_variables_initializer())



    def cgradient(self, states, actions, y):
        grads = self.sess.run(self.normalized_gradients, feed_dict={
            self.states: states,
            self.actions: actions,
            self.ys: y
        })
        return grads

    def train_separate(self, x, action, target):
        """
        Train on batch without using zero padding in batch processing
        """
        for i in range(self.BATCH_SIZE):
            if len(target[i].shape)!=3:
                target[i] = np.expand_dims(target[i], axis=0)
            self.sess.run(self.evaluate_batch, feed_dict={
                            self.ys: target[i],
                            self.actions: action[i],
                            self.states: x[i]
                            })
        self.sess.run(self.apply_gradients)
        self.sess.run(self.reset_gradients)


    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.states: states,
            self.actions: actions
        })[0]

    def gradients_separate(self, x, actions):
        """
        Compute gradients for the actor separate (not batch style)
        """
        gradients = []
        for i in range(self.BATCH_SIZE):
            gradient = self.sess.run(self.action_grads, feed_dict={
                        self.states: x[i],
                        self.actions: actions[i]})
            gradients.append(gradient[0])
        return gradients


    def train(self, states, actions, y):
        self.sess.run(self.optimize, feed_dict={
            self.states: states,
            self.actions: actions,
            self.ys: y
        })
        # loss = self.model.train_on_batch([states, actions], y)

    def predict(self, states, actions):
        return self.model.predict([states, actions])

    def predict_target(self, states, actions):
        return self.target_model.predict([states, actions])

    def predict_separate(self, state, actions):
        out = []
        for i in range(self.BATCH_SIZE):
            out.append(self.model.predict([state[i], actions[i]]))
        return out

    def predict_target_separate(self, state, actions):
        out = []
        for i in range(self.BATCH_SIZE):
            out.append(self.target_model.predict([state[i], actions[i]]))
        return out

    def update_learning_rate(self, learning_rate):
        self.LEARNING_RATE = learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

    def update_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in np.arange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def load_weights(self, filepath):
        self.model.load_weights(filepath, by_name=False)

    def load_target_weights(self, filepath):
        self.target_model.load_weights(filepath, by_name=False)

"""
class CriticNetwork_shared_obs(object):
    def __init__(self, sess, state_size, action_size, max_agents, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        K.set_session(sess)

        self.model, self.Q_values, self.actions, \
        self.states, self.shared_states = BiCNet.build_critic_shared_obs(max_agents, state_size[0], state_size[1], action_size, 64, 64, LEARNING_RATE, 'critic')
        self.target_model, self.target_out, self.target_actions, \
        self.target_states, self.target_shared_states = BiCNet.build_critic_shared_obs(max_agents, state_size[0], state_size[1], action_size, 64, 64, LEARNING_RATE, 'critic_target')
        self.action_grads = tf.gradients(self.Q_values, self.actions)

        self.ys = tf.placeholder(shape = (None, None, None), dtype=tf.float32, name='ys')
        cost = tf.square(self.Q_values-self.ys)
        self.unnormalized_gradients = tf.gradients(cost, self.model.trainable_weights)
        self.normalized_gradients = list(map(lambda x: tf.div(x, self.BATCH_SIZE), self.unnormalized_gradients))
        # self.grads = zip(self.gradients, self.model.trainable_weights)
        # self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(self.grads)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.optimize = self.optimizer.apply_gradients(zip(self.normalized_gradients, self.model.trainable_weights))

        # define variables to save the gradients in each batch
        self.accumulated_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()),
                                             trainable=False) for tv in
                                 self.model.trainable_weights]

        # define operation to reset the accumulated gradients to zero
        self.reset_gradients = [gradient.assign(tf.zeros_like(gradient)) for gradient in
                           self.accumulated_gradients]

        self.evaluate_batch = [accumulated_gradient.assign_add(gradient/self.BATCH_SIZE) for accumulated_gradient, gradient in zip(self.accumulated_gradients, self.unnormalized_gradients)]

        self.apply_gradients = self.optimizer.apply_gradients(zip(self.accumulated_gradients, self.model.trainable_weights))

        self.sess.run(tf.global_variables_initializer())
        print(self.model.summary())


    def cgradient(self, states, actions, y):
        grads = self.sess.run(self.normalized_gradients, feed_dict={
            self.states: states,
            self.actions: actions,
            self.ys: y
        })
        return grads

    def train_separate(self, x, action, target):
"""
        # Train on batch without using zero padding in batch processing
"""
        for i in range(self.BATCH_SIZE):
            if len(target[i].shape)!=3:
                target[i] = np.expand_dims(target[i], axis=0)
            self.sess.run(self.evaluate_batch, feed_dict={
                            self.ys: target[i],
                            self.actions: action[i],
                            self.states: x[i]
                            })
        self.sess.run(self.apply_gradients)
        self.sess.run(self.reset_gradients)


    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.states: states[0],
            self.shared_states: states[1],
            self.actions: actions
        })[0]

    def gradients_separate(self, x, actions):
"""
        # Compute gradients for the actor separate (not batch style)
"""
        gradients = []
        for i in range(self.BATCH_SIZE):
            gradient = self.sess.run(self.action_grads, feed_dict={
                        self.states: x[i],
                        self.actions: actions[i]})
            gradients.append(gradient[0])
        return gradients


    def train(self, states, actions, y):
        self.sess.run(self.optimize, feed_dict={
            self.states: states[0],
            self.shared_states: states[1],
            self.actions: actions,
            self.ys: y
        })
        # loss = self.model.train_on_batch([states, actions], y)

    def predict(self, states, actions):
        return self.model.predict([states[0], states[1], actions])

    def predict_target(self, states, actions):
        return self.target_model.predict([states[0], states[1], actions])

    def predict_separate(self, state, actions):
        out = []
        for i in range(self.BATCH_SIZE):
            out.append(self.model.predict([state[i], actions[i]]))
        return out

    def predict_target_separate(self, state, actions):
        out = []
        for i in range(self.BATCH_SIZE):
            out.append(self.target_model.predict([state[i], actions[i]]))
        return out

    def update_learning_rate(self, learning_rate):
        self.LEARNING_RATE = learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

    def update_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in np.arange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def load_weights(self, filepath):
        self.model.load_weights(filepath, by_name=False)

    def load_target_weights(self, filepath):
        self.target_model.load_weights(filepath, by_name=False)
"""

def bidirectional_layer(inputs):
    data = inputs[0]
    sequence_length=inputs[1]
    lstmUnits = 28
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits, forget_bias=1.0, state_is_tuple=True)
    out = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                          cell_bw=lstm_bw_cell,
                                          inputs=data,
                                          sequence_length=sequence_length,
                                          time_major = False,
                                          dtype=tf.float32)
    return out


class CriticNetwork_shared_obs(object):
    def __init__(self, sess, state_size, action_size, MAX_AIRCRAFT, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        K.set_session(sess)

        self.model, self.Q_values, self.actions, self.states, \
        self.shared_states = BiCNet.build_critic_shared_obs(MAX_AIRCRAFT, state_size[0], state_size[1], action_size, 32, 32, LEARNING_RATE, 'critic')
        self.target_model, self.target_out, self.target_actions, self.target_states, \
        self.target_shared_states = BiCNet.build_critic_shared_obs(MAX_AIRCRAFT, state_size[0], state_size[1], action_size, 32, 32, LEARNING_RATE, 'critic_target')
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

    def predict(self, states, actions):
        return self.model.predict([states[0], states[1], actions])

    def predict_target(self, states, actions):
        return self.target_model.predict([states[0], states[1], actions])


    def update_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in np.arange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)


if __name__ == '__main__':
    sess = tf.Session()
    critic = CriticNetwork(sess, 10, 5, 20, 32, 0.9, 0.0001)
