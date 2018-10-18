from keras.models import Model
from keras.layers import Dense, Input, TimeDistributed, Bidirectional, LSTM, Concatenate, Masking, Activation, Multiply, Lambda
from keras.optimizers import Adam
import tensorflow as tf

class BiCNet:
    @staticmethod
    def build_actor(max_agents, obs_dim, act_dim, H1, H2, name, config='continuous'):
        S = Input(shape=(max_agents, obs_dim), name='input_states')
        bool_mask = Input(shape =(max_agents, act_dim), dtype=tf.float32, name='bool_mask')
        # Set the masking value to ignore 0 padded sequence inputs.
        mask = Masking(mask_value=0., name='sequence_masking')(S)
        h0 = TimeDistributed(Dense(H1, activation='relu'), name='pre_brnn')(mask)

        h1 = Bidirectional(LSTM(H2, return_sequences=True), name='brnn')(h0)
        h2 = TimeDistributed(Dense(act_dim, activation='tanh'), name='post_brnn')(h1)
        # out1 = TimeDistributed(Dense(act_dim, activation='linear', name=name+'_actionoutput'))(h1)
        # out2 = TimeDistributed(Dense(act_dim, activation='linear', name=name+'_speedoutput'))(h1)
        h3c = Activation('tanh', name='tanh')(h2)
        h3d = Activation('softmax', name='softmax')(h2)
        actions = Multiply()([h3d, bool_mask])
        rescaled = Lambda(renormalized_mask_after_softmax)(actions)

        if config=='discrete':
            out = rescaled
            model = Model(inputs=[S, bool_mask], outputs=out)
            return model, model.trainable_weights, out, S, bool_mask
        elif config=='continuous':
            out = h3c
            model = Model(inputs=S, outputs=out)
            # layers = model.layers
            # with tf.name_scope(name):
            #     for layer in layers:
            #         with tf.name_scope(layer.name):
            #             for i in range(len(layer.weights)):
            #                 if i % 2 == 0:
            #                     tf.summary.histogram('weights', layer.weights[i])
            #                 else:
            #                     tf.summary.histogram('bias', layer.weights[i])
            layers = model.layers
            with tf.name_scope(name):
                for layer in layers:
                    with tf.name_scope(layer.name):
                        for weight in layer.weights:
                            layer_name = weight.name.split('/')[-1]
                            tf.summary.histogram(layer_name, weight)

            return model, model.trainable_weights, out, S




    @staticmethod
    def build_critic(max_agents, obs_dim, act_dim, H1, H2, LR, name):
        S = Input(shape=(max_agents, obs_dim), name='input_states')
        A = Input(shape=(max_agents, act_dim), name='input_actions')
        M = Concatenate()([S,A])
        # Set the masking value to ignore 0 padded sequence inputs.
        mask = Masking(mask_value=0.)(M)
        h0 = TimeDistributed(Dense(H1, activation='relu'), name='pre_brnn')(mask)
        h1 = Bidirectional(LSTM(H2, return_sequences=True), name='brnn')(h0)
        h2 = TimeDistributed(Dense(1, activation='linear'), name='post_brnn')(h1)
        out = h2
        model = Model(inputs=[S,A], outputs=out)
        adam = Adam(lr=LR)
        model.compile(loss='mse', optimizer=adam)
        layers = model.layers
        with tf.name_scope(name):
            for layer in layers:
                with tf.name_scope(layer.name):
                    for weight in layer.weights:
                        layer_name = weight.name.split('/')[-1]
                        tf.summary.histogram(layer_name, weight)
        return model, out, A, S


def renormalized_mask_after_softmax(x):
    scale_factor = tf.divide(tf.constant(1, tf.float32), tf.reduce_sum(x))
    rescaled = tf.scalar_mul(scale_factor, x)
    return rescaled

if __name__ == '__main__':
    tf.reset_default_graph()
    actor, _, _ = BiCNet.build_actor(80, 50, 5, 64 ,64)
    critic, _, _ = BiCNet.build_critic(80, 50, 5, 64, 64, 0.001)
    actor.summary()
    critic.summary()
