from keras.models import Model
from keras.layers import Dense, Input, TimeDistributed, Bidirectional, LSTM, Concatenate, Masking, Activation, Multiply, Lambda, Reshape
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
    def build_actor_shared_obs(max_agents, local_obs_dim, shared_obs_dim, act_dim, H1, H2, name):
        S1 = Input(shape=(max_agents, local_obs_dim), name='local_input_states')
        S2 = Input(shape=(max_agents, max_agents-1, shared_obs_dim), name='shared_input_states')

        # Pre-process S2

        reshaped = Reshape([S2.shape[1].value * S2.shape[2].value, S2.shape[3].value], name="shared_reshape")(S2)
        mask = Masking(mask_value=-999., name = 'shared_mask')(reshaped)
        layer1 = TimeDistributed(Dense(12, activation='relu'), name='shared_TD1')(mask)
        layer2 = TimeDistributed(Dense(12, activation='linear'), name='shared_TD2')(layer1)
        layer3 = Lambda(remove_mask, name='remove_mask')(layer2)
        layer4 = Reshape((S2.shape[1].value, S2.shape[2].value, layer3.shape[2].value), name='post_reshapse')(layer3)
        layer5 = Lambda(max_pool_sequence, name='max_pool')(layer4)
        S2_preprocessed = layer5
        # Merge S1, S2
        M = Concatenate(axis=-1, name='Merge_local_shared')([S1, S2_preprocessed])
        mask = Masking(mask_value=0., name = 'merged_mask')(M)
        h0 = TimeDistributed(Dense(H1, activation='relu'), name='pre_brnn')(mask)
        h1 = Bidirectional(LSTM(H2, return_sequences=True), name='brnn')(h0)
        h2 = TimeDistributed(Dense(act_dim, activation='tanh'), name='post_brnn')(h1)
        h3c = Activation('tanh', name='tanh')(h2)
        out = h3c
        model = Model(inputs=[S1, S2], outputs=out)
        layers = model.layers
        with tf.name_scope(name):
            for layer in layers:
                with tf.name_scope(layer.name):
                    for weight in layer.weights:
                        layer_name = weight.name.split('/')[-1]
                        tf.summary.histogram(layer_name, weight)

        return model, model.trainable_weights, out, S1, S2



    @staticmethod
    def build_critic(max_agents, obs_dim, act_dim, H1, H2, LR, name):
        S = Input(shape=(max_agents, obs_dim), name='input_states')
        A = Input(shape=(max_agents, act_dim), name='input_actions')
        M = Concatenate()([S, A])
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

    @staticmethod
    def build_critic_shared_obs(max_agents, obs_dim, shared_obs_dim, act_dim, H1, H2, LR, name):
        # Define inputs
        S1 = Input(shape=(max_agents, obs_dim), name='local_input_states')
        S2 = Input(shape=(max_agents, max_agents-1, shared_obs_dim), name='shared_input_states')
        A = Input(shape=(max_agents, act_dim), name='input_actions')

        # Pre-process S2
        # Set the masking value to ignore 0 padded sequence inputs.
        reshaped = Reshape([S2.shape[1].value * S2.shape[2].value, S2.shape[3].value], name="shared_reshape")(S2)
        mask = Masking(mask_value=-999., name = 'shared_mask')(reshaped)
        layer1 = TimeDistributed(Dense(12, activation='relu'), name='shared_TD1')(mask)
        layer2 = TimeDistributed(Dense(12, activation='linear'), name='shared_TD2')(layer1)
        layer3 = Lambda(remove_mask, name='remove_mask')(layer2)
        layer4 = Reshape((S2.shape[1].value, S2.shape[2].value, layer3.shape[2].value), name='post_reshapse')(layer3)
        layer5 = Lambda(max_pool_sequence, name='max_pool')(layer4)
        S2_preprocessed = layer5

        # Merge inputs
        M = Concatenate()([S1, S2_preprocessed, A])
        # Set the masking value to ignore 0 padded sequence inputs.
        mask = Masking(mask_value=0.)(M)
        h0 = TimeDistributed(Dense(H1, activation='relu'), name='pre_brnn')(mask)
        h1 = Bidirectional(LSTM(H2, return_sequences=True), name='brnn')(h0)
        h2 = TimeDistributed(Dense(1, activation='linear'), name='post_brnn')(h1)
        out = h2
        model = Model(inputs=[S1, S2 ,A], outputs=out)
        adam = Adam(lr=LR)
        model.compile(loss='mse', optimizer=adam)
        layers = model.layers
        with tf.name_scope(name):
            for layer in layers:
                with tf.name_scope(layer.name):
                    for weight in layer.weights:
                        layer_name = weight.name.split('/')[-1]
                        tf.summary.histogram(layer_name, weight)


        return model, out, A, S1, S2

def renormalized_mask_after_softmax(x):
    scale_factor = tf.divide(tf.constant(1, tf.float32), tf.reduce_sum(x))
    rescaled = tf.scalar_mul(scale_factor, x)
    return rescaled


def max_pool_sequence(x):
    max_pool = tf.reduce_max(x, axis=2, keep_dims=False)
    return max_pool

def remove_mask(x):
    return x

if __name__ == '__main__':
    tf.reset_default_graph()
    actor, _, _ = BiCNet.build_actor(80, 50, 5, 64 ,64)
    critic, _, _ = BiCNet.build_critic(80, 50, 5, 64, 64, 0.001)
    actor.summary()
    critic.summary()

