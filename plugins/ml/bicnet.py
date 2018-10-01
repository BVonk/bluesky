from keras.models import Model
from keras.layers import Dense, Input, TimeDistributed, Bidirectional, LSTM, Concatenate, Masking, Activation, Multiply
from keras.optimizers import Adam
import keras
import tensorflow as tf

class BiCNet:
    @staticmethod
    def build_actor(max_agents, obs_dim, act_dim, H1, H2, name):
        S = Input(shape=(max_agents, obs_dim), name=name+'_input_states')
        bool_mask = Input(shape =(max_agents, act_dim), dtype=tf.float32, name=name+'_bool_mask')
        # Set the masking value to ignore 0 padded sequence inputs.
        mask = Masking(mask_value=0.)(S)
        h0 = TimeDistributed(Dense(H1, activation='relu'))(mask)
        h1 = Bidirectional(LSTM(H2, return_sequences=True))(h0)
        h2 = TimeDistributed(Dense(act_dim, activation='linear', name=name + '_output'))(h1)
        out1 = TimeDistributed(Dense(act_dim, activation='linear', name=name+'_actionoutput'))(h1)
        out2 = TimeDistributed(Dense(act_dim, activation='linear', name=name+'_speedoutput'))(h1)
        h3 = Multiply()([h2, bool_mask])
        actions = Activation('softmax', name=name+'_softmax')(h3)
        out = actions
        model = Model(inputs=[S, bool_mask], outputs=out)
        return model, model.trainable_weights, out, S, bool_mask

    @staticmethod
    def build_critic(max_agents, obs_dim, act_dim, H1, H2, LR, name):
        S = Input(shape=(max_agents, obs_dim), name=name+'_input_states')
        A = Input(shape=(max_agents, act_dim), name=name+'_input_actions')
        M = Concatenate()([S,A])
        # Set the masking value to ignore 0 padded sequence inputs.
        mask = Masking(mask_value=0.)(M)
        h0 = TimeDistributed(Dense(H1, activation='relu'))(mask)
        h1 = Bidirectional(LSTM(H2, return_sequences=True))(h0)
        h2 = TimeDistributed(Dense(1, activation='linear'), name=name+'_output')(h1)
        out = h2
        model = Model(inputs=[S,A], outputs=out)
        adam = Adam(lr=LR)
        model.compile(loss='mse', optimizer=adam)
        return model, out, A, S


if __name__ == '__main__':
    import tensorflow as tf
    tf.reset_default_graph()
    actor, _, _ = BiCNet.build_actor(80, 50, 5, 64 ,64)
    critic, _, _ = BiCNet.build_critic(80, 50, 5, 64, 64, 0.001)
    actor.summary()
    critic.summary()
