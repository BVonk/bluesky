from keras.models import Model
from keras.layers import Dense, Input, TimeDistributed, Bidirectional, LSTM, Concatenate
from keras.optimizers import Adam

class BiCNet:
    @staticmethod
    def build_actor(max_agents, obs_dim, act_dim, H1, H2, name):
        S = Input(shape=(max_agents, obs_dim), name=name+'_input_states')
        h0 = TimeDistributed(Dense(H1, activation='relu'))(S)
        h1 = Bidirectional(LSTM(H2, return_sequences=True))(h0)
        h2 = TimeDistributed(Dense(act_dim, activation='relu', name=name+'_output'))(h1)
        model = Model(inputs=S, outputs=h2)
        return model, model.trainable_weights, S

    @staticmethod
    def build_critic(max_agents, obs_dim, act_dim, H1, H2, LR, name):
        S = Input(shape=(max_agents, obs_dim), name=name+'_input_states')
        A = Input(shape=(max_agents, act_dim), name=name+'_input_actions')
        M = Concatenate()([S,A])
        h0 = TimeDistributed(Dense(H1, activation='relu'))(M)
        h1 = Bidirectional(LSTM(H2, return_sequences=True))(h0)
        h2 = TimeDistributed(Dense(1, activation='relu'), name=name+'_output')(h1)
        model = Model(inputs=[S,A], outputs=h2)
        adam = Adam(lr=LR)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S


if __name__ == '__main__':
    import tensorflow as tf
    tf.reset_default_graph()
    actor, _, _ = BiCNet.build_actor(80, 50, 5, 64 ,64)
    critic, _, _ = BiCNet.build_critic(80, 50, 5, 64, 64, 0.001)
    actor.summary()
    critic.summary()
