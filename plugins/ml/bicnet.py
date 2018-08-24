from keras.models import Model
from keras.layers import Dense, Input, TimeDistributed, Bidirectional, LSTM

class BiCNet:
    @staticmethod
    def build_actor(max_agents, obs_dim, act_dim, H1, H2):
        S = Input(shape=(max_agents, obs_dim))
        h0 = TimeDistributed(Dense(H1, activation='relu'))(S)
        h1 = Bidirectional(LSTM(H2, return_sequences=True))(h0)
        h2 = TimeDistributed(Dense(act_dim, activation='relu'))(h1)
        model = Model(inputs=S, outputs=h2)
        return model, model.trainable_weights, S

    @staticmethod
    def build_critic(max_agents, obs_dim, H1, H2):
        S = Input(shape=(max_agents, obs_dim))
        h0 = TimeDistributed(Dense(H1, activation='relu'))(S)
        h1 = Bidirectional(LSTM(H2, return_sequences=True))(h0)
        h2 = TimeDistributed(Dense(1, activation='relu'))(h1)
        model = Model(inputs=S, outputs=h2)
        return model, model.trainable_weights, S


if __name__ == '__main__':
    actor, _, _ = BiCNet.build_actor(80, 50, 5, 64 ,64)
    critic, _, _ = BiCNet.build_critic(80, 50, 64, 64)
    actor.summary()
    critic.summary()
