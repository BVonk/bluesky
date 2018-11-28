from keras.models import Model
from keras.layers import Dense, Input, TimeDistributed, Bidirectional, LSTM, Concatenate, Masking, Activation, Multiply, Lambda, Reshape
from keras.optimizers import Adam
import tensorflow as tf

class BiCNet:
    @staticmethod
    def build_actor(max_agents, obs_dim, act_dim, H1, H2, name):
        max_agents=None
        S = Input(shape=(max_agents, obs_dim), name='input_states')
        # Set the masking value to ignore 0 padded sequence inputs.
        mask = Masking(mask_value=0., name='sequence_masking')(S)
        h0 = TimeDistributed(Dense(H1, activation='relu'), name='pre_brnn')(mask)
        h1 = Bidirectional(LSTM(H2, return_sequences=True), name='brnn')(h0)
        h2 = TimeDistributed(Dense(act_dim, activation='linear'), name='post_brnn')(h1)
        h3c = Activation('tanh', name='tanh')(h2)
        out = h3c
        model = Model(inputs=S, outputs=out)

        # Create summaries for layer visualisation
        layers = model.layers
        with tf.name_scope(name):
            for layer in layers:
                with tf.name_scope(layer.name):
                    for weight in layer.weights:
                        layer_name = weight.name.split('/')[-1]
                        tf.summary.histogram(layer_name, weight)

        return model, model.trainable_weights, model.output, S

    @staticmethod
    def build_actor_shared_obs(max_agents, local_obs_dim, shared_obs_dim, act_dim, H1, H2, name):
        max_agents = None
        S1 = Input(shape=(max_agents, local_obs_dim), name='local_input_states')
        S2 = Input(shape=(max_agents, shared_obs_dim), name='shared_input_states')

        # Pre-process S2

        # reshaped = Reshape([S2.shape[1].value * S2.shape[2].value, S2.shape[3].value], name="shared_reshape")(S2)
        # reshaped = Reshape([tf.shape(S2)[1] * tf.shape(S2)[2], S2.shape[3].value], name="shared_reshape")(S2)
        # reshaped = Lambda(reshape_input, name="shared_reshape")(S2)
        mask = Masking(mask_value=-999., name = 'shared_mask')(S2)
        layer1 = TimeDistributed(Dense(12, activation='relu'), name='shared_TD1')(mask)
        layer2 = TimeDistributed(Dense(12, activation='linear'), name='shared_TD2')(layer1)
        layer3 = Lambda(remove_mask, name='remove_mask')(layer2)
        # layer4 = Reshape((S2.shape[1].value, S2.shape[2].value, layer3.shape[2].value), name='reshape')(layer3)
        layer5 = Lambda(max_pool_sequence, name='max_pool')(layer3)
        S2_preprocessed = layer5
        # Merge S1, S2

        # S1_preprocessed = Lambda(remove_zeros, name='remove_zeros')(S1)
        M = Concatenate(axis=-1, name='Merge_local_shared')([S1, S2_preprocessed])
        mask = Masking(mask_value=0., name = 'merged_mask')(M)
        # print("Hallo kunnen jullie mij horen!", S1_preprocessed.shape, S2_preprocessed.shape)
        h0 = TimeDistributed(Dense(H1, activation='relu'), name='pre_brnn')(mask)
        h1 = Bidirectional(LSTM(H2, return_sequences=True), name='brnn')(h0)
        h2 = TimeDistributed(Dense(act_dim, activation='linear'), name='post_brnn')(h1)
        h3c = Activation('tanh', name='tanh')(h2)
        out = h3c
        model = Model(inputs=[S1, S2], outputs=out)
        layers = model.layers
        # with tf.name_scope(name):
        #     for layer in layers:
        #         with tf.name_scope(layer.name):
        #             for weight in layer.weights:
        #                 layer_name = weight.name.split('/')[-1]
        #                 tf.summary.histogram(layer_name, weight)

        return model, model.trainable_weights, model.output, S1, S2



    @staticmethod
    def build_critic(max_agents, obs_dim, act_dim, H1, H2, LR, name):
        max_agents = None
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

        #Create summaries for layer visualisation
        layers = model.layers
        with tf.name_scope(name):
            for layer in layers:
                with tf.name_scope(layer.name):
                    for weight in layer.weights:
                        layer_name = weight.name.split('/')[-1]
                        tf.summary.histogram(layer_name, weight)

        return model, model.output, A, S

    @staticmethod
    def build_critic_shared_obs(max_agents, obs_dim, shared_obs_dim, act_dim, H1, H2, LR, name):
        # Define inputs
        max_agents = None
        S1 = Input(shape=(max_agents, obs_dim), name='local_input_states')
        S2 = Input(shape=(max_agents, shared_obs_dim), name='shared_input_states')
        A = Input(shape=(max_agents, act_dim), name='input_actions')

        # Pre-process S2
        # Set the masking value to ignore 0 padded sequence inputs.
        # reshaped = Reshape([S2.shape[1].value * S2.shape[2].value, S2.shape[3].value], name="shared_reshape")(S2)
        # reshaped = Reshape([tf.shape(S2)[1] * tf.shape(S2)[2], S2.shape[3].value], name="shared_reshape")(S2)
        # reshaped = Lambda(reshape_input, name="shared_reshape")(S2)
        mask = Masking(mask_value=-999., name = 'shared_mask')(S2)
        layer1 = TimeDistributed(Dense(12, activation='relu'), name='shared_TD1')(mask)
        layer2 = TimeDistributed(Dense(12, activation='linear'), name='shared_TD2')(layer1)
        layer3 = Lambda(remove_mask, name='remove_mask')(layer2)
        # layer4 = Reshape((S2.shape[1].value, S2.shape[2].value, layer3.shape[2].value), name='post_reshape')(layer3)
        layer5 = Lambda(max_pool_sequence, name='max_pool')(layer3)
        S2_preprocessed = layer5

        # A_preprocessed = Lambda(remove_zeros, name='remove_zeros_A')(A)
        # S1_preprocessed = Lambda(remove_zeros, name='remove_zeros_S1')(S1)

        # Merge inputs
        M = Concatenate(name='concatenate_inputs')([S1, S2_preprocessed, A])
        # Set the masking value to ignore 0 padded sequence inputs.
        mask = Masking(mask_value=0., name='input_mask')(M)
        h0 = TimeDistributed(Dense(H1, activation='relu'), name='pre_brnn')(mask)
        h1 = Bidirectional(LSTM(H2, return_sequences=True), name='brnn')(h0)
        h2 = TimeDistributed(Dense(1, activation='linear'), name='post_brnn')(h1)
        out = h2
        model = Model(inputs=[S1, S2 ,A], outputs=out)
        adam = Adam(lr=LR)
        model.compile(loss='mse', optimizer=adam)
        layers = model.layers
        # with tf.name_scope(name):
        #     for layer in layers:
        #         with tf.name_scope(layer.name):
        #             for weight in layer.weights:
        #                 layer_name = weight.name.split('/')[-1]
        #                 tf.summary.histogram(layer_name, weight)


        return model, model.output, A, S1, S2

def renormalized_mask_after_softmax(x):
    scale_factor = tf.divide(tf.constant(1, tf.float32), tf.reduce_sum(x))
    rescaled = tf.scalar_mul(scale_factor, x)
    return rescaled


def max_pool_sequence1(x):
    """

    :param x Input tensor for max_pool_sequence:
    :return:
    """
    # For the difference between tf.shape(x) and x.shape refer to https://ipt.ai/2018/08/shapeless-variables/
    # tf.shape(x) returns the dynamic shape, where x.shape returns the static shape at the moment the graph is build
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(x, zero)
    x1 = tf.boolean_mask(x, where)
    shape = tf.shape(x1)[0]
    one = tf.Variable(1, dtype=tf.float32, trainable=False)
    minone = tf.constant(-1, dtype=tf.float32)
    two = tf.constant(2, dtype=tf.float32)
    four = tf.constant(4, dtype=tf.float32)
    add = tf.add(one, tf.multiply(four, tf.divide(tf.cast(shape, tf.float32), tf.cast(x.shape[-1], tf.float32))))
    sqrt = tf.sqrt(add)
    n_aircraft = tf.divide(tf.add(minone, sqrt), two)
    print("n_aircraft", n_aircraft)
    x2 = tf.reshape(x1, [tf.cast(one, tf.int32), tf.cast(tf.add(n_aircraft, one), tf.int32), tf.cast(n_aircraft, tf.int32), x.shape[-1]], name='reshape_x1')
    max_pool = tf.reduce_max(x2, axis=2, keep_dims=False)
    return max_pool


def max_pool_sequence(x):
    """

    :param x Input tensor for max_pool_sequence:
    :return:
    """
    # For the difference between tf.shape(x) and x.shape refer to https://ipt.ai/2018/08/shapeless-variables/
    # tf.shape(x) returns the dynamic shape, where x.shape returns the static shape at the moment the graph is build

    # Compute number of aircraft
    zero = tf.constant(0, dtype=tf.float32)
    y = tf.maximum(tf.constant(1, dtype=tf.int32), tf.shape(x)[1])
    one = tf.Variable(1, dtype=tf.float32, trainable=False)
    minone = tf.constant(-1, dtype=tf.float32)
    two = tf.constant(2, dtype=tf.float32)
    four = tf.constant(4, dtype=tf.float32)
    add = tf.add(one, tf.multiply(four, tf.cast(y, tf.float32)))
    sqrt = tf.sqrt(add)
    n_aircraft = tf.divide(tf.add(minone, sqrt), two)

    # Max pool
    mask = tf.equal(x, zero)
    index = tf.where(mask)
    index.shape[0]
    shape = tf.shape(x, out_type=tf.int64)
    # fill = tf.Variable(1000., shape=tf.shape(index)[0])
    values = tf.fill([tf.shape(index)[0]], -99999.)
    sparse = tf.sparse.SparseTensor(indices=index, values=values, dense_shape=shape)
    # sparse = tf.sparse.reorder(sparse)
    dense = tf.sparse.to_dense(sparse, default_value=0.)
    x1 = tf.add(x, dense)
    x2 = tf.reshape(x1, [tf.shape(x)[0],
                         tf.maximum(tf.constant(1, dtype=tf.int32), tf.cast(tf.add(n_aircraft, one), tf.int32)),
                         tf.maximum(tf.constant(1, dtype=tf.int32),tf.cast(n_aircraft, tf.int32)),
                         x.shape[-1]],
                    name='reshape_x1')
    max_pool = tf.reduce_max(x2, axis=2, keepdims=False)

    return max_pool


def remove_zeros(x):
    batch_size = tf.shape(x)[0]
    one = tf.Variable(1, dtype=tf.int32, trainable=False)
    # shape = tf.cast(tf.divide(tf.shape(x)[0],tf.shape(x)[0]), tf.int32)
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(x, zero)
    x1 = tf.boolean_mask(x, where)
    n_aircraft = tf.Variable(1, trainable=False)
    n_aircraft = tf.cast(tf.divide(tf.cast(tf.divide(tf.shape(x1)[-1], tf.shape(x)[0]), tf.int32), tf.shape(x)[-1]), dtype=tf.int32)
    # print(sess.run(n_aircraft))
    x2 = tf.reshape(x1, [tf.shape(x)[0], n_aircraft, x.shape[-1]], name='remove_zeros_reshape')
    # x2 = tf.reshape(x1, [tf.shape, n_aircraft, x.shape[-1]], name='remove_zeros_reshape')
    return x2

def reshape_input(x):
    return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], x.shape[3]], name="shared_reshape")


def remove_mask(x):
    return x

if __name__ == '__main__':
    tf.reset_default_graph()
    actor, _, _ = BiCNet.build_actor(80, 50, 5, 64 ,64)
    critic, _, _ = BiCNet.build_critic(80, 50, 5, 64, 64, 0.001)
    actor.summary()
    critic.summary()

