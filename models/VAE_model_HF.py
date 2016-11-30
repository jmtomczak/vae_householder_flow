import tensorflow as tf
import numpy as np


epsilon = 1e-8

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class VAE_model_HF(object):
    def __init__(self, configuration):

        self.customVariables = {}
        self.customVariables['beta_warm_up'] = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.network_architecture = configuration.network_architecture
        if configuration.transfer_fct == 'softplus':
            self.transfer_fct = tf.nn.softplus
        elif configuration.transfer_fct == 'tanh':
            self.transfer_fct = tf.nn.tanh
        elif configuration.transfer_fct == 'relu':
            self.transfer_fct = tf.nn.relu
        self.learning_rate = configuration.learning_rate
        self.batch_size = configuration.batch_size

        self.x = tf.placeholder(tf.float32, [None, self.network_architecture['n_input']])
        self._create_network()
        self._create_loss_optimizer()

        self.partials = ['rec','kl']

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights()

        # Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
        self.z_mean, self.z_log_sigma_sq, self.z_v = self._recognition_network(network_weights['weights_recog'],
                                                                               network_weights['biases_recog'])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture['n_z']
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)
        z_0 = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Householder flow
        v_norm = tf.nn.l2_normalize(self.z_v, dim=[1])

        v1 = tf.expand_dims(v_norm, 2)
        v1 = tf.tile(v1, [1, 1, self.network_architecture['n_z']])

        v2 = tf.expand_dims(v_norm, 1)
        v2 = tf.tile(v2, [1, self.network_architecture['n_z'], 1])

        self.z_1 = z_0 - tf.reduce_sum(2 * (v1 * v2) * tf.expand_dims(z_0, -1), 2)

        # Use generator to determine mean of Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = self._generator_network(network_weights['weights_gener'], network_weights['biases_gener'])

    def _initialize_weights(self):

        all_weights = dict()

        # encoder weights
        tmp_dict = dict()
        tmp_array = list()
        s_previous = self.network_architecture['n_input']
        for s_current in self.network_architecture['encoder']:
            tmp_array.append(tf.Variable(xavier_init(s_previous, s_current)))
            s_previous = s_current
        tmp_dict['weights'] = tmp_array
        tmp_dict['out_mean'] = tf.Variable(xavier_init(s_previous, self.network_architecture['n_z']))
        tmp_dict['out_log_sigma'] = tf.Variable(xavier_init(s_previous, self.network_architecture['n_z']))
        tmp_dict['out_v'] = tf.Variable(xavier_init(s_previous, self.network_architecture['n_z']))
        all_weights['weights_recog'] = tmp_dict

        # encoder bias
        tmp_dict = dict()
        tmp_array = list()
        for s_current in self.network_architecture['encoder']:
            tmp_array.append(tf.Variable(tf.zeros([s_current], dtype=tf.float32)))
        tmp_dict['weights'] = tmp_array
        tmp_dict['out_mean'] = tf.Variable(tf.zeros([self.network_architecture['n_z']], dtype=tf.float32))
        tmp_dict['out_log_sigma'] = tf.Variable(tf.zeros([self.network_architecture['n_z']], dtype=tf.float32))
        tmp_dict['out_v'] = tf.Variable(tf.zeros([self.network_architecture['n_z']], dtype=tf.float32))
        all_weights['biases_recog'] = tmp_dict

        # decoder weights
        tmp_dict = dict()
        tmp_array = list()
        s_previous = self.network_architecture['n_z']
        for s_current in self.network_architecture['encoder']:
            tmp_array.append(tf.Variable(xavier_init(s_previous, s_current)))
            s_previous = s_current
        tmp_dict['weights'] = tmp_array
        tmp_dict['out_mean'] = tf.Variable(xavier_init(s_previous, self.network_architecture['n_input']))
        all_weights['weights_gener'] = tmp_dict

        # decoder bias
        tmp_dict = dict()
        tmp_array = list()
        for s_current in self.network_architecture['encoder']:
            tmp_array.append(tf.Variable(tf.zeros([s_current], dtype=tf.float32)))
        tmp_dict['weights'] = tmp_array
        tmp_dict['out_mean'] = tf.Variable(tf.zeros([self.network_architecture['n_input']], dtype=tf.float32))
        all_weights['biases_gener'] = tmp_dict

        return all_weights

    def _recognition_network(self, weights, biases):
        state = self.x
        for (w, b) in zip(weights['weights'], biases['weights']):
            state = self.transfer_fct(tf.add(tf.matmul(state, w), b))

        z_mean = tf.add(tf.matmul(state, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(state, weights['out_log_sigma']), biases['out_log_sigma'])
        z_v = tf.add(tf.matmul(state, weights['out_v']), biases['out_v'])
        return (z_mean, z_log_sigma_sq, z_v)

    def _generator_network(self, weights, biases):
        state = self.z_1
        for (w,b) in zip(weights['weights'], biases['weights']):
            state = self.transfer_fct(tf.add(tf.matmul(state, w), b))

        x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(state, weights['out_mean']), biases['out_mean']))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        reconstr_loss = -tf.reduce_sum(
                                self.x * tf.log(tf.clip_by_value(self.x_reconstr_mean, epsilon, 1.0-epsilon)) +
                                (1-self.x) * tf.log(tf.clip_by_value(1.0 - self.x_reconstr_mean, epsilon, 1.0-epsilon)), 1)
        kl_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1) * self.customVariables['beta_warm_up']


        self.partial_costs = [ tf.reduce_mean(reconstr_loss), tf.reduce_mean(kl_loss) ]
        self.cost = tf.reduce_mean(reconstr_loss + kl_loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def createFeed(self, batch_xs, batch_ys):
        return {self.x: batch_xs}
