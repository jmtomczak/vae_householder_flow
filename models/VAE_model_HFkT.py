import tensorflow as tf
import numpy as np
from VAE_model import VAE


epsilon = 1e-8

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class VAE_model_HFkT(VAE):

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights()

        # Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
        self.z_mean, self.z_log_sigma_sq, self.z_v = self._recognition_network(network_weights['weights_recog'],
                                                                               network_weights['biases_recog'])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture['n_z']
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)

        # T = 0: vanilla VAE
        self.z = {}
        self.z['0'] = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # T = 1,...,K: Householder flow
        for k in range(1,self.number_of_Householders+1):
            v = tf.nn.l2_normalize(self.z_v[str(k)], dim=[1])
            v_1 = tf.tile(tf.expand_dims(v, 2), [1, 1, self.network_architecture['n_z']])
            v_2 = tf.tile(tf.expand_dims(v, 1), [1, self.network_architecture['n_z'], 1])
            self.z[str(k)] = self.z[str(k-1)] - tf.reduce_sum( 2 * (v_1 * v_2) * tf.expand_dims(self.z[str(k-1)], -1), 2)

        # Use generator to determine mean of Bernoulli distribution of reconstructed input
        self.x_reconstr_mean, self.x_reconstr_log_var = self._generator_network(network_weights['weights_gener'], network_weights['biases_gener'])

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
        for k in range(1, self.number_of_Householders+1):
            tmp_dict['out_v'+str(k)] = tf.Variable(xavier_init(s_previous, self.network_architecture['n_z']))
        all_weights['weights_recog'] = tmp_dict

        # encoder bias
        tmp_dict = dict()
        tmp_array = list()
        for s_current in self.network_architecture['encoder']:
            tmp_array.append(tf.Variable(tf.zeros([s_current], dtype=tf.float32)))
        tmp_dict['weights'] = tmp_array
        tmp_dict['out_mean'] = tf.Variable(tf.zeros([self.network_architecture['n_z']], dtype=tf.float32))
        tmp_dict['out_log_sigma'] = tf.Variable(tf.zeros([self.network_architecture['n_z']], dtype=tf.float32))
        for k in range(1, self.number_of_Householders+1):
            tmp_dict['out_v'+str(k)] = tf.Variable(tf.zeros([self.network_architecture['n_z']], dtype=tf.float32))
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
        if self.generative_distribution == 'Normal' or self.generative_distribution == 'NormalGray':
            tmp_dict['out_log_var'] = tf.Variable(xavier_init(s_previous, self.network_architecture['n_input']))
        all_weights['weights_gener'] = tmp_dict

        # decoder bias
        tmp_dict = dict()
        tmp_array = list()
        for s_current in self.network_architecture['encoder']:
            tmp_array.append(tf.Variable(tf.zeros([s_current], dtype=tf.float32)))
        tmp_dict['weights'] = tmp_array
        tmp_dict['out_mean'] = tf.Variable(tf.zeros([self.network_architecture['n_input']], dtype=tf.float32))
        if self.generative_distribution == 'Normal' or self.generative_distribution == 'NormalGray':
            tmp_dict['out_log_var'] = tf.Variable(tf.zeros([self.network_architecture['n_input']], dtype=tf.float32))
        all_weights['biases_gener'] = tmp_dict

        return all_weights

    def _recognition_network(self, weights, biases):
        state = self.x
        for (w, b) in zip(weights['weights'], biases['weights']):
            state = self.transfer_fct(tf.add(tf.matmul(state, w), b))

        z_mean = tf.add(tf.matmul(state, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(state, weights['out_log_sigma']), biases['out_log_sigma'])
        z_v = {}
        for k in range(1,self.number_of_Householders+1):
            z_v[str(k)] = tf.add(tf.matmul(state, weights['out_v'+str(k)]), biases['out_v'+str(k)])

        return (z_mean, z_log_sigma_sq, z_v)

    def _generator_network(self, weights, biases):
        state = self.z[str(self.number_of_Householders)]
        for (w,b) in zip(weights['weights'], biases['weights']):
            state = self.transfer_fct(tf.add(tf.matmul(state, w), b))

        # mean value
        if self.generative_distribution == 'Normal':
            x_reconstr_mean = tf.add(tf.matmul(state, weights['out_mean']), biases['out_mean'])
        else:
            x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(state, weights['out_mean']), biases['out_mean']))

        # log-variance value
        if self.generative_distribution == 'Bernoulli':
            x_reconstr_log_var = 0.
        else:
            x_reconstr_log_var = tf.add(tf.matmul(state, weights['out_log_var']), biases['out_log_var'])

        return x_reconstr_mean, x_reconstr_log_var