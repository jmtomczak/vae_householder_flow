import tensorflow as tf
import numpy as np


epsilon = 1e-8

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class VAE(object):
    def __init__(self, configuration):

        #model definition
        self.network_architecture = configuration.network_architecture
        if configuration.transfer_fct == 'softplus':
            self.transfer_fct = tf.nn.softplus
        elif configuration.transfer_fct == 'tanh':
            self.transfer_fct = tf.nn.tanh
        elif configuration.transfer_fct == 'relu':
            self.transfer_fct = tf.nn.relu

        self.number_of_Householders = configuration.number_of_Householders

        self.generative_distribution = configuration.generative_distribution

        #optimizer
        self.optimizer = configuration.optimizer
        self.learning_rate = configuration.learning_rate
        self.momentum = configuration.momentum
        self.annealing_rate = configuration.annealing_rate
        self.batch_size = configuration.batch_size

        #regularizer
        self.regularization = configuration.regularization
        self.lambda_3B = configuration.lambda_3B
        self.customVariables = {}
        self.customVariables['beta_warm_up'] = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.customVariables['step'] = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        #input
        self.x = tf.placeholder(tf.float32, [None, self.network_architecture['n_input']])

        #model creation
        self._create_network()
        self._create_loss_optimizer()

        self.partials = ['rec','kl']

    def _create_network(self):

        self.z_mean = 0.
        self.z_log_sigma_sq = 0.

        self.x_reconstr_mean = 0.
        self.x_reconstr_log_var = 0.

    def _create_loss_optimizer(self):
       #---EVALUATION---
        if self.generative_distribution == 'Bernoulli':
            reconstr_loss_eval = -tf.reduce_sum( self.x * tf.log(tf.clip_by_value(self.x_reconstr_mean, epsilon, 1.0 - epsilon)) +
                (1 - self.x) * tf.log(tf.clip_by_value(1.0 - self.x_reconstr_mean, epsilon, 1.0 - epsilon)), 1)

        elif self.generative_distribution == 'NormalGray' or self.generative_distribution == 'Normal':
            reconstr_loss_eval = -tf.reduce_sum( -0.5 * np.log(2 * np.pi) - 0.5 * 1. / tf.exp(self.x_reconstr_log_var) * tf.square(
                self.x - self.x_reconstr_mean) - 0.5 * self.x_reconstr_log_var, 1)
        else:
            raise Exception('Wrong generative distribution!')

        kl_loss_eval = -0.5 * tf.reduce_sum(1. + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

        self.partial_costs_eval = [tf.reduce_mean(reconstr_loss_eval), tf.reduce_mean(kl_loss_eval)]
        self.cost_eval = tf.reduce_mean(reconstr_loss_eval + kl_loss_eval)

        #---OBJECTIVE---
        if self.generative_distribution == 'Bernoulli':
            reconstr_loss = -tf.reduce_mean(
                self.x * tf.log(tf.clip_by_value(self.x_reconstr_mean, epsilon, 1.0 - epsilon)) +
                (1 - self.x) * tf.log(tf.clip_by_value(1.0 - self.x_reconstr_mean, epsilon, 1.0 - epsilon)), 0)
        elif self.generative_distribution == 'NormalGray' or self.generative_distribution == 'Normal':
            reconstr_loss = -tf.reduce_mean( -0.5 * np.log(2 * np.pi) -0.5 * 1. / tf.exp(self.x_reconstr_log_var) * tf.square(
                self.x - self.x_reconstr_mean) - 0.5 * self.x_reconstr_log_var, 0)
        else:
            raise Exception('Wrong generative distribution!')

        if self.regularization == 'none':
            kl_loss = -0.5 * tf.reduce_mean(1. + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 0)

        elif self.regularization == 'warmup':
            kl_loss = -0.5 * tf.reduce_mean(
                1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 0) * self.customVariables['beta_warm_up']

        elif self.regularization == 'freebits':
            kl_loss = tf.maximum(self.lambda_3B,
                                 -0.5 * tf.reduce_mean(
                                     1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 0)
                                 )
        else:
            raise Exception('Wrong name of the regularization!')

        self.partial_costs = [tf.reduce_sum(reconstr_loss), tf.reduce_sum(kl_loss)]
        self.cost = tf.reduce_sum(reconstr_loss) + tf.reduce_sum(kl_loss)

        #---OPTIMIZER---
        annealing_learning_rate = self.learning_rate / tf.sqrt(1. + self.annealing_rate * self.customVariables['step'])

        if self.optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=annealing_learning_rate).minimize(self.cost)
        elif self.optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=annealing_learning_rate).minimize(self.cost)
        elif self.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=annealing_learning_rate).minimize(self.cost)
        elif self.optimizer == 'Momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=annealing_learning_rate, momentum=self.momentum, use_nesterov=False).minimize(self.cost)
        elif self.optimizer == 'NAG':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=annealing_learning_rate, momentum=self.momentum, use_nesterov=True).minimize(self.cost)
        else:
            raise Exception('Wrong name of the optimizer!')

    def createFeed(self, batch_xs, batch_ys):
        return {self.x: batch_xs}