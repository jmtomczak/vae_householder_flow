import numpy as np
from keras import backend as K

epsilon = 1e-7

#### DISTRIBUTIONS
def log_Normal_diag(sample, mean, log_var):
    # return -0.5 * ( np.log( 2. * np.pi ) + log_var + K.square( sample - mean ) / K.exp( log_var ) )
    return -0.5 * ( log_var + K.square( sample - mean ) / K.exp( log_var ) )

def log_Normal_standard(sample):
    # return -0.5 * ( np.log( 2. * np.pi ) + K.square( sample ) )
    return -0.5 * ( K.square( sample ) )

def log_Bernoulli( sample, probs ):
    probs =  K.clip(probs, epsilon, 1.-epsilon)
    return sample * K.log(probs) + (1. - sample) * K.log( 1. - probs )