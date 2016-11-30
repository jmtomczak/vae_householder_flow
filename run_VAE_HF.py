import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pickle
import random
import pprint
from utils.VAE_config import get_config
from utils.VAE_train import train
from models.VAE_model_HFkT import VAE_model_HFkT
from utils.VAE_commons import checkOrMake
from utils.archive_results import save_results
import tensorflow as tf

SNAPSHOT_PATH = 'snapshots/'

def arch2str(s):
    import numpy as np
    s = np.core.defchararray.replace(s, '[', '')
    s = np.core.defchararray.replace(s, ']', '')
    s = np.core.defchararray.replace(s, ' ', '')
    return str(s)

def run(regularization, regularization_param, dataset, repeat, mini_batch_size = 100, max_epochs = 1500, number_of_Householders = 0, encoder=[300,300], decoder=[300,300], number_z = 40):
    # RESET
    tf.reset_default_graph()

    # re-set configuration
    configuration = get_config()
    configuration.regularization = regularization
    if regularization is 'warmup':
        configuration.warm_up = regularization_param
    elif regularization is 'freebits':
        configuration.lambda_3B = regularization_param
    else:
        raise ValueError
    configuration.number_of_Householders = number_of_Householders

    configuration.random_seed = repeat

    configuration.datasetName = dataset

    configuration.batch_size = mini_batch_size

    configuration.max_epoch = max_epochs

    configuration.network_architecture['n_z'] = number_z
    configuration.network_architecture['encoder'] = encoder
    configuration.network_architecture['decoder'] = decoder

    if dataset == 'mnist':
        configuration.generative_distribution = 'Bernoulli'
        configuration.network_architecture['n_input'] = 28 * 28
    elif dataset == 'histopathology':
        configuration.generative_distribution = 'NormalGray'
        configuration.network_architecture['n_input'] = 28 * 28
    else:
        raise Exception('Wrong name of dataset!')

    model_id = configuration.datasetName + \
               '|HF'+str(configuration.number_of_Householders)+'T|' + \
               str(repeat) + '|'+\
               regularization+'|'+\
               'param='+str(regularization_param) + '|' +\
               str(configuration.learning_rate) + '|' + \
               'enc:' + arch2str(str(configuration.network_architecture['encoder'])) + '|' + \
               'dec:' + arch2str(str(configuration.network_architecture['decoder'])) + '|' + \
               str(configuration.network_architecture['n_z']) + '|' + \
               configuration.transfer_fct

    if checkOrMake(SNAPSHOT_PATH, model_id) or True:
        print(model_id)
        dir_name = SNAPSHOT_PATH + '/' + model_id
        checkpoint_name = 'model'

        pickle.dump(configuration, open( dir_name + '/' + 'config.bin', mode='wb') )

        final_result = train(configuration, os.path.join(dir_name, checkpoint_name), VAE_model_HFkT)

        save_results(model_id + final_result)

if __name__ == '__main__':
    run(999,'mnist')
