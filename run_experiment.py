import numpy as np
import pickle

import os

from models.vae_HF import build_vae_HF

from utils.configuration import Config
from utils.load_data import load_dataset
from utils.commons import arch2str

from utils.save_results import save_results

# == CONFIGURATION ==
config = Config()

experiment_repetitions = [1,2,3]

# == DATASET ==
x_train, x_val, x_test, y_train, y_val, y_test = load_dataset( config.dataset_name )

# == EXPERIMENT ==
config.model = config.model + '_T=' + str(config.number_of_flows)
for r in experiment_repetitions:
    config.model_name = config.model + '|' + arch2str(str(config.encoder_arch)) + '|' + arch2str(
                        str(config.decoder_arch))+ '|' +'latent_dim_' + str(config.latent_dim) + '|' + config.regularization + '_' + str(
                        config.regularization_param) + '|' + 'learning_rate_' + str(config.learning_rate)

    config.path = 'snapshots' + '/' + config.model_name + '|rep_' + str(r)
    if not os.path.exists(config.path):
        os.makedirs(config.path)

    # == MODEL + LOSS + OPTIMIZER + TRAINING DETAILS ==
    vae, callbacks = build_vae_HF( config )

    config.model_name = config.model_name + '|num_params_' + str(vae.count_params()) + '|'

    # == TRAINING ==
    if config.dataset_name == 'mnistDynamic':
        print('***training with dynamic binarization***')

        def mnistDynamicBinarization():
            number_of_batches = int(np.shape(x_train)[0] / config.batch_size)
            while 1:
                X = np.random.binomial( 1, x_train )
                np.random.shuffle(X)
                for index in range(number_of_batches):
                    x_b = X[ index * config.batch_size : (index+1) * config.batch_size, : ]
                    yield x_b, x_b

        my_gen = mnistDynamicBinarization()
        vae.fit_generator( generator=my_gen,
                       samples_per_epoch=np.shape(x_train)[0],
                       nb_epoch=config.number_epochs,
                       validation_data=(x_val, x_val),
                       callbacks=callbacks )
    else:
        vae.fit(x_train, x_train,
                shuffle=True,
                nb_epoch=config.number_epochs,
                batch_size=config.batch_size,
                validation_data=(x_val, x_val),
                callbacks=callbacks)

    # == EVALUATION ==
    test_results = vae.evaluate(x_test, x_test,
                                 batch_size=config.batch_size,
                                 verbose=1)

    test_elbo = np.asarray(test_results[1]) + np.asarray(test_results[2]) # ELBO = RE + KL

    print('\nFinal test ELBO: {:.2f}\n * RE: {:.2f},\n * KL: {:.2f}\n'.format( test_elbo, test_results[1], test_results[2] ) )

    save_results( config.model_name + '||' + 'TEST ELBO ' + str(test_elbo) + ' RE ' + str(test_results[1]) + ' KL ' + str(test_results[2]) + ' || epochs: ' + str(len(callbacks[0].history['loss'])) )

    # == HISTORY ==
    with open( os.path.join(config.path, 'history.pkl'), 'wb') as f:
        pickle.dump(callbacks[0].history, f)

    # == Dump config ==
    with open(os.path.join(config.path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)