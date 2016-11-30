import tensorflow as tf

class Config(object):
    # ----------experiment----------
    random_seed = 111

    # ----------model----------
    classifier = False

    transfer_fct = 'tanh'

    network_architecture = dict(
        encoder=[300, 300],  # encoder neurons
        decoder=[300, 300],  # decoder neurons
        n_input=784,  # MNIST data input (img shape: 28*28)
        n_z=40)  # dimensionality of latent space

    number_of_Householders = 1

    # ----------data----------
    datasetName = 'mnist'

    if datasetName == 'mnist':
        generative_distribution = 'Bernoulli'
        network_architecture['n_input'] = 28 * 28
    elif datasetName == 'histopathology':
        generative_distribution = 'NormalGray'
        network_architecture['n_input'] = 28 * 28
    else:
        raise ValueError

    # ----------optimization----------
    optimizer = 'Adam' #'Adam', 'SGD', 'Adadelta', 'NAG', 'Momentum'
    learning_rate = 0.0002
    momentum = 0.0
    annealing_rate = 0.

    batch_size = 100

    max_epoch = 1500
    min_epoch = 0
    early_stopping = 100

    # ----------regularization----------
    regularization = 'none' # 'none', 'freebits', 'warmup'

    warm_up = 0
    lambda_3B = 0.25 #value of lamba in "free bits" constraint

def get_config():
    return Config()
