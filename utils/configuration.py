class Config():
    # == CONFIGURATION ==
    batch_size = 100
    original_dim = 784
    latent_dim = 40
    encoder_arch= [ [300, None], [300, None]]
    decoder_arch = [ [300, None], [300, None]]
    number_epochs = 5000
    epsilon_std = 1.0
    early_stopping_epochs = 100
    learning_rate = 0.0002

    kl_sample = True

    regularization = 'none'

    if regularization == 'none':
        regularization_param = 0
    elif regularization == 'warmup':
        regularization_param = 200
    else:
        raise Exception('Wrong name of regularizer!')

    dataset_name = 'mnistDynamic' #'histopathology'

    data_type = 'binary' #'gray'

    model = 'VAE'

    number_of_flows = 1