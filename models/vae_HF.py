import os

from keras.layers import Input, Dense, Lambda, Merge
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, CSVLogger

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard

from utils.myKeras import Warmup, EarlyStoppingWarmup

# === BUILD VAE ===
def build_vae_HF(config):
    print('=== VAE + HF (T={}) ==='.format(config.number_of_flows))

    def gateCalculation(args):
            h, g = args
            return h * g

    # == ENCODER ==
    x = Input(batch_shape=(config.batch_size, config.original_dim))
    encoder_pre_h = {}
    encoder_gate_h = {}
    encoder_h = {}
    h_pre_encoder = {}
    h_gate_encoder = {}
    h_encoder = {}
    h_encoder['0'] = x
    for l_e in range( len(config.encoder_arch) ):
        #calculate X
        encoder_pre_h[str(l_e+1)] = Dense(config.encoder_arch[l_e][0], activation=config.encoder_arch[l_e][1], name='encoder_pre'+str(l_e+1))
        h_pre_encoder[str(l_e+1)] = encoder_pre_h[str(l_e+1)]( h_encoder[str(l_e)] )
        #calculate G
        encoder_gate_h[str(l_e+1)] = Dense(config.encoder_arch[l_e][0], activation='sigmoid', name='encoder_gate'+str(l_e+1))
        h_gate_encoder[str(l_e+1)] = encoder_gate_h[str(l_e+1)]( h_encoder[str(l_e)] )
        #calculate H = X * G
        encoder_h[str(l_e+1)] = Lambda(gateCalculation, output_shape=(config.encoder_arch[l_e][0],), name='encoder_hidden'+str(l_e+1))
        h_encoder[str(l_e+1)] = encoder_h[str(l_e+1)]( [h_pre_encoder[str(l_e+1)], h_gate_encoder[str(l_e+1)]] )

    encoder_z_mean = Dense(config.latent_dim, name='ecnoder_mean')
    z_mean = encoder_z_mean( h_encoder[str(len(config.encoder_arch))] )
    encoder_log_var = Dense(config.latent_dim, name='encoder_log_var')
    z_log_var = encoder_log_var( h_encoder[str(len(config.encoder_arch))] )

    #sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(config.batch_size, config.latent_dim), mean=0.,
                                  std=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    #sample from the encoder
    z_0 = Lambda(sampling, output_shape=(config.latent_dim,), name='normal-sampling')([z_mean, z_log_var])

    # HOUSEHOLDER FLOW
    z_v_pre = {}
    z_v_gate = {}
    z_v = {}
    z = {}
    z['0'] = z_0

    def HF(args):
            z_v, z = args
            A = K.batch_dot(K.expand_dims(z_v, 2), K.expand_dims(z_v, 1))

            Az = K.reshape( K.batch_dot(A, K.expand_dims(z,-1) ), (-1, config.latent_dim) )

            v_norm_sq = K.sum( K.square(z_v), axis=1, keepdims=True)

            return z - 2 * Az / v_norm_sq

    #T>0
    if config.number_of_flows > 0:
        z_v['1'] = Dense(config.latent_dim, name='HF-v1')(h_encoder[str(len(config.encoder_arch))])
        z['1'] = Lambda(HF, output_shape=(config.latent_dim,), name='HF-z1')([z_v['1'], z_0])
        for i in range(1,config.number_of_flows):
            z_v[str(i+1)] = Dense(config.latent_dim, name='HF-v'+str(i+1))( z_v[str(i)] )
            z[str(i+1)] = Lambda(HF, output_shape=(config.latent_dim,), name='HF-z'+str(i+1))([z_v[str(i+1)], z[str(i)]])

    # == DECODER ==
    decoder_pre_h = {}
    decoder_gate_h = {}
    decoder_h = {}
    h_pre_decoder = {}
    h_gate_decoder = {}
    h_decoder = {}
    h_decoder['0'] = z[str(config.number_of_flows)]
    for l_d in range( len(config.decoder_arch) ):
        #calculate X
        decoder_pre_h[str(l_d+1)] = Dense(config.decoder_arch[l_d][0], activation=config.decoder_arch[l_d][1], name='decoder_pre'+str(l_d+1))
        h_pre_decoder[str(l_d+1)] = decoder_pre_h[str(l_d+1)]( h_decoder[str(l_d)] )
        #calculate G
        decoder_gate_h[str(l_d+1)] = Dense(config.decoder_arch[l_d][0], activation='sigmoid', name='decoder_gate'+str(l_d+1))
        h_gate_decoder[str(l_d+1)] = decoder_gate_h[str(l_d+1)]( h_decoder[str(l_d)] )
        #calculate H = X * G
        decoder_h[str(l_d+1)] = Lambda(gateCalculation, output_shape=(config.decoder_arch[l_d][0],), name='decoder_hidden'+str(l_d+1))
        h_decoder[str(l_d+1)] = decoder_h[str(l_d+1)]( [h_pre_decoder[str(l_d+1)], h_gate_decoder[str(l_d+1)]] )

    # mean
    if config.data_type == 'binary' or config.data_type == 'gray':
        decoder_mean = Dense(config.original_dim, activation='sigmoid', name='decoder_mean')
        x_decoded_mean = decoder_mean(h_decoder[str(len(config.decoder_arch))])
    else:
        raise ValueError
    # variance
    if config.data_type == 'gray':
        decoder_log_var = Dense(config.original_dim, name='decoder_log_var')
        x_decoded_log_var = decoder_log_var(h_decoder[str(len(config.decoder_arch))])

    # MODEL
    vae = Model(x, x_decoded_mean)
    if config.regularization == 'warmup':
            vae.beta = K.variable(0.)
    print(vae.summary())

    # LOSS FUNCTION
    def vae_loss(x, x_decoded_mean):
        # RE part
        if config.data_type == 'binary':
            re_loss = log_Bernoulli(x, x_decoded_mean)
            RE = K.mean(K.sum(re_loss, axis=1), axis=0)
        elif config.data_type == 'gray':
            re_loss = log_Normal_diag(x, x_decoded_mean, x_decoded_log_var)
            RE = K.mean(K.sum(re_loss, axis=1), axis=0)
        else:
            raise ValueError

        #KL part
        log_q = log_Normal_diag(z_0, z_mean, z_log_var)
        log_p = log_Normal_standard( z[str(config.number_of_flows)] )
        kl =  log_q - log_p

        if config.regularization == 'none':
            KL = K.mean( K.sum( kl, axis=1 ), axis=0 )
        elif config.regularization == 'warmup':
            KL = K.mean( K.sum( kl, axis=1 ), axis=0 ) * vae.beta
        else:
            raise Exception('wrong regularization name')

        return -RE + KL

    # METRICS
    def vae_re(x, x_decoded_mean):
        # RE part
        if config.data_type == 'binary':
            re_loss = log_Bernoulli(x, x_decoded_mean)
            RE = K.mean(K.sum(re_loss, axis=1), axis=0)
        elif config.data_type == 'gray':
            re_loss = log_Normal_diag(x, x_decoded_mean, x_decoded_log_var)
            RE = K.mean(K.sum(re_loss, axis=1), axis=0)
        else:
            raise ValueError
        return -RE

    def vae_kl(x, x_decoded_mean):
        #KL part
        log_q = log_Normal_diag(z_0, z_mean, z_log_var)
        log_p = log_Normal_standard( z[str(config.number_of_flows)] )
        kl_loss =  log_q - log_p
        return K.mean( K.sum( kl_loss, axis=1 ), axis=0 )

    #COMPILE
    vae.compile( optimizer=Adam(lr=config.learning_rate),
                 loss=vae_loss,
                 metrics=[vae_re, vae_kl])

    # TRAINING
    # Callbacks
    callbacks = []

    history = History()
    callbacks.append(history)

    csvlogger = CSVLogger(filename=os.path.join(config.path, 'training.log'))
    callbacks.append(csvlogger)

    if config.regularization == 'warmup':
        earlystopping = EarlyStoppingWarmup(patience=config.early_stopping_epochs, mode='min', warmup=config.regularization_param)
        callbacks.append(earlystopping)

        warmup = Warmup(config.regularization_param)
        callbacks.append(warmup)

    else:
        earlystopping = EarlyStopping(patience=config.early_stopping_epochs, mode='min')
        callbacks.append(earlystopping)

    checkpointer = ModelCheckpoint( filepath= os.path.join(config.path, 'weights_vae.hdf5'), verbose=0, save_best_only=True, save_weights_only=True )
    callbacks.append(checkpointer)

    return vae, callbacks
