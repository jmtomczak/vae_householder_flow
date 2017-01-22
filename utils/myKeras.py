import numpy as np

import warnings

import keras.backend as K
from keras.callbacks import Callback

class Warmup(Callback):
    '''Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    '''
    def __init__(self, max_epochs):
        super(Warmup, self).__init__()
        if max_epochs <= 0:
            self.max_epochs = 1.
        else:
            self.max_epochs = max_epochs

    def on_epoch_begin(self, epoch, logs={}):
        # assert hasattr(self.model, 'beta'), \
        #     'Optimizer must have a "beta" attribute.'

        beta = np.minimum(1., (epoch*1.) / (self.max_epochs*1.) )

        if not isinstance(beta, (float, np.float32, np.float64)):
            raise ValueError('The output of the "beta" param'
                             'should be float.')

        K.set_value( self.model.beta, beta)

        print('Beta (warm-up with {} epochs): {:.3f}'.format( self.max_epochs, beta ) )

# === EARLY STOPPING FOR WARMUP - skipping epochs with warm-up ===
class EarlyStoppingWarmup(Callback):
    '''Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    '''
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, warmup=0, mode='auto'):
        super(EarlyStoppingWarmup, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        self.warmup = warmup

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs={}):
        self.wait = 0       # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)
        # include warmup
        if epoch > self.warmup:
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                self.wait += 1
        else:
            self.best = current
            self.wait = 0

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))
