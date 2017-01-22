import numpy as np
import pickle

import urllib
import os

from keras.datasets import mnist


def load_dataset(name):

    if name == 'mnistGray':
        print('--GRAY MNIST DATASET--')
        (x_train, y_train), (x_test, y_test) = mnist.load_data( )

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        # shuffling
        np.random.shuffle(x_train)
        x_val = x_train[50000:60000]
        y_val = y_train[50000:60000]
        x_train = x_train[0:50000]
        y_train = y_train[0:50000]

    elif name == 'mnistDynamic':
        print('--DYNAMIC BINARIZATION MNIST DATASET--')
        (x_train, y_train), (x_test, y_test) = mnist.load_data( )

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        # shuffling
        np.random.shuffle(x_train)
        x_val = x_train[50000:60000]
        y_val = y_train[50000:60000]
        x_train = x_train[0:50000]
        y_train = y_train[0:50000]

        #binarization of validatiaon and test set
        x_val = np.random.binomial(1,x_val)
        x_test = np.random.binomial(1,x_test)

    elif name == 'mnistBinary':
        print('--BINARY MNIST DATASET--')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        # binarization
        x_train = np.random.binomial(1,x_train)
        x_test = np.random.binomial(1,x_test)
        # shuffling
        np.random.shuffle(x_train)
        x_val = x_train[50000:60000]
        y_val = y_train[50000:60000]
        x_train = x_train[0:50000]
        y_train = y_train[0:50000]

    elif name == 'histopathology':
        print('--GRAY HISTOPATHOLOGY DATASET--')
        with open('datasets/histopathologyGray/histopathology.pkl', 'rb') as f:
            data = pickle.load(f)

        x_train = np.asarray(data['training']).reshape(-1,28*28)
        x_val = np.asarray(data['validation']).reshape(-1,28*28)
        x_test = np.asarray(data['test']).reshape(-1,28*28)

        # IDLE LABELS just to fit the framework
        y_train = 0.
        y_val = 0.
        y_test = 0.
    else:
        raise Exception('Wrong dataset name!')

    return x_train, x_val, x_test, y_train, y_val, y_test

if __name__ == '__main__':
    load_dataset('mnistDynamic')
