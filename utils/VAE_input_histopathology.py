
import numpy as np
import tensorflow as tf
import urllib
import os
from scipy.io import loadmat
import gzip
import pickle as pkl
# from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from histopathology_data import DataSet
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets

def openHistopathology(path = 'dataset/histopathology', random_seed=111):

    np.random.seed(random_seed)

    reshape=True

    with open(os.path.join(path, 'histopathology.pkl'), 'rb') as f:
        data = pkl.load(f)

    # train images
    train_images = np.asarray(data['training']).reshape(-1,28,28,1)
    # validation images
    validation_images = np.asarray(data['validation']).reshape(-1,28,28,1)
    # test images
    test_images = np.asarray(data['test']).reshape(-1,28,28,1)

    #artificial labels that make no sense but these are created to fit the framework
    N_train = np.shape(train_images)
    N_val = np.shape(validation_images)
    N_test = np.shape(test_images)

    train_labels = np.ones(shape=(N_train[0],1), dtype=np.int32)
    validation_labels = np.ones(shape=(N_val[0],1), dtype=np.int32)
    test_labels = np.ones(shape=(N_test[0],1), dtype=np.int32)

    train = DataSet(train_images, train_labels, dtype=tf.float32, reshape=reshape)
    validation = DataSet(validation_images,validation_labels, dtype=tf.float32, reshape=reshape)
    test = DataSet(test_images, test_labels, dtype=tf.float32, reshape=reshape)

    return Datasets(train=train, validation=validation, test=test)

if __name__ == '__main__':
    data = openHistopathology28x28()