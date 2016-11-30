
import os
import tensorflow.examples.tutorials.mnist.input_data as mnist_io
import numpy as np
import tensorflow as tf
from mnist_data import extract_images
from mnist_data import extract_labels
from mnist_data import DataSet
from tensorflow.contrib.learn.python.learn.datasets.base import maybe_download
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets

np.random.seed(0)
tf.set_random_seed(0)

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
VALIDATION_SIZE = 10000


def openMnist(path = 'dataset/mnist', small = False):

    if not os.path.exists(path):
        os.makedirs(path)
    mnist_io.read_data_sets(path)

    one_hot=True
    reshape=True

    local_file = maybe_download(TRAIN_IMAGES, path, SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = maybe_download(TRAIN_LABELS, path, SOURCE_URL + TRAIN_LABELS)
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = maybe_download(TEST_IMAGES, path, SOURCE_URL + TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = maybe_download(TEST_LABELS, path, SOURCE_URL + TEST_LABELS)
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    if small:
        train_images = train_images[0:10000]
        train_labels = train_labels[0:10000]

    train = DataSet(train_images, train_labels, dtype=tf.float32, reshape=reshape, binarize=True)
    validation = DataSet(validation_images,validation_labels, dtype=tf.float32, reshape=reshape, binarize=True)
    test = DataSet(test_images, test_labels, dtype=tf.float32, reshape=reshape, binarize=True)

    return Datasets(train=train, validation=validation, test=test)


if __name__ == '__main__':
    data = openMnist()
