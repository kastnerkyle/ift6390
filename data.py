import numpy as np


def load_mnist():
    X_train = np.loadtxt('data/train_images.txt', delimiter=',')
    X_test = np.loadtxt('data/test_images.txt', delimiter=',')
    y_train = np.loadtxt('data/train_labels.txt', delimiter=',')
    y_test = np.loadtxt('data/test_labels.txt', delimiter=',')
    return [(X_train, y_train), (X_test, y_test)]


def load_faithful():
    return np.loadtxt('data/faithful.txt', delimiter=' ')
