from data import load_mnist
from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np

train, test = load_mnist()
X_train, y_train = train
X_test, y_test = test

X_train = X_train / 255.
X_test = X_test / 255.
mean = np.mean(X_train, axis=0)
X_train = X_train - mean
X_test = X_test - mean

n_components = 2
U, S, V = svd(X_train, full_matrices=False)
basis = V[:n_components].T
test_classes = np.argmax(y_test, axis=1)
f, axarr = plt.subplots(5, 2)
for i in np.unique(test_classes):
    test_examples = np.where(test_classes == i)[0]
    X_proj = X_test[test_examples].dot(basis)
    axarr.ravel()[i].scatter(X_proj[:, 0], X_proj[:, 1], color="steelblue")
    axarr.ravel()[i].set_title("Class %i" % i)
    axarr.ravel()[i].set_xlim([-.025, .025])
    axarr.ravel()[i].set_ylim([-.025, .025])
plt.tight_layout()
plt.show()
