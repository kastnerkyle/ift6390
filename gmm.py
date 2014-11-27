import numpy as np
from data import load_faithful
from scipy import linalg
import matplotlib.pyplot as plt


def plot_covariance(mean, cov, color="darkred", subplot_ref=None):
    """
    Plot a 2D gaussian with given mean and covariance.
    Based on code from Roland Memisevic.
    """
    t = np.arange(-np.pi, np.pi, 0.01)
    x = np.sin(t)[:, None]
    y = np.cos(t)[:, None]

    D, V = linalg.eigh(cov)
    A = np.real(np.dot(V, np.diag(np.sqrt(D))).T)
    z = np.dot(np.hstack([x, y]), A)

    if subplot_ref is not None:
        p = subplot_ref
    else:
        p = plt
    p.plot(z[:, 0] + mean[:, 0], z[:, 1] + mean[:, 1], linewidth=2, color=color)


def lognorm_pdf(X, means, covars, min_covar=1.e-7):
    """
    Log probability for full covariance matrices.
    Modified from scikit-learn.
    """
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        cv_chol = linalg.cholesky(cv, lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = -.5 * (np.sum(cv_sol ** 2, axis=1) +
                                n_dim * np.log(2 * np.pi) + cv_log_det)
    return log_prob


def expectation_step(X, means, covs, weights):
    likelihood = np.exp(lognorm_pdf(X, means, covs))
    responsibilities = weights * likelihood
    responsibilities /= np.sum(responsibilities, axis=1)[:, None]
    return likelihood, responsibilities


def maximization_step(X, means, covs, weights, responsibilities):
    new_weights = np.mean(responsibilities, axis=0)
    new_means = means
    new_covs = covs
    for i in range(len(means)):
        new_means[i] = responsibilities[:, i][None].dot(X) / np.sum(
            responsibilities[:, i])
        X_c = X - means[i]
        new_cov = (responsibilities[:, i] * X_c.T).dot(X_c)
        new_cov /= np.sum(responsibilities[:, i])
        new_covs[i] = new_cov
    return new_means, new_covs, new_weights


random_state = np.random.RandomState(1999)
X = load_faithful()
n_components_list = [2, 3, 5]
f, axarr = plt.subplots(1, len(n_components_list))
for n, n_components in enumerate(n_components_list):
    means = []
    covs = []
    weights = np.ones((n_components)) / float(n_components)
    n_iter = 1000

    loglikelihood = lognorm_pdf(X, [np.array([0, 0])] * n_components,
                                     [np.identity(X.shape[1])] * n_components)
    init_ll = loglikelihood.sum()

    for i in range(n_components):
        mean = X[random_state.randint(0, len(X))]
        means.append(mean)
        rs = random_state.rand(X.shape[1], X.shape[1])
        cov = np.eye(X.shape[1])
        cov += random_state.rand(*cov.shape)
        covs.append(cov)

    for i in range(n_iter):
        likelihood, responsibilities = expectation_step(X, means, covs, weights)
        means, covs, weights = maximization_step(X, means, covs, weights, responsibilities)

    loglikelihood = lognorm_pdf(X, means, covs)
    total_ll = loglikelihood.sum()
    axarr[n].set_title("n_components %i\n init log-L %4f\n final log-L %4f" % (
        n_components, init_ll, total_ll))

    axarr[n].scatter(X[:, 0], X[:, 1], c="steelblue", alpha=0.4)
    for i in range(len(means)):
        axarr[n].plot(means[i][:, 0], means[i][:, 1], color="darkred")
        plot_covariance(means[i], covs[i], color="darkred",
                        subplot_ref=axarr[n])
plt.show()
