import torch
import numpy as np
from sklearn.neighbors import KernelDensity


class TPESelector:

    def __init__(self, gamma=0.25, bandwidth=0.1, maximize=True):
        """
        gamma: fraction of "good" observations
        bandwidth: KDE bandwidth
        maximize: if True, assumes we want to maximize y; if False, minimize y
        """
        self.gamma = gamma
        self.bandwidth = bandwidth
        self.maximize = maximize
        self.history_x = []  # list of molecules/features
        self.history_y = []  # list of target values
        self.kde_good = None
        self.kde_bad = None

    def condition_on_observations(self, x, y):
        """
        Add new observation
        x: feature vector, shape [D] or 1D array
        y: scalar target
        """
        self.history_x.append(np.array(x))
        self.history_y.append(y)
        self._fit_kdes()

    def _fit_kdes(self):
        """
        Fit KDE for good and bad points
        """
        if len(self.history_y) < 2:
            return  # not enough data to fit

        y = np.array(self.history_y)
        x = np.array(self.history_x)

        topk = max(1, int(len(y) * self.gamma))

        # Sort based on optimization direction
        if self.maximize:
            # For maximization: good = highest values
            idx_sorted = np.argsort(y)[::-1]  # descending order
        else:
            # For minimization: good = lowest values
            idx_sorted = np.argsort(y)  # ascending order

        good_x = x[idx_sorted[:topk]]
        bad_x = x[idx_sorted[topk:]]

        # Fit KDE
        self.kde_good = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(good_x)
        if len(bad_x) > 0:
            self.kde_bad = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(bad_x)
        else:
            self.kde_bad = None

    def suggest(self, candidates):
        """
        candidates: array-like [N_cand, D]
        Returns index of chosen candidate
        """
        if self.kde_good is None:
            # Not enough history, random pick
            return np.random.randint(len(candidates))

        candidates = np.array(candidates)
        log_likelihood_good = self.kde_good.score_samples(candidates)
        if self.kde_bad is not None:
            log_likelihood_bad = self.kde_bad.score_samples(candidates)
        else:
            log_likelihood_bad = np.zeros_like(log_likelihood_good)

        # TPE acquisition: maximize l(x)/g(x) -> log space
        score = log_likelihood_good - log_likelihood_bad
        prob = np.exp(score - score.max())  # for numerical stability
        prob = prob / prob.sum()

        choice = np.random.choice(len(candidates), p=prob)
        return choice
