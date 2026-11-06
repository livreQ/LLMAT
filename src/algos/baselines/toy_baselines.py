#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Baseline BO algorithms for toy benchmarks (GP, Laplace)
These baselines work directly on raw features without LLMs.
"""

import torch
import numpy as np
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
import torch.nn as nn

from src.bayesopt.gp_baselines import MLLGP

# Try to import Laplace, but make it optional
try:
    from src.bayesopt.laplace_botorch import LaplaceBoTorch
    LAPLACE_AVAILABLE = True
except ImportError:
    LAPLACE_AVAILABLE = False
    print("Warning: Laplace not available. Install with: pip install laplace-torch")


class GPBaseline:
    """Gaussian Process baseline with RBF kernel for toy benchmarks"""

    def __init__(self, dim, device='cpu', lr=0.01, n_epochs=100):
        self.dim = dim
        self.device = device
        self.lr = lr
        self.n_epochs = n_epochs
        self.model = None

    def fit(self, X, y):
        """
        Fit GP model to data

        Args:
            X: torch.Tensor of shape (n, dim)
            y: torch.Tensor of shape (n,) or (n, 1)
        """
        X = X.to(self.device)
        y = y.to(self.device)

        if len(y.shape) == 1:
            y = y.unsqueeze(-1)

        # Create RBF kernel
        kernel = ScaleKernel(RBFKernel(ard_num_dims=self.dim))
        likelihood = GaussianLikelihood()

        # Create and train GP model
        self.model = MLLGP(
            train_X=X,
            train_Y=y,
            kernel=kernel,
            likelihood=likelihood,
            lr=self.lr,
            n_epochs=self.n_epochs
        )

    def predict(self, X):
        """
        Predict mean and variance at X

        Args:
            X: torch.Tensor of shape (n, dim)

        Returns:
            mean: torch.Tensor of shape (n,)
            variance: torch.Tensor of shape (n,)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        X = X.to(self.device)
        self.model.eval()

        with torch.no_grad():
            posterior = self.model.posterior(X)
            mean = posterior.mean
            variance = posterior.variance

            # Ensure proper shapes
            if len(mean.shape) > 1:
                mean = mean.squeeze()
            if len(variance.shape) > 1:
                variance = variance.squeeze()

        return mean, variance

    def update(self, X_new, y_new):
        """
        Update model with new observations

        Args:
            X_new: torch.Tensor of shape (1, dim)
            y_new: torch.Tensor of shape (1,) or (1, 1)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        X_new = X_new.to(self.device)
        y_new = y_new.to(self.device)

        if len(y_new.shape) == 1:
            y_new = y_new.unsqueeze(-1)

        self.model = self.model.condition_on_observations(X_new, y_new)


class LaplaceBaseline:
    """Laplace-approximated BNN baseline for toy benchmarks"""

    def __init__(self, dim, hidden_dims=[64, 64], device='cpu',
                 lr=1e-3, n_epochs=500, batch_size=20):
        self.dim = dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = None

    def _get_net(self):
        """Create a simple MLP for Laplace"""
        layers = []
        input_dim = self.dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        return nn.Sequential(*layers)

    def fit(self, X, y):
        """
        Fit Laplace model to data

        Args:
            X: torch.Tensor of shape (n, dim)
            y: torch.Tensor of shape (n,) or (n, 1)
        """
        X = X.to(self.device)
        y = y.to(self.device)

        if len(y.shape) == 1:
            y = y.unsqueeze(-1)

        self.model = LaplaceBoTorch(
            get_net=self._get_net,
            train_X=X,
            train_Y=y,
            likelihood='regression',
            last_layer=False,
            hess_factorization='kron',
            marglik_mode='posthoc',
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            lr=self.lr,
            device=self.device
        )

    def predict(self, X):
        """
        Predict mean and variance at X

        Args:
            X: torch.Tensor of shape (n, dim)

        Returns:
            mean: torch.Tensor of shape (n,)
            variance: torch.Tensor of shape (n,)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        X = X.to(self.device)

        with torch.no_grad():
            mean, var = self.model.get_prediction(X, joint=False, use_test_loader=False)

        # Extract diagonal variance
        if len(var.shape) > 1:
            variance = torch.diagonal(var.squeeze(), dim1=-2, dim2=-1)
        else:
            variance = var

        return mean.squeeze(), variance.squeeze()

    def update(self, X_new, y_new):
        """
        Update model with new observations

        Args:
            X_new: torch.Tensor of shape (1, dim)
            y_new: torch.Tensor of shape (1,) or (1, 1)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        X_new = X_new.to(self.device)
        y_new = y_new.to(self.device)

        if len(y_new.shape) == 1:
            y_new = y_new.unsqueeze(-1)

        self.model = self.model.condition_on_observations(X_new, y_new)


def run_baseline_bo(baseline_model, toy_bench, n_init=40, n_iters=100,
                    acq_func='ei', device='cpu', seed=42):
    """
    Run baseline BO algorithm on toy benchmark

    Args:
        baseline_model: 'gp' or 'laplace'
        toy_bench: ToyBench instance
        n_init: Number of initial random points
        n_iters: Number of BO iterations
        acq_func: Acquisition function ('ei' or 'ucb')
        device: Device to run on
        seed: Random seed

    Returns:
        trace_best_y: Array of best y values over iterations
        trace_y_his: Array of selected y values over iterations
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get initial data
    dataset = toy_bench.dataset
    target_col = toy_bench.target_col
    target_col_transformed = toy_bench.target_col_transformed

    # Sample initial points
    indices = np.random.choice(len(dataset), size=n_init, replace=False)
    train_data = dataset.iloc[indices]

    # Get features (x values) and targets
    X_train = torch.tensor(np.stack(train_data['x'].values), dtype=torch.float32)
    y_train = torch.tensor(train_data[target_col].values, dtype=torch.float32)
    y_train_transformed = torch.tensor(train_data[target_col_transformed].values, dtype=torch.float32)

    # Initialize model
    dim = X_train.shape[1]
    if baseline_model == 'gp':
        model = GPBaseline(dim=dim, device=device, lr=0.01, n_epochs=100)
    elif baseline_model == 'laplace':
        model = LaplaceBaseline(dim=dim, hidden_dims=[64, 64], device=device,
                               lr=1e-3, n_epochs=500, batch_size=20)
    else:
        raise ValueError(f"Unknown baseline model: {baseline_model}")

    # Fit initial model on transformed targets (for maximization)
    model.fit(X_train, y_train_transformed)

    # Track best observed value (on original scale)
    id_max_observed = train_data[target_col_transformed].idxmax()
    opt_y_observed = train_data.loc[id_max_observed][target_col]
    best_y_transformed = train_data[target_col_transformed].max()

    # Initialize traces
    trace_best_y = [opt_y_observed] * (n_iters + 1)
    trace_y_his = [opt_y_observed] * (n_iters + 1)

    # Get all possible candidates
    remaining_indices = list(set(range(len(dataset))) - set(indices))
    X_candidates = torch.tensor(
        np.stack(dataset.iloc[remaining_indices]['x'].values),
        dtype=torch.float32
    )

    # BO loop
    for t in range(n_iters):
        # Predict on candidates
        with torch.no_grad():
            mean, var = model.predict(X_candidates.to(device))
            mean = mean.cpu()
            var = var.cpu()

        # Compute acquisition function
        if acq_func == 'ei':
            # Expected Improvement
            sigma = torch.sqrt(var)
            z = (mean - best_y_transformed) / (sigma + 1e-9)
            ei = (mean - best_y_transformed) * torch.distributions.Normal(0, 1).cdf(z) + \
                 sigma * torch.distributions.Normal(0, 1).log_prob(z).exp()
            acq_values = ei
        elif acq_func == 'ucb':
            # Upper Confidence Bound
            beta = 2.0
            acq_values = mean + beta * torch.sqrt(var)
        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

        # Select best candidate
        best_idx = torch.argmax(acq_values).item()
        selected_global_idx = remaining_indices[best_idx]

        # Get new observation
        new_data = dataset.iloc[selected_global_idx]
        X_new = torch.tensor(new_data['x'], dtype=torch.float32).unsqueeze(0)
        y_new = torch.tensor(new_data[target_col], dtype=torch.float32).unsqueeze(0)
        y_new_transformed = torch.tensor(new_data[target_col_transformed], dtype=torch.float32).unsqueeze(0)

        # Update best observed
        if y_new_transformed.item() > best_y_transformed:
            best_y_transformed = y_new_transformed.item()
            opt_y_observed = y_new.item()

        # Record traces
        trace_best_y[t + 1] = opt_y_observed
        trace_y_his[t + 1] = y_new.item()

        # Update model
        model.update(X_new, y_new_transformed)

        # Remove selected candidate
        remaining_indices.pop(best_idx)
        X_candidates = torch.cat([X_candidates[:best_idx], X_candidates[best_idx+1:]], dim=0)

        if len(remaining_indices) == 0:
            # Fill remaining iterations with last value
            for j in range(t + 2, n_iters + 1):
                trace_best_y[j] = opt_y_observed
                trace_y_his[j] = y_new.item()
            break

    return np.array(trace_best_y), np.array(trace_y_his)
