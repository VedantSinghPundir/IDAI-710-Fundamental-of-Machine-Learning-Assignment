"""
HW3: Unsupervised Learning & Dimensionality Reduction — Utility Functions
==========================================================================
Provided helper functions for data generation, visualization, and numerical
stability. Do NOT modify this file.

Approved libraries: numpy, pandas, matplotlib, seaborn, scipy, sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import multivariate_normal

# ============================================================================
# Color Palette (consistent across all plots)
# ============================================================================
COLORS = ['#4a90a4', '#e85d4c', '#22c55e', '#eab308', '#8b5cf6',
          '#f97316', '#06b6d4', '#ec4899']


# ============================================================================
# Data Generation
# ============================================================================

def generate_synthetic_clusters(means, covariances, n_samples_per_cluster=500,
                                random_state=None):
    """Generate synthetic 2D cluster data from multivariate Gaussians.

    Parameters
    ----------
    means : list of array-like
        Mean vectors for each cluster, each of shape (n_features,).
    covariances : list of array-like
        Covariance matrices for each cluster, each of shape
        (n_features, n_features).
    n_samples_per_cluster : int or list of int
        Number of samples per cluster. If int, same for all clusters.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_total_samples, n_features)
        Combined data from all clusters.
    y_true : np.ndarray of shape (n_total_samples,)
        True cluster labels (0, 1, ..., K-1).
    """
    rng = np.random.default_rng(random_state)
    K = len(means)

    if isinstance(n_samples_per_cluster, int):
        n_samples_per_cluster = [n_samples_per_cluster] * K

    X_list, y_list = [], []
    for k in range(K):
        samples = rng.multivariate_normal(
            np.asarray(means[k], dtype=float),
            np.asarray(covariances[k], dtype=float),
            size=n_samples_per_cluster[k]
        )
        X_list.append(samples)
        y_list.append(np.full(n_samples_per_cluster[k], k))

    X = np.vstack(X_list)
    y_true = np.concatenate(y_list)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y_true[idx]


# ============================================================================
# Numerical Utilities
# ============================================================================

def log_sum_exp(log_vals, axis=None):
    """Numerically stable log-sum-exp computation.

    Computes log(sum(exp(log_vals))) without overflow/underflow.
    This is critical for computing responsibilities in GMMs.

    Parameters
    ----------
    log_vals : np.ndarray
        Array of log values.
    axis : int or None
        Axis along which to compute.

    Returns
    -------
    result : np.ndarray or float
        log(sum(exp(log_vals))) along the specified axis.
    """
    max_val = np.max(log_vals, axis=axis, keepdims=True)
    result = max_val + np.log(np.sum(np.exp(log_vals - max_val), axis=axis,
                                      keepdims=True))
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    else:
        result = np.squeeze(result)
    return result


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_clusters(X, labels, centroids=None, title="", ax=None):
    """Plot 2D scatter with cluster coloring and optional centroids.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
    labels : np.ndarray of shape (n_samples,)
    centroids : np.ndarray of shape (K, 2) or None
    title : str
    ax : matplotlib Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = COLORS[int(i) % len(COLORS)]
        ax.scatter(X[mask, 0], X[mask, 1], s=20, alpha=0.5, c=color,
                   label=f'Cluster {label}', edgecolors='none')

    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black',
                   marker='X', edgecolors='white', linewidths=2, zorder=5,
                   label='Centroids')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best')
    return ax


def plot_gmm_contours(X, means, covariances, weights, labels=None,
                      title="", ax=None):
    """Plot GMM contours overlaid on data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
    means : np.ndarray of shape (K, 2)
    covariances : np.ndarray of shape (K, 2, 2)
    weights : np.ndarray of shape (K,)
    labels : np.ndarray of shape (n_samples,) or None
    title : str
    ax : matplotlib Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    K = len(means)

    if labels is not None:
        for k in range(K):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], s=15, alpha=0.4,
                       c=COLORS[k % len(COLORS)], edgecolors='none')
    else:
        ax.scatter(X[:, 0], X[:, 1], s=15, alpha=0.3, c='gray',
                   edgecolors='none')

    # Contours for each component
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    for k in range(K):
        color = COLORS[k % len(COLORS)]
        rv = multivariate_normal(means[k], covariances[k])
        zz = rv.pdf(grid).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=5, colors=[color], linewidths=1.5,
                   alpha=0.8)
        ax.scatter(*means[k], s=200, c='black', marker='X',
                   edgecolors='white', linewidths=2, zorder=5)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title, fontweight='bold')
    return ax


def plot_responsibilities(X, responsibilities, title="", ax=None):
    """Plot soft assignments as blended RGB colors.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
    responsibilities : np.ndarray of shape (n_samples, K)
    title : str
    ax : matplotlib Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    K = responsibilities.shape[1]
    base_colors = np.array([mcolors.to_rgb(COLORS[k % len(COLORS)])
                            for k in range(K)])

    point_colors = responsibilities @ base_colors
    point_colors = np.clip(point_colors, 0, 1)

    ax.scatter(X[:, 0], X[:, 1], c=point_colors, s=20, alpha=0.6,
               edgecolors='none')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title, fontweight='bold')

    for k in range(K):
        ax.scatter([], [], c=COLORS[k % len(COLORS)], s=60,
                   label=f'Component {k}')
    ax.legend(loc='best')
    return ax


def plot_scree(explained_variance_ratio, threshold=None,
               title="Scree Plot", ax=None):
    """Plot scree/elbow plot for PCA explained variance.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
    threshold : float or None
    title : str
    ax : matplotlib Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    ax.bar(range(1, n + 1), explained_variance_ratio, alpha=0.6,
           color=COLORS[0], label='Individual')
    ax.step(range(1, n + 1), cumulative, where='mid', color=COLORS[1],
            linewidth=2, label='Cumulative')

    if threshold is not None:
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.7,
                   label=f'Threshold ({threshold:.0%})')
        idx = np.argmax(cumulative >= threshold)
        ax.axvline(x=idx + 1, color='gray', linestyle=':', alpha=0.5)
        ax.scatter([idx + 1], [cumulative[idx]], s=100, c=COLORS[1],
                   zorder=5, edgecolors='black')

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(range(1, n + 1))
    ax.legend(loc='center right')
    ax.set_ylim(0, 1.05)
    return ax


def plot_projection_2d(X_proj, labels=None, class_names=None,
                       xlabel="Component 1", ylabel="Component 2",
                       title="", ax=None):
    """Plot 2D projection with optional class coloring.

    Parameters
    ----------
    X_proj : np.ndarray of shape (n_samples, 2+)
    labels : np.ndarray of shape (n_samples,) or None
    class_names : list of str or None
    xlabel, ylabel : str
    title : str
    ax : matplotlib Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if labels is not None:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = class_names[i] if class_names else f'Class {label}'
            ax.scatter(X_proj[mask, 0], X_proj[mask, 1], s=30, alpha=0.6,
                       c=COLORS[i % len(COLORS)], label=name,
                       edgecolors='none')
        ax.legend(loc='best')
    else:
        ax.scatter(X_proj[:, 0], X_proj[:, 1], s=30, alpha=0.5,
                   c=COLORS[0], edgecolors='none')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    return ax


def plot_comparison(X_proj_yours, X_proj_sklearn, labels=None,
                    class_names=None, method_name="PCA",
                    xlabel="Component 1", ylabel="Component 2"):
    """Side-by-side comparison of your implementation vs sklearn.

    Parameters
    ----------
    X_proj_yours : np.ndarray of shape (n_samples, 2)
    X_proj_sklearn : np.ndarray of shape (n_samples, 2)
    labels : np.ndarray or None
    class_names : list of str or None
    method_name : str
    xlabel, ylabel : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_projection_2d(X_proj_yours, labels, class_names,
                       xlabel, ylabel,
                       title=f'Your {method_name}', ax=axes[0])
    plot_projection_2d(X_proj_sklearn, labels, class_names,
                       xlabel, ylabel,
                       title=f'sklearn {method_name}', ax=axes[1])

    plt.tight_layout()
    return fig, axes
