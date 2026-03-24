"""
HW3: Unsupervised Learning & Dimensionality Reduction — Clustering
===================================================================
Implement K-Means and Gaussian Mixture Model (GMM) clustering algorithms.

Your implementations should follow the API described in each class docstring.
Methods marked with TODO require your implementation.

Approved libraries: numpy, scipy
"""

import numpy as np
from scipy.stats import multivariate_normal
from utils import log_sum_exp


# ============================================================================
# K-Means Clustering
# ============================================================================

class KMeans:
    """K-Means clustering algorithm.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (K).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance. If the centroids move less than this
        (max Euclidean distance), stop early.
    random_state : int or None
        Random seed for reproducibility.

    Attributes (available after fit)
    ----------
    centroids_ : np.ndarray of shape (n_clusters, n_features)
        Learned centroid positions.
    labels_ : np.ndarray of shape (n_samples,)
        Cluster assignments from the last fit call.
    n_iter_ : int
        Number of iterations run.
    inertia_ : float
        Sum of squared distances to nearest centroid.
    """

    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_centroids(self, X):
        """Initialize centroids by selecting K random data points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        centroids : np.ndarray of shape (n_clusters, n_features)
        """
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _assign_clusters(self, X, centroids):
        """Assign each sample to the nearest centroid.

        TODO: Implement this method.

        For each sample in X, compute the Euclidean distance to each centroid
        and assign the sample to the nearest one.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points.
        centroids : np.ndarray of shape (n_clusters, n_features)
            Current centroid positions.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster assignment for each sample (integer in [0, K-1]).

        Hints
        -----
        - You can compute all pairwise distances efficiently using broadcasting
          or np.linalg.norm with appropriate axis arguments.
        - np.argmin is useful for finding the nearest centroid.
        """
        # ==================== TODO ====================
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            distances = []
            for k in range(self.n_clusters):
                diff = X[i] - centroids[k]
                dist = np.sqrt(np.sum(diff ** 2))
                distances.append(dist)
                labels[i] = np.argmin(distances)
        return labels
        raise NotImplementedError("Implement _assign_clusters")
        # ==================== END TODO ====================

    def _update_centroids(self, X, labels):
        """Update centroids as the mean of assigned points.

        TODO: Implement this method.

        For each cluster k, compute the mean of all samples assigned to
        cluster k. This becomes the new centroid for cluster k.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points.
        labels : np.ndarray of shape (n_samples,)
            Current cluster assignments.

        Returns
        -------
        new_centroids : np.ndarray of shape (n_clusters, n_features)
            Updated centroid positions.

        Hints
        -----
        - Use boolean masking: X[labels == k] gives all points in cluster k.
        - Handle edge case: if a cluster has no assigned points, keep its
          centroid unchanged (or reinitialize randomly).
        """
        # ==================== TODO ====================
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features), dtype=float)
        rng = np.random.default_rng(self.random_state)
        for k in range(self.n_clusters):
            X_k = X[labels == k]
            if X_k.shape[0] == 0:
                idx = rng.integers(0, X.shape[0])
                new_centroids[k] = X[idx]
            else:
                new_centroids[k] = X_k.mean(axis=0)
        return new_centroids
        raise NotImplementedError("Implement _update_centroids")
        # ==================== END TODO ====================

    def fit(self, X):
        """Run K-Means clustering on the data.

        TODO: Implement the main K-Means loop.

        Algorithm:
        1. Initialize centroids using self._init_centroids(X)
        2. Repeat until convergence or max_iter:
           a. Assign each point to nearest centroid (_assign_clusters)
           b. Update centroids as cluster means (_update_centroids)
           c. Check convergence: if max centroid shift < self.tol, stop
        3. Store results in self.centroids_, self.labels_, self.n_iter_

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        # ==================== TODO ====================
        centroids = self._init_centroids(X)
        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)
            max_shift = 0
            for k in range(self.n_clusters):
                diff = new_centroids[k] - centroids[k]
                distance = np.sqrt(np.sum(diff ** 2))
                if distance > max_shift:
                    max_shift = distance
            centroids = new_centroids
            
            if max_shift < self.tol:
                break
                
        self.centroids_ = centroids
        self.labels_ = self._assign_clusters(X, centroids)
        self.n_iter_ = iteration + 1
        total_error = 0
        
        for i in range(X.shape[0]):
            centroid = self.centroids_[self.labels_[i]]
            diff = X[i] - centroid
            total_error += np.sum(diff ** 2)
        self.inertia_ = total_error
        return self
        
        raise NotImplementedError("Implement fit")
        # ==================== END TODO ====================

    def predict(self, X):
        """Assign new data points to nearest centroid.

        TODO: Implement this method.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
        """
        # ==================== TODO ====================
        labels = []
        for i in range(X.shape[0]):
            distances = []
            for k in range(self.n_clusters):
                diff = X[i] - self.centroids_[k]
                dist = np.sqrt(np.sum(diff ** 2))
                distances.append(dist)
            cluster = np.argmin(distances)
            labels.append(cluster)
        
        return np.array(labels)
        raise NotImplementedError("Implement predict")
        # ==================== END TODO ====================

    def get_centroids(self):
        """Return the learned centroids.

        Returns
        -------
        centroids : np.ndarray of shape (n_clusters, n_features)
        """
        return self.centroids_.copy()


# ============================================================================
# Gaussian Mixture Model (EM Algorithm)
# ============================================================================

class GMM:
    """Gaussian Mixture Model with Expectation-Maximization.

    The EM algorithm alternates between:
      - E-step: Compute responsibilities (posterior probabilities)
      - M-step: Update parameters using weighted MLE

    Parameters
    ----------
    n_components : int
        Number of mixture components (K).
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.
    random_state : int or None
        Random seed for reproducibility.

    Attributes (available after fit)
    ----------
    means_ : np.ndarray of shape (n_components, n_features)
        Learned component means.
    covariances_ : np.ndarray of shape (n_components, n_features, n_features)
        Learned component covariance matrices.
    weights_ : np.ndarray of shape (n_components,)
        Learned mixing weights (sum to 1).
    n_iter_ : int
        Number of EM iterations run.
    log_likelihoods_ : list of float
        Log-likelihood at each iteration (for monitoring convergence).
    """

    def __init__(self, n_components=2, max_iter=100, tol=1e-6,
                 random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _initialize_params(self, X):
        """Initialize GMM parameters using K-Means.

        This is provided for you. K-Means initialization is a standard
        strategy that gives EM a good starting point.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        K = self.n_components

        # Use K-Means for initial cluster assignments
        km = KMeans(n_clusters=K, max_iter=50, random_state=self.random_state)
        km.fit(X)

        self.means_ = km.centroids_.copy()
        self.covariances_ = np.array([np.cov(X[km.labels_ == k].T) + 1e-6 * np.eye(n_features)
                                       for k in range(K)])
        self.weights_ = np.array([np.mean(km.labels_ == k) for k in range(K)])

    def _compute_component_log_likelihoods(self, X):
        """Compute log N(x_n | mu_k, Sigma_k) for all samples and components.

        This is provided for you. It handles the multivariate Gaussian PDF
        computation with numerical stability.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        log_likelihoods : np.ndarray of shape (n_samples, n_components)
            Entry (n, k) = log N(x_n | mu_k, Sigma_k).
        """
        n_samples = X.shape[0]
        K = self.n_components
        log_liks = np.zeros((n_samples, K))

        for k in range(K):
            try:
                rv = multivariate_normal(mean=self.means_[k],
                                         cov=self.covariances_[k],
                                         allow_singular=True)
                log_liks[:, k] = rv.logpdf(X)
            except np.linalg.LinAlgError:
                log_liks[:, k] = -1e10  # fallback for singular covariance

        return log_liks

    def compute_responsibilities(self, X):
        """E-step: Compute responsibilities (posterior probabilities).

        TODO: Implement this method.

        The responsibility gamma(z_nk) is the posterior probability that
        sample n belongs to component k:

            gamma(z_nk) = pi_k * N(x_n | mu_k, Sigma_k)
                          ---------------------------------
                          sum_j pi_j * N(x_n | mu_j, Sigma_j)

        This is Bayes' theorem applied to the mixture model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        responsibilities : np.ndarray of shape (n_samples, n_components)
            Each row sums to 1. Entry (n, k) = gamma(z_nk).

        Hints
        -----
        - Use self._compute_component_log_likelihoods(X) to get
          log N(x_n | mu_k, Sigma_k) for all (n, k).
        - Work in log-space: log(gamma_nk) = log(pi_k) + log N(x_n|...)
          minus the log of the normalizing constant.
        - The log normalizing constant is log_sum_exp of the numerator
          across components (axis=1). Use the provided log_sum_exp function
          from utils.py for numerical stability.
        - Convert back from log-space with np.exp.
        """
        # ==================== TODO ====================
        n_samples = X.shape[0]
        K = self.n_components
        log_liks = self._compute_component_log_likelihoods(X)
        
        eps = 1e-15
        log_weights = np.log(self.weights_ + eps)

        log_num = log_liks + log_weights
        
        log_denom = log_sum_exp(log_num, axis=1)
        
        log_resp = log_num - log_denom[:, None]
        
        responsibilities = np.exp(log_resp)
        
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
        
        return responsibilities
        raise NotImplementedError("Implement compute_responsibilities")
        # ==================== END TODO ====================

    def _m_step(self, X, responsibilities):
        """M-step: Update parameters using weighted MLE.

        TODO: Implement this method.

        Given the responsibilities from the E-step, update the parameters
        using weighted maximum likelihood estimates:

        Effective count:   N_k = sum_n gamma(z_nk)

        Updated mean:      mu_k = (1/N_k) * sum_n gamma(z_nk) * x_n

        Updated covariance: Sigma_k = (1/N_k) * sum_n gamma(z_nk) *
                                      (x_n - mu_k)(x_n - mu_k)^T

        Updated weight:    pi_k = N_k / N

        These are the same MLE formulas from Session 3, but with each
        sample weighted by its responsibility gamma(z_nk).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        responsibilities : np.ndarray of shape (n_samples, n_components)

        Side Effects
        ------------
        Updates self.means_, self.covariances_, self.weights_ in-place.

        Hints
        -----
        - N_k = responsibilities[:, k].sum()
        - For the mean: weighted average with responsibilities as weights.
        - For the covariance: compute diff = X - mu_k, then
          Sigma_k = (diff.T @ diag(gamma_k) @ diff) / N_k
          Or equivalently: (gamma_k[:, None] * diff).T @ diff / N_k
        - Add a small regularization term (e.g., 1e-6 * I) to the
          covariance to prevent singularity.
        """
        # ==================== TODO ====================
        N, D = X.shape
        K = self.n_components
        Nk = responsibilities.sum(axis=0)
        eps = 1e-15
        Nk_safe = Nk + eps
        self.weights_ = Nk_safe / N
        self.weights_ = self.weights_ / self.weights_.sum()

        self.means_ = (responsibilities.T @ X) / Nk_safe[:, None]

        covariances = np.zeros((K, D, D), dtype=float)
        reg = 1e-6

        for k in range(K):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k][:, None] * diff
            cov_k = (weighted_diff.T @ diff) / Nk_safe[k]
            cov_k = cov_k + reg * np.eye(D)
            covariances[k] = cov_k

        self.covariances_ = covariances
        
        
        # raise NotImplementedError("Implement _m_step")
        # ==================== END TODO ====================

    def fit(self, X):
        """Fit GMM using the EM algorithm.

        TODO: Implement the EM loop.

        Algorithm:
        1. Initialize parameters using self._initialize_params(X)
        2. Repeat until convergence or max_iter:
           a. E-step: compute responsibilities via self.compute_responsibilities(X)
           b. M-step: update parameters via self._m_step(X, responsibilities)
           c. Compute log-likelihood and check for convergence
        3. Store iteration count in self.n_iter_

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        self

        Hints
        -----
        - The data log-likelihood is:
          L = sum_n log( sum_k pi_k * N(x_n | mu_k, Sigma_k) )
          You can compute this using _compute_component_log_likelihoods
          and log_sum_exp.
        - Store log-likelihoods in self.log_likelihoods_ to monitor convergence.
        - Converge when |L_new - L_old| < self.tol.
        """
        # ==================== TODO ====================
        self._initialize_params(X)
        self.log_likelihoods_ = []
        
        prev_ll = None

        for it in range(1, self.max_iter + 1):
            responsibilities = self.compute_responsibilities(X)
            self._m_step(X, responsibilities)

            log_liks = self._compute_component_log_likelihoods(X)
            eps = 1e-15
            log_weights = np.log(self.weights_ + eps)
            log_num = log_liks + log_weights
            ll = np.sum(log_sum_exp(log_num, axis=1))
            
            self.log_likelihoods_.append(float(ll))

            if prev_ll is not None:
                if abs(ll - prev_ll) < self.tol:
                    self.n_iter_ = it
                    return self
            
            prev_ll = ll

        self.n_iter_ = self.max_iter
        return self
        
        raise NotImplementedError("Implement fit")
        # ==================== END TODO ====================

    def predict_proba(self, X):
        """Return responsibilities for each sample (soft assignments).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        responsibilities : np.ndarray of shape (n_samples, n_components)
        """
        return self.compute_responsibilities(X)

    def predict(self, X):
        """Assign each sample to the most likely component (hard assignments).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
        """
        return np.argmax(self.compute_responsibilities(X), axis=1)

    def get_params(self):
        """Return learned parameters.

        Returns
        -------
        params : dict
            Keys: 'means', 'covariances', 'weights'.
        """
        return {
            'means': self.means_.copy(),
            'covariances': self.covariances_.copy(),
            'weights': self.weights_.copy()
        }
