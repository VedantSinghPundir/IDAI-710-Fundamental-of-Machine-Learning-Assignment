"""
HW3: Unsupervised Learning & Dimensionality Reduction — Dim. Reduction
=======================================================================
Implement Principal Component Analysis (PCA) and Linear Discriminant
Analysis (LDA) as dimensionality reduction / projection techniques.

Methods marked with TODO require your implementation.

Approved libraries: numpy, scipy
"""

import numpy as np
from scipy.linalg import eigh


# ============================================================================
# Principal Component Analysis (PCA)
# ============================================================================

class PCA:
    """Principal Component Analysis for dimensionality reduction.

    PCA finds orthogonal directions (principal components) that capture
    the most variance in the data. It is an unsupervised technique — it
    does not use class labels.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. If None, use variance_threshold
        to determine the number automatically.
    variance_threshold : float
        Minimum cumulative explained variance to retain. Only used when
        n_components is None. Default: 0.95 (keep 95% of variance).

    Attributes (available after fit)
    ----------
    components_ : np.ndarray of shape (n_components, n_features)
        Principal component directions (rows are eigenvectors).
    explained_variance_ : np.ndarray of shape (n_components,)
        Variance explained by each component (eigenvalues).
    explained_variance_ratio_ : np.ndarray of shape (n_components,)
        Proportion of total variance explained by each component.
    n_components_ : int
        Actual number of components retained.
    mean_ : np.ndarray of shape (n_features,)
        Per-feature mean of the training data (used for centering).
    """

    def __init__(self, n_components=None, variance_threshold=0.95):
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    def fit(self, X):
        """Fit PCA to the data.

        TODO: Implement this method.

        Algorithm:
        1. Compute and store the mean of each feature: self.mean_
        2. Center the data by subtracting the mean
        3. Compute the covariance matrix of the centered data
        4. Perform eigendecomposition of the covariance matrix
        5. Sort eigenvectors by eigenvalue in descending order
        6. Determine the number of components to keep:
           - If self.n_components is set, keep that many
           - Otherwise, keep enough to exceed self.variance_threshold
             of cumulative explained variance
        7. Store results in the attributes listed in the class docstring

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self

        Hints
        -----
        - np.cov(X_centered.T) computes the covariance matrix (note the .T).
        - np.linalg.eigh is numerically stable for symmetric matrices and
          returns eigenvalues in ascending order. You will need to reverse.
        - The explained variance ratio is eigenvalue_k / sum(all eigenvalues).
        - np.cumsum is useful for computing cumulative sums.
        """
        # ==================== TODO ====================
        N, D = X.shape
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        cov = np.cov(X_centered.T, bias=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigvals = np.maximum(eigvals, 0.0)

        total_var = eigvals.sum()
        if total_var <= 0:
            explained_ratio_all = np.zeros_like(eigvals)
        else:
            explained_ratio_all = eigvals / total_var

        if self.n_components is not None:
            n_keep = int(self.n_components)
            n_keep = max(1, min(n_keep, D))
        else:
            cumulative = np.cumsum(explained_ratio_all)
            n_keep = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
            n_keep = max(1, min(n_keep, D))

        self.n_components_ = n_keep
        self.components_ = eigvecs[:, :n_keep].T
        self.explained_variance_ = eigvals[:n_keep]
        self.explained_variance_ratio_ = explained_ratio_all[:n_keep]
        
        return self

        raise NotImplementedError("Implement fit")
        # ==================== END TODO ====================

    def transform(self, X):
        """Project data onto the principal components.

        TODO: Implement this method.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        X_proj : np.ndarray of shape (n_samples, n_components_)
            Projected data.

        Hints
        -----
        - Center X using self.mean_ (same centering as in fit).
        - Project: X_proj = X_centered @ self.components_.T
        """
        # ==================== TODO ====================
        X_centered = X - self.mean_
        X_proj = X_centered @ self.components_.T
        return X_proj
        raise NotImplementedError("Implement transform")
        # ==================== END TODO ====================

    def get_explained_variance_ratio(self):
        """Return explained variance ratio for each retained component.

        Returns
        -------
        ratios : np.ndarray of shape (n_components_,)
        """
        return self.explained_variance_ratio_.copy()

    def get_components(self):
        """Return the principal component directions (projection matrix).

        Returns
        -------
        components : np.ndarray of shape (n_components_, n_features)
        """
        return self.components_.copy()


# ============================================================================
# LDA Projection (Supervised Dimensionality Reduction)
# ============================================================================

class LDAProjection:
    """Linear Discriminant Analysis as a dimensionality reduction technique.

    Unlike PCA (unsupervised), LDA uses class labels to find directions
    that maximize between-class separation relative to within-class scatter.

    The maximum number of LDA components is min(n_features, n_classes - 1).

    Parameters
    ----------
    n_components : int or None
        Number of discriminant components to keep.
        If None, keep min(n_features, n_classes - 1).

    Attributes (available after fit)
    ----------
    components_ : np.ndarray of shape (n_components_, n_features)
        LDA projection directions (rows are eigenvectors).
    explained_variance_ratio_ : np.ndarray of shape (n_components_,)
        Proportion of discriminant information captured by each component
        (based on eigenvalues of S_W^{-1} S_B).
    n_components_ : int
        Actual number of components retained.
    classes_ : np.ndarray
        Unique class labels.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def _compute_scatter_matrices(self, X, y):
        """Compute within-class and between-class scatter matrices.

        This is provided for you.

        Within-class scatter:
            S_W = sum_k sum_{x in C_k} (x - mu_k)(x - mu_k)^T

        Between-class scatter:
            S_B = sum_k N_k * (mu_k - mu)(mu_k - mu)^T

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        S_W : np.ndarray of shape (n_features, n_features)
            Within-class scatter matrix.
        S_B : np.ndarray of shape (n_features, n_features)
            Between-class scatter matrix.
        """
        classes = np.unique(y)
        n_features = X.shape[1]
        overall_mean = X.mean(axis=0)

        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in classes:
            X_c = X[y == c]
            mean_c = X_c.mean(axis=0)
            n_c = X_c.shape[0]

            # Within-class scatter
            diff = X_c - mean_c
            S_W += diff.T @ diff

            # Between-class scatter
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        return S_W, S_B

    def fit(self, X, y):
        """Fit LDA projection to labeled data.

        TODO: Implement this method.

        Algorithm:
        1. Store unique class labels in self.classes_
        2. Call self._compute_scatter_matrices(X, y) to get S_W and S_B
        3. Solve the generalized eigenvalue problem: S_B w = lambda S_W w
           (equivalently, compute eigenvectors of S_W^{-1} S_B)
        4. Sort eigenvectors by eigenvalue in descending order
        5. Keep only eigenvectors with positive eigenvalues
        6. Determine number of components:
           - Max possible is min(n_features, n_classes - 1)
           - If self.n_components is set, use min(self.n_components, max_possible)
           - Otherwise, use max_possible
        7. Store projection matrix and explained variance ratio

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        self

        Hints
        -----
        - scipy.linalg.eigh(S_B, S_W) solves the generalized eigenvalue
          problem directly and is numerically stable.
        - Alternatively: np.linalg.eigh(np.linalg.inv(S_W) @ S_B) works
          but may be less stable. Add regularization to S_W if needed:
          S_W + 1e-6 * np.eye(n_features).
        - eigh returns eigenvalues in ascending order — reverse them.
        - Some eigenvalues may be near-zero or negative due to numerical
          noise; keep only those that are positive (> 1e-10).
        """
        # ==================== TODO ====================
        N, D = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        S_W, S_B = self._compute_scatter_matrices(X, y)

        reg = 1e-6
        S_W = S_W + reg * np.eye(D)

        eigvals, eigvecs = eigh(S_B, S_W)
        
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        positive = eigvals > 1e-10
        eigvals = eigvals[positive]
        eigvecs = eigvecs[:, positive]

        max_possible = min(D, n_classes - 1)

        if self.n_components is None:
            n_keep = max_possible
        else:
            n_keep = min(self.n_components, max_possible)
        self.n_components_ = n_keep

        self.components_ = eigvecs[:, :n_keep].T
        
        total = eigvals.sum()
        if total > 0:
            self.explained_variance_ratio_ = eigvals[:n_keep] / total
        else:
            self.explained_variance_ratio_ = np.zeros(n_keep)

        return self
        raise NotImplementedError("Implement fit")
        # ==================== END TODO ====================

    def transform(self, X):
        """Project data onto LDA components.

        TODO: Implement this method.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_proj : np.ndarray of shape (n_samples, n_components_)

        Hints
        -----
        - X_proj = X @ self.components_.T
        - Note: LDA does not require centering (the scatter matrices
          already account for means).
        """
        # ==================== TODO ====================
        X_proj = X @ self.components_.T
        return X_proj
        raise NotImplementedError("Implement transform")
        # ==================== END TODO ====================
