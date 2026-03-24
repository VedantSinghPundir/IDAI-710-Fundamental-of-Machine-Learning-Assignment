"""
HW3: Autograder — INSTRUCTOR ONLY
===================================
Automated tests for student implementations of KMeans, GMM, PCA,
and LDAProjection. Run with: python autograder.py

Scoring: Each test is worth a fraction of the autograder portion.
The autograder total corresponds to ~20% of the HW grade.
"""

import numpy as np
import sys
import traceback
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.datasets import load_iris

# ============================================================================
# Test infrastructure
# ============================================================================

class TestResult:
    def __init__(self, name, passed, points, max_points, message=""):
        self.name = name
        self.passed = passed
        self.points = points
        self.max_points = max_points
        self.message = message

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        msg = f"  [{status}] {self.name}: {self.points}/{self.max_points}"
        if self.message:
            msg += f"  — {self.message}"
        return msg


def run_test(name, test_fn, max_points):
    """Run a single test and return a TestResult."""
    try:
        passed, msg = test_fn()
        points = max_points if passed else 0
        return TestResult(name, passed, points, max_points, msg)
    except NotImplementedError as e:
        return TestResult(name, False, 0, max_points, f"Not implemented: {e}")
    except Exception as e:
        tb = traceback.format_exc()
        return TestResult(name, False, 0, max_points, f"Error: {e}\n{tb}")


# ============================================================================
# Shared test data
# ============================================================================

def get_cluster_data():
    """Well-separated 2D clusters for testing."""
    rng = np.random.default_rng(42)
    X0 = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], 200)
    X1 = rng.multivariate_normal([8, 8], [[1, 0], [0, 1]], 200)
    X = np.vstack([X0, X1])
    y = np.array([0]*200 + [1]*200)
    idx = rng.permutation(400)
    return X[idx], y[idx]


def get_three_cluster_data():
    """Three well-separated 2D clusters."""
    rng = np.random.default_rng(42)
    X0 = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], 200)
    X1 = rng.multivariate_normal([8, 0], [[1, 0.3], [0.3, 1]], 200)
    X2 = rng.multivariate_normal([4, 7], [[0.8, 0], [0, 0.8]], 200)
    X = np.vstack([X0, X1, X2])
    y = np.array([0]*200 + [1]*200 + [2]*200)
    idx = rng.permutation(600)
    return X[idx], y[idx]


# ============================================================================
# K-Means Tests
# ============================================================================

def test_kmeans_fit_shape():
    from clustering import KMeans
    X, _ = get_cluster_data()
    km = KMeans(n_clusters=2, random_state=42)
    result = km.fit(X)
    if result is not km:
        return False, "fit() should return self"
    if km.centroids_.shape != (2, 2):
        return False, f"centroids_ shape: expected (2,2), got {km.centroids_.shape}"
    if km.labels_.shape != (400,):
        return False, f"labels_ shape: expected (400,), got {km.labels_.shape}"
    return True, "Shapes correct"


def test_kmeans_predict_shape():
    from clustering import KMeans
    X, _ = get_cluster_data()
    km = KMeans(n_clusters=2, random_state=42).fit(X)
    labels = km.predict(X[:10])
    if labels.shape != (10,):
        return False, f"predict shape: expected (10,), got {labels.shape}"
    if not np.all(np.isin(labels, [0, 1])):
        return False, "Labels should be in {0, 1}"
    return True, "Shapes correct"


def test_kmeans_convergence():
    from clustering import KMeans
    X, _ = get_cluster_data()
    km = KMeans(n_clusters=2, random_state=42).fit(X)
    if km.n_iter_ >= km.max_iter:
        return False, f"Did not converge within {km.max_iter} iterations"
    return True, f"Converged in {km.n_iter_} iterations"


def test_kmeans_known_clusters():
    from clustering import KMeans
    X, y_true = get_cluster_data()
    km = KMeans(n_clusters=2, random_state=42).fit(X)

    # Centroids should be near [0,0] and [8,8] (order may vary)
    centroids_sorted = km.centroids_[np.argsort(km.centroids_[:, 0])]
    expected = np.array([[0, 0], [8, 8]])

    if np.max(np.abs(centroids_sorted - expected)) > 1.5:
        return False, f"Centroids far from expected: {centroids_sorted}"

    return True, f"Centroids near expected values"


def test_kmeans_matches_sklearn():
    from clustering import KMeans
    X, _ = get_cluster_data()

    km_ours = KMeans(n_clusters=2, random_state=42).fit(X)
    km_sk = SklearnKMeans(n_clusters=2, random_state=42, n_init=1).fit(X)

    # Compare centroids (order-independent)
    c_ours = km_ours.centroids_[np.argsort(km_ours.centroids_[:, 0])]
    c_sk = km_sk.cluster_centers_[np.argsort(km_sk.cluster_centers_[:, 0])]

    if np.max(np.abs(c_ours - c_sk)) > 2.0:
        return False, f"Centroids differ significantly from sklearn"

    return True, "Centroids match sklearn (within tolerance)"


# ============================================================================
# GMM Tests
# ============================================================================

def test_gmm_responsibilities_shape():
    from clustering import GMM
    X, _ = get_cluster_data()
    gmm = GMM(n_components=2, random_state=42)
    gmm._initialize_params(X)
    resp = gmm.compute_responsibilities(X)
    if resp.shape != (400, 2):
        return False, f"Shape: expected (400,2), got {resp.shape}"
    return True, "Shape correct"


def test_gmm_responsibilities_sum_to_one():
    from clustering import GMM
    X, _ = get_cluster_data()
    gmm = GMM(n_components=2, random_state=42)
    gmm._initialize_params(X)
    resp = gmm.compute_responsibilities(X)
    row_sums = resp.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        max_err = np.max(np.abs(row_sums - 1.0))
        return False, f"Rows don't sum to 1; max deviation: {max_err:.6f}"
    return True, "All rows sum to 1"


def test_gmm_responsibilities_nonnegative():
    from clustering import GMM
    X, _ = get_cluster_data()
    gmm = GMM(n_components=2, random_state=42)
    gmm._initialize_params(X)
    resp = gmm.compute_responsibilities(X)
    if np.any(resp < -1e-10):
        return False, f"Negative responsibilities found: min = {resp.min()}"
    return True, "All responsibilities non-negative"


def test_gmm_fit_runs():
    from clustering import GMM
    X, _ = get_cluster_data()
    gmm = GMM(n_components=2, max_iter=50, random_state=42)
    result = gmm.fit(X)
    if result is not gmm:
        return False, "fit() should return self"
    if gmm.means_.shape != (2, 2):
        return False, f"means_ shape: expected (2,2), got {gmm.means_.shape}"
    if gmm.covariances_.shape != (2, 2, 2):
        return False, f"covariances_ shape: expected (2,2,2), got {gmm.covariances_.shape}"
    if gmm.weights_.shape != (2,):
        return False, f"weights_ shape: expected (2,), got {gmm.weights_.shape}"
    return True, "Fit completed with correct shapes"


def test_gmm_weights_sum_to_one():
    from clustering import GMM
    X, _ = get_cluster_data()
    gmm = GMM(n_components=2, random_state=42).fit(X)
    if not np.isclose(gmm.weights_.sum(), 1.0, atol=1e-5):
        return False, f"Weights sum to {gmm.weights_.sum()}, expected 1.0"
    return True, "Weights sum to 1"


def test_gmm_known_clusters():
    from clustering import GMM
    X, _ = get_cluster_data()
    gmm = GMM(n_components=2, random_state=42).fit(X)

    means_sorted = gmm.means_[np.argsort(gmm.means_[:, 0])]
    expected = np.array([[0, 0], [8, 8]])

    if np.max(np.abs(means_sorted - expected)) > 1.5:
        return False, f"Means far from expected: {means_sorted}"

    return True, "Means near expected values"


def test_gmm_log_likelihood_increases():
    from clustering import GMM
    X, _ = get_cluster_data()
    gmm = GMM(n_components=2, max_iter=50, random_state=42).fit(X)

    if not hasattr(gmm, 'log_likelihoods_') or len(gmm.log_likelihoods_) < 2:
        return False, "log_likelihoods_ not stored or too short"

    lls = gmm.log_likelihoods_
    decreases = sum(1 for i in range(1, len(lls)) if lls[i] < lls[i-1] - 1e-6)
    if decreases > 1:
        return False, f"Log-likelihood decreased {decreases} times"

    return True, "Log-likelihood monotonically increases"


def test_gmm_predict_shape():
    from clustering import GMM
    X, _ = get_cluster_data()
    gmm = GMM(n_components=2, random_state=42).fit(X)
    labels = gmm.predict(X[:10])
    proba = gmm.predict_proba(X[:10])

    if labels.shape != (10,):
        return False, f"predict shape: expected (10,), got {labels.shape}"
    if proba.shape != (10, 2):
        return False, f"predict_proba shape: expected (10,2), got {proba.shape}"
    return True, "Shapes correct"


# ============================================================================
# PCA Tests
# ============================================================================

def test_pca_fit_shape():
    from dimensionality_reduction import PCA
    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=2).fit(X)
    if pca.components_.shape != (2, 4):
        return False, f"components_ shape: expected (2,4), got {pca.components_.shape}"
    if pca.explained_variance_ratio_.shape != (2,):
        return False, f"evr shape: expected (2,), got {pca.explained_variance_ratio_.shape}"
    return True, "Shapes correct"


def test_pca_transform_shape():
    from dimensionality_reduction import PCA
    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=2).fit(X)
    X_proj = pca.transform(X)
    if X_proj.shape != (150, 2):
        return False, f"Projected shape: expected (150,2), got {X_proj.shape}"
    return True, "Shape correct"


def test_pca_variance_sorted():
    from dimensionality_reduction import PCA
    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=4).fit(X)
    evr = pca.explained_variance_ratio_
    if not np.all(evr[:-1] >= evr[1:] - 1e-10):
        return False, f"Variance ratios not sorted descending: {evr}"
    return True, "Variance ratios sorted correctly"


def test_pca_variance_sum():
    from dimensionality_reduction import PCA
    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=4).fit(X)
    total = pca.explained_variance_ratio_.sum()
    if not np.isclose(total, 1.0, atol=0.01):
        return False, f"Total variance ratio: {total}, expected ~1.0"
    return True, f"Total variance ratio: {total:.4f}"


def test_pca_threshold():
    from dimensionality_reduction import PCA
    iris = load_iris()
    X = iris.data

    pca = PCA(variance_threshold=0.95).fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    if cumvar[-1] < 0.95:
        return False, f"Cumulative variance {cumvar[-1]:.3f} < 0.95"
    return True, f"Threshold met: {pca.n_components_} components, cumvar={cumvar[-1]:.3f}"


def test_pca_matches_sklearn():
    from dimensionality_reduction import PCA
    iris = load_iris()
    X = iris.data

    pca_ours = PCA(n_components=2).fit(X)
    pca_sk = SklearnPCA(n_components=2).fit(X)

    X_ours = pca_ours.transform(X)
    X_sk = pca_sk.transform(X)

    # Components may have opposite sign; check correlation
    for i in range(2):
        corr = abs(np.corrcoef(X_ours[:, i], X_sk[:, i])[0, 1])
        if corr < 0.99:
            return False, f"Component {i} correlation with sklearn: {corr:.4f}"

    return True, "Matches sklearn (up to sign)"


def test_pca_centering():
    from dimensionality_reduction import PCA
    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=2).fit(X)

    if not hasattr(pca, 'mean_'):
        return False, "mean_ attribute not set"
    if not np.allclose(pca.mean_, X.mean(axis=0), atol=1e-10):
        return False, "mean_ does not match data mean"

    return True, "Centering correct"


# ============================================================================
# LDA Projection Tests
# ============================================================================

def test_lda_fit_shape():
    from dimensionality_reduction import LDAProjection
    iris = load_iris()
    X, y = iris.data, iris.target
    lda = LDAProjection(n_components=2).fit(X, y)
    if lda.components_.shape != (2, 4):
        return False, f"components_ shape: expected (2,4), got {lda.components_.shape}"
    return True, "Shape correct"


def test_lda_transform_shape():
    from dimensionality_reduction import LDAProjection
    iris = load_iris()
    X, y = iris.data, iris.target
    lda = LDAProjection(n_components=2).fit(X, y)
    X_proj = lda.transform(X)
    if X_proj.shape != (150, 2):
        return False, f"Projected shape: expected (150,2), got {X_proj.shape}"
    return True, "Shape correct"


def test_lda_max_components():
    from dimensionality_reduction import LDAProjection
    iris = load_iris()
    X, y = iris.data, iris.target
    lda = LDAProjection().fit(X, y)
    max_possible = min(X.shape[1], len(np.unique(y)) - 1)
    if lda.n_components_ > max_possible:
        return False, f"n_components_={lda.n_components_} > max possible {max_possible}"
    return True, f"n_components_={lda.n_components_} <= {max_possible}"


def test_lda_class_separation():
    from dimensionality_reduction import LDAProjection
    iris = load_iris()
    X, y = iris.data, iris.target
    lda = LDAProjection(n_components=2).fit(X, y)
    X_proj = lda.transform(X)

    # Check that projected classes have different means
    class_means = [X_proj[y == c].mean(axis=0) for c in np.unique(y)]
    pairwise_dists = []
    for i in range(len(class_means)):
        for j in range(i+1, len(class_means)):
            pairwise_dists.append(np.linalg.norm(class_means[i] - class_means[j]))

    min_dist = min(pairwise_dists)
    if min_dist < 0.1:
        return False, f"Classes not well separated; min pairwise dist: {min_dist:.4f}"
    return True, f"Classes separated; min pairwise dist: {min_dist:.2f}"


def test_lda_matches_sklearn():
    from dimensionality_reduction import LDAProjection
    iris = load_iris()
    X, y = iris.data, iris.target

    lda_ours = LDAProjection(n_components=2).fit(X, y)
    lda_sk = SklearnLDA(n_components=2).fit(X, y)

    X_ours = lda_ours.transform(X)
    X_sk = lda_sk.transform(X)

    for i in range(2):
        corr = abs(np.corrcoef(X_ours[:, i], X_sk[:, i])[0, 1])
        if corr < 0.95:
            return False, f"Component {i} correlation with sklearn: {corr:.4f}"

    return True, "Matches sklearn (up to sign)"


# ============================================================================
# Run all tests
# ============================================================================

def main():
    print("=" * 65)
    print("HW3 Autograder: Unsupervised Learning & Dimensionality Reduction")
    print("=" * 65)

    tests = [
        # K-Means (5 tests)
        ("KMeans: fit shapes",          test_kmeans_fit_shape,          2),
        ("KMeans: predict shapes",      test_kmeans_predict_shape,      1),
        ("KMeans: convergence",         test_kmeans_convergence,        1),
        ("KMeans: known clusters",      test_kmeans_known_clusters,     2),
        ("KMeans: matches sklearn",     test_kmeans_matches_sklearn,    2),

        # GMM (8 tests)
        ("GMM: responsibilities shape",   test_gmm_responsibilities_shape,     1),
        ("GMM: responsibilities sum=1",   test_gmm_responsibilities_sum_to_one, 2),
        ("GMM: responsibilities >= 0",    test_gmm_responsibilities_nonnegative, 1),
        ("GMM: fit runs",                 test_gmm_fit_runs,                    2),
        ("GMM: weights sum=1",            test_gmm_weights_sum_to_one,          1),
        ("GMM: known clusters",           test_gmm_known_clusters,              2),
        ("GMM: log-lik increases",        test_gmm_log_likelihood_increases,    2),
        ("GMM: predict shapes",           test_gmm_predict_shape,               1),

        # PCA (7 tests)
        ("PCA: fit shapes",              test_pca_fit_shape,             1),
        ("PCA: transform shape",         test_pca_transform_shape,       1),
        ("PCA: variance sorted",         test_pca_variance_sorted,       1),
        ("PCA: variance sums to 1",      test_pca_variance_sum,          1),
        ("PCA: threshold selection",      test_pca_threshold,             2),
        ("PCA: centering",               test_pca_centering,             1),
        ("PCA: matches sklearn",         test_pca_matches_sklearn,       2),

        # LDA Projection (5 tests)
        ("LDA: fit shapes",              test_lda_fit_shape,             1),
        ("LDA: transform shape",         test_lda_transform_shape,       1),
        ("LDA: max components",          test_lda_max_components,        1),
        ("LDA: class separation",        test_lda_class_separation,      2),
        ("LDA: matches sklearn",         test_lda_matches_sklearn,       2),
    ]

    results = []
    current_section = ""
    for name, fn, pts in tests:
        section = name.split(":")[0]
        if section != current_section:
            current_section = section
            print(f"\n--- {section} ---")
        result = run_test(name, fn, pts)
        results.append(result)
        print(result)

    # Summary
    total_earned = sum(r.points for r in results)
    total_possible = sum(r.max_points for r in results)
    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)

    print("\n" + "=" * 65)
    print(f"SUMMARY: {n_passed}/{n_total} tests passed")
    print(f"AUTOGRADER SCORE: {total_earned}/{total_possible} points")
    print("=" * 65)

    return total_earned, total_possible


if __name__ == "__main__":
    main()
