"""Validation metrics."""

import numpy as np
from scipy.spatial.distance import cdist


class ValidationMetrics:
    """Compute validation metrics for CFM model."""
    
    @staticmethod
    def mmd(X: np.ndarray, Y: np.ndarray, kernel: str = 'rbf', gamma: float = None) -> float:
        """
        Maximum Mean Discrepancy (MMD).

        MMD²(X, Y) = E[φ(x)] - E[φ(y)]²

        Args:
            X: distribution 1, shape (n1, d)
            Y: distribution 2, shape (n2, d)
            kernel: 'rbf' or 'linear'
            gamma: bandwidth for RBF kernel. If None, uses the median heuristic:
                   gamma = 1 / (2 * median_pairwise_dist²). This is critical for
                   high-dimensional data (e.g. d=2000) where a fixed gamma=1.0
                   makes exp(-gamma*||x-y||²) → 0 for all cross-pairs, collapsing
                   MMD to sqrt(1/n + 1/m) regardless of distributions.

        Returns:
            mmd_value: non-negative float
        """
        n, m = X.shape[0], Y.shape[0]

        if kernel == 'rbf':
            # Subsample for bandwidth estimation (expensive otherwise)
            n_sub = min(500, n, m)
            rng = np.random.default_rng(0)
            X_sub = X[rng.choice(n, n_sub, replace=False)]
            Y_sub = Y[rng.choice(m, n_sub, replace=False)]

            if gamma is None:
                # Median heuristic: sigma = median(pairwise distances), gamma = 1/(2*sigma²)
                XY = np.vstack([X_sub, Y_sub])
                dists = cdist(XY, XY)
                # Use upper triangle (excluding diagonal)
                upper = dists[np.triu_indices(len(XY), k=1)]
                median_dist = np.median(upper)
                if median_dist < 1e-8:
                    median_dist = 1.0  # fallback
                gamma = 1.0 / (2.0 * median_dist ** 2)

            # RBF kernel: exp(-gamma * ||x - y||²)
            K_XX = np.exp(-gamma * cdist(X, X) ** 2)
            K_YY = np.exp(-gamma * cdist(Y, Y) ** 2)
            K_XY = np.exp(-gamma * cdist(X, Y) ** 2)
        else:  # linear
            K_XX = X @ X.T
            K_YY = Y @ Y.T
            K_XY = X @ Y.T
        
        mmd2 = (1.0 / (n * n) * K_XX.sum() +
                1.0 / (m * m) * K_YY.sum() -
                2.0 / (n * m) * K_XY.sum())
        
        return np.sqrt(np.maximum(mmd2, 0.0))
    
    @staticmethod
    def wasserstein_2_approx(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Approximate Wasserstein-2 distance (sliced Wasserstein).
        
        Sample random projections and compute 1D Wasserstein.
        
        Args:
            X: distribution 1, shape (n1, d)
            Y: distribution 2, shape (n2, d)
        
        Returns:
            w2_approx: float
        """
        d = X.shape[1]
        n_projections = 100
        
        w2_values = []
        
        for _ in range(n_projections):
            # Random projection direction
            theta = np.random.randn(d)
            theta = theta / np.linalg.norm(theta)
            
            # Project
            x_proj = X @ theta
            y_proj = Y @ theta
            
            # 1D Wasserstein (sort and compare)
            x_sorted = np.sort(x_proj)
            y_sorted = np.sort(y_proj)
            
            # Interpolate to same size
            min_size = min(len(x_sorted), len(y_sorted))
            x_sorted = x_sorted[:min_size]
            y_sorted = y_sorted[:min_size]
            
            w1 = np.mean((x_sorted - y_sorted) ** 2)
            w2_values.append(w1)
        
        return np.sqrt(np.mean(w2_values))
    
    @staticmethod
    def trajectory_correlation(t_pred: np.ndarray, t_true: np.ndarray) -> float:
        """
        Correlation between predicted and true pseudotime.
        
        Args:
            t_pred: predicted pseudotime, shape (n_cells,)
            t_true: true pseudotime, shape (n_cells,)
        
        Returns:
            correlation: float in [-1, 1]
        """
        return np.corrcoef(t_pred, t_true)[0, 1]
    
    @staticmethod
    def state_purity(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Purity of state assignment.
        
        For each predicted state, find the most common true state.
        Purity = fraction of correctly assigned cells.
        
        Args:
            pred_labels: predicted state, shape (n_cells,)
            true_labels: true state, shape (n_cells,)
        
        Returns:
            purity: float in [0, 1]
        """
        n_pred_states = len(np.unique(pred_labels))
        n_true_states = len(np.unique(true_labels))
        
        confusion = np.zeros((n_pred_states, n_true_states))
        
        for i in range(n_pred_states):
            for j in range(n_true_states):
                confusion[i, j] = ((pred_labels == i) & (true_labels == j)).sum()
        
        # Greedy matching
        purity = 0.0
        used_true = set()
        
        for i in range(n_pred_states):
            best_j = None
            best_count = 0
            
            for j in range(n_true_states):
                if j not in used_true and confusion[i, j] > best_count:
                    best_j = j
                    best_count = confusion[i, j]
            
            if best_j is not None:
                purity += best_count
                used_true.add(best_j)
        
        return purity / len(pred_labels)
    
    @staticmethod
    def gene_recovery(
        pred_important_genes: np.ndarray,
        true_important_genes: np.ndarray,
    ) -> dict:
        """
        Evaluate gene discovery.
        
        Args:
            pred_important_genes: set of predicted important gene indices
            true_important_genes: set of true important gene indices
        
        Returns:
            dict with metrics
        """
        pred_set = set(pred_important_genes)
        true_set = set(true_important_genes)
        
        intersection = len(pred_set & true_set)
        union = len(pred_set | true_set)
        
        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = intersection / len(true_set) if len(true_set) > 0 else 0.0
        
        iou = intersection / union if union > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "f1": f1,
        }


def compute_metrics(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    t_pred: np.ndarray = None,
    t_true: np.ndarray = None,
) -> dict:
    """
    Compute a comprehensive set of metrics.
    
    Args:
        X_pred: predicted distribution, shape (n, d)
        X_true: true distribution, shape (n, d)
        t_pred: predicted pseudotime (optional)
        t_true: true pseudotime (optional)
    
    Returns:
        dict with all metrics
    """
    metrics = ValidationMetrics()
    
    results = {
        "mmd": metrics.mmd(X_pred, X_true),
        "wasserstein_2": metrics.wasserstein_2_approx(X_pred, X_true),
    }
    
    if t_pred is not None and t_true is not None:
        results["trajectory_correlation"] = metrics.trajectory_correlation(t_pred, t_true)
    
    return results


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    
    n = 500
    d = 100
    
    X_true = np.random.randn(n, d)
    X_pred = X_true + np.random.randn(n, d) * 0.1  # Small perturbation
    
    metrics = compute_metrics(X_pred, X_true)
    print(f"Metrics: {metrics}")
    
    # Gene recovery test
    true_genes = np.array([0, 1, 2, 3, 4])
    pred_genes = np.array([1, 2, 3, 4, 5])
    
    gene_metrics = ValidationMetrics.gene_recovery(pred_genes, true_genes)
    print(f"Gene recovery: {gene_metrics}")
