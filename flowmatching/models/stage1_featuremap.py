"""Stage 1: Simplified manifold extraction (placeholder for FeatureMAP)."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch


class SimpleManifoldExtractor:
    """
    Simplified Stage 1: extract state labels (P0, P_T, P1) and transition scores.
    
    In production, this would use FeatureMAP. Here we use:
    - PCA for dimensionality reduction
    - KMeans for state clustering
    - Distance-to-cluster center for transition score
    """
    
    def __init__(self, n_components: int = 20, n_states: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.n_states = n_states
        self.random_state = random_state
        
        self.pca = None
        self.kmeans = None
        self.scaler = None
        self.state_labels = None
        self.transition_scores = None
    
    def fit(self, X: np.ndarray) -> 'SimpleManifoldExtractor':
        """
        Fit manifold extractor.
        
        Args:
            X: expression matrix, shape (n_cells, n_genes)
        
        Returns:
            self
        """
        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA for dimensionality reduction
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # KMeans to identify states
        self.kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state)
        self.kmeans.fit(X_pca)
        
        return self
    
    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """
        Predict state labels for new data.
        
        Args:
            X: shape (n_cells, n_genes)
        
        Returns:
            state_labels: shape (n_cells,), values in {0, 1, 2}
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        state_labels = self.kmeans.predict(X_pca)
        
        return state_labels
    
    def compute_transition_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute transition score s(x) = d1/d2 for each cell.
        
        Semantics:
        - Stable cells near cluster center: d1 << d2 → score ≈ 0 (LOW)
        - Transition cells between clusters: d1 ≈ d2 → score ≈ 1 (HIGH)
        
        Args:
            X: shape (n_cells, n_genes)
        
        Returns:
            scores: shape (n_cells,), values in [0, 1]
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Compute distances to all cluster centers (n_cells, n_states)
        distances = np.linalg.norm(X_pca[:, np.newaxis, :] - self.kmeans.cluster_centers_, axis=2)
        
        # Sort distances per cell (ascending)
        sorted_dists = np.sort(distances, axis=1)  # (n_cells, n_states)
        
        d1 = sorted_dists[:, 0]  # nearest cluster
        d2 = sorted_dists[:, 1]  # second nearest cluster
        
        # Transition score: measure how close cell is to cluster boundary
        # When cell is between two clusters: d1 ≈ d2 → ratio ≈ 1 → score HIGH (TRANSITION)
        # When cell is in stable cluster: d1 << d2 → ratio << 1 → score LOW (STABLE)
        
        scores = d1 / (d2 + 1e-8)
        scores = np.clip(scores, 0, 1)
        
        return scores
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform to PCA space.
        
        Args:
            X: shape (n_cells, n_genes)
        
        Returns:
            X_pca: shape (n_cells, n_components)
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca


class Stage1Pipeline:
    """Full Stage 1 pipeline."""
    
    def __init__(self, n_components: int = 20, n_states: int = 3):
        self.extractor = SimpleManifoldExtractor(
            n_components=n_components,
            n_states=n_states,
        )
        self.state_labels = None
        self.transition_scores = None
    
    def fit_predict(self, X: np.ndarray) -> dict:
        """
        Fit and predict states.
        
        Args:
            X: expression matrix, shape (n_cells, n_genes)
        
        Returns:
            dict with:
                - state_labels: shape (n_cells,)
                - transition_scores: shape (n_cells,)
                - X_pca: shape (n_cells, n_components)
        """
        self.extractor.fit(X)
        
        self.state_labels = self.extractor.predict_states(X)
        self.transition_scores = self.extractor.compute_transition_scores(X)
        X_pca = self.extractor.transform(X)
        
        return {
            "state_labels": self.state_labels,
            "transition_scores": self.transition_scores,
            "X_pca": X_pca,
        }


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    
    n_cells = 2000
    n_genes = 2000
    X = np.random.randn(n_cells, n_genes)
    
    pipeline = Stage1Pipeline(n_components=20, n_states=3)
    result = pipeline.fit_predict(X)
    
    print(f"State labels: {result['state_labels']}")
    print(f"Transition scores: {result['transition_scores']}")
    print(f"X_pca shape: {result['X_pca'].shape}")
    print(f"State distribution: {np.bincount(result['state_labels'])}")
