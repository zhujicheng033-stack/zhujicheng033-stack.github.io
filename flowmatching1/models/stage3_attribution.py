"""Stage 3: Post-hoc gene/pathway attribution."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class AttributionAnalyzer:
    r"""
    Stage 3: Gene attribution using trajectory-integrated velocity magnitude.

    Core idea (CFM-native):
        importance_i = E_{t, x_t ~ trajectory} [ |v_i(x_t, t, c)| ]

    This directly answers "which gene is being pushed the most along the learned
    flow", which is the natural definition of a kinetic driver in flow matching.

    Computes three-way gene set decomposition:
    - G_flow ∩ G_DGV: core drivers   (high velocity AND high DE)
    - G_flow \ G_DGV: kinetic drivers (high velocity, not DE — novel kinetic signal)
    - G_DGV \ G_flow: static markers  (high DE, but low velocity — endpoint markers)
    """

    def __init__(self, top_n_genes: int = 100):
        self.top_n_genes = top_n_genes

    # ── Primary attribution: trajectory-integrated velocity magnitude ──────────

    def compute_trajectory_importance(
        self,
        v_net: torch.nn.Module,
        X_p0: np.ndarray,
        c: np.ndarray,
        n_steps: int = 50,
        device: str = 'cpu',
    ) -> np.ndarray:
        """
        Compute gene importance as the trajectory-averaged absolute velocity.

        importance_i = (1/T) Σ_t  mean_cells |v_i(x_t, t, c)|

        This is the natural CFM metric: genes with high |v_i| are the ones the
        velocity field is actively transporting — i.e., the kinetic drivers.

        Args:
            v_net:  trained CFM model
            X_p0:   P0 cells to start integration from, shape (n_cells, n_genes)
            c:      context for each cell, shape (n_cells, context_dim)
            n_steps: number of Euler integration steps
            device: 'cpu' or 'cuda'

        Returns:
            importance: shape (n_genes,)
        """
        v_net.eval()
        v_net.to(device)

        x = torch.from_numpy(X_p0).float().to(device)
        c_t = torch.from_numpy(c).float().to(device)

        t_span = np.linspace(0.0, 1.0, n_steps)
        dt = t_span[1] - t_span[0]

        accumulated = np.zeros(X_p0.shape[1])   # (n_genes,)

        with torch.no_grad():
            for t_val in t_span:
                t_tensor = torch.full((x.shape[0], 1), t_val, dtype=torch.float32, device=device)
                v = v_net(x, t_tensor, c_t)          # (n_cells, n_genes)

                # Accumulate mean absolute velocity per gene
                accumulated += v.abs().mean(dim=0).cpu().numpy()

                # Euler step to advance x along trajectory
                x = x + dt * v

        importance = accumulated / n_steps           # average over time steps
        return importance

    # ── Kept for reference / ablation; no longer used as the primary metric ───

    def compute_jacobian_importance(
        self,
        v_net: torch.nn.Module,
        X: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        device: str = 'cpu',
    ) -> np.ndarray:
        """
        Gene importance via ∂v/∂x Jacobian (kept for ablation).

        Note: this measures *network sensitivity* to each input gene, not the
        *magnitude of transport*. In practice it often fails to recover known
        driver genes because LayerNorm / skip connections flatten sensitivities.
        Prefer compute_trajectory_importance() for CFM models.
        """
        v_net.eval()
        v_net.to(device)

        X = X.to(device)
        t = t.to(device)
        c = c.to(device)

        n_genes = X.shape[1]
        eps = 1e-3
        importance = np.zeros(n_genes)

        with torch.no_grad():
            v_base = v_net(X, t, c)

        for gene_idx in range(n_genes):
            X_perturbed = X.clone()
            X_perturbed[:, gene_idx] += eps
            with torch.no_grad():
                v_pert = v_net(X_perturbed, t, c)
            delta_v = (v_pert - v_base).abs().mean(dim=1)
            importance[gene_idx] = delta_v.mean().item() / eps

        return importance
    
    def compute_static_importance(
        self,
        X: np.ndarray,
        state_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Compute static gene importance via DE analysis.
        
        Simulate DEG: genes with high variance across states.
        
        Args:
            X: expression matrix, shape (n_cells, n_genes)
            state_labels: state assignments, shape (n_cells,)
        
        Returns:
            importance: shape (n_genes,), DE scores
        """
        n_states = len(np.unique(state_labels))
        n_genes = X.shape[1]
        
        importance = np.zeros(n_genes)
        
        for gene_idx in range(n_genes):
            expr = X[:, gene_idx]
            
            # ANOVA-like: variance between states vs within states
            between_var = 0.0
            within_var = 0.0
            
            global_mean = expr.mean()
            
            for state_idx in range(n_states):
                mask = state_labels == state_idx
                state_expr = expr[mask]
                
                if len(state_expr) > 0:
                    # Between-state variance
                    between_var += len(state_expr) * (state_expr.mean() - global_mean) ** 2
                    
                    # Within-state variance
                    within_var += np.sum((state_expr - state_expr.mean()) ** 2)
            
            # F-score approximation
            if within_var > 1e-8:
                importance[gene_idx] = between_var / (within_var + 1e-8)
        
        return importance
    
    def three_way_decomposition(
        self,
        importance_flow: np.ndarray,
        importance_static: np.ndarray,
        percentile_flow: float = 90,
        percentile_static: float = 90,
    ) -> dict:
        """
        Decompose genes into three categories.
        
        Args:
            importance_flow: kinetic importance (Jacobian), shape (n_genes,)
            importance_static: static importance (DE), shape (n_genes,)
            percentile_flow: threshold for flow importance
            percentile_static: threshold for static importance
        
        Returns:
            dict with gene indices for each category
        """
        threshold_flow = np.percentile(importance_flow, percentile_flow)
        threshold_static = np.percentile(importance_static, percentile_static)
        
        high_flow = importance_flow > threshold_flow
        high_static = importance_static > threshold_static
        
        core_drivers = np.where(high_flow & high_static)[0]
        kinetic_drivers = np.where(high_flow & ~high_static)[0]
        static_markers = np.where(~high_flow & high_static)[0]
        
        return {
            "core_drivers": core_drivers,
            "kinetic_drivers": kinetic_drivers,
            "static_markers": static_markers,
        }
    
    def analyze(
        self,
        v_net: torch.nn.Module,
        X: np.ndarray,
        t: np.ndarray,
        c: np.ndarray,
        state_labels: np.ndarray,
        device: str = 'cpu',
        X_p0: np.ndarray = None,
        c_p0: np.ndarray = None,
    ) -> dict:
        """
        Full attribution analysis.

        Args:
            v_net:        trained CFM model
            X:            full expression matrix, shape (n_cells, n_genes)
            t:            per-cell pseudotime, shape (n_cells,) or (n_cells, 1)
            c:            per-cell context encoding, shape (n_cells, context_dim)
            state_labels: cluster assignments, shape (n_cells,)
            device:       'cpu' or 'cuda'
            X_p0:         P0 cells for trajectory integration, shape (n_p0, n_genes).
                          If None, uses cells with state_label == 0.
            c_p0:         context for P0 cells, shape (n_p0, context_dim).
                          If None, uses c[state_labels == 0].

        Returns:
            dict with importance arrays and decomposition
        """
        # ── Trajectory importance (primary) ───────────────────────────────────
        if X_p0 is None:
            p0_mask = state_labels == 0
            X_p0 = X[p0_mask]
            c_p0 = c[p0_mask]

        print("Computing trajectory-integrated velocity importance...")
        importance_flow = self.compute_trajectory_importance(
            v_net, X_p0, c_p0, n_steps=50, device=device
        )

        # ── Static (DE) importance ─────────────────────────────────────────────
        print("Computing static importance...")
        importance_static = self.compute_static_importance(X, state_labels)

        # ── Three-way decomposition ────────────────────────────────────────────
        decomposition = self.three_way_decomposition(importance_flow, importance_static)

        return {
            "importance_flow": importance_flow,
            "importance_static": importance_static,
            "decomposition": decomposition,
        }


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_cells = 500
    n_genes = 100
    
    X = np.random.randn(n_cells, n_genes)
    t = np.random.rand(n_cells)
    c = np.random.randn(n_cells, 5)
    state_labels = np.random.randint(0, 3, n_cells)
    
    # Simple dummy network
    class DummyNet(torch.nn.Module):
        def forward(self, x, t, c):
            return x  # Identity for testing
    
    v_net = DummyNet()
    
    analyzer = AttributionAnalyzer()
    result = analyzer.analyze(v_net, X, t, c, state_labels)
    
    print(f"Flow importance: {result['importance_flow'][:10]}")
    print(f"Static importance: {result['importance_static'][:10]}")
    print(f"Core drivers: {result['decomposition']['core_drivers'].shape}")
    print(f"Kinetic drivers: {result['decomposition']['kinetic_drivers'].shape}")
    print(f"Static markers: {result['decomposition']['static_markers'].shape}")
