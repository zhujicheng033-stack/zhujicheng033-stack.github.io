"""Stage 3: Post-hoc gene/pathway attribution."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class AttributionAnalyzer:
    r"""
    Stage 3: Gene attribution using Jacobian and Integrated Gradients.
    
    Computes three-way gene set decomposition:
    - G_flow ∩ G_DGV: core drivers
    - G_flow \ G_DGV: kinetic drivers (novel discoveries)
    - G_DGV \ G_flow: static markers
    """
    
    def __init__(self, top_n_genes: int = 100):
        self.top_n_genes = top_n_genes
    
    def compute_jacobian_importance(
        self,
        v_net: torch.nn.Module,
        X: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        device: str = 'cpu',
    ) -> np.ndarray:
        """
        Compute gene importance via vectorized Jacobian using torch.func.
        
        Much faster than per-gene finite differences!
        O(1) backward pass for all genes simultaneously.
        
        Args:
            v_net: velocity network
            X: expression matrix, shape (n_cells, n_genes)
            t: time, shape (n_cells, 1)
            c: context, shape (n_cells, context_dim)
            device: 'cpu' or 'cuda'
        
        Returns:
            importance: shape (n_genes,), importance scores per gene
        """
        try:
            from torch.func import jacrev, vmap
        except ImportError:
            # Fallback if torch.func not available (older PyTorch)
            return self._compute_jacobian_importance_fallback(v_net, X, t, c, device)
        
        v_net.eval()
        v_net.to(device)
        
        X = X.to(device)
        t = t.to(device)
        c = c.to(device)
        
        # Define function for single sample
        def single_forward(x, t_i, c_i):
            """Forward pass for single sample."""
            return v_net(x.unsqueeze(0), t_i.unsqueeze(0), c_i.unsqueeze(0)).squeeze(0)
        
        # Ensure t is shape (n_cells,) for vmap
        t_flat = t.squeeze(-1) if t.dim() > 1 else t  # (n_cells,)
        
        # For torch.func, we need to allow gradients even though we only care about Jacobian
        # Use clone() to avoid polluting original tensors
        X_req = X.clone().requires_grad_(True)
        t_flat_req = t_flat.clone().requires_grad_(True)
        c_req = c.clone().requires_grad_(True)
        
        # Compute Jacobian for each sample: ∂v/∂x, shape (n_cells, n_genes_out, n_genes_in)
        jacobian_fn = vmap(jacrev(single_forward))
        batch_jacobian = jacobian_fn(X_req, t_flat_req, c_req)  # (n_cells, n_genes, n_genes)
        
        # Aggregate: mean over cells and output genes
        importance = batch_jacobian.abs().mean(dim=(0, 1))  # (n_genes,)
        
        return importance.detach().cpu().numpy()
    
    def _compute_jacobian_importance_fallback(
        self,
        v_net: torch.nn.Module,
        X: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        device: str = 'cpu',
    ) -> np.ndarray:
        """
        Fallback: use finite differences if torch.func unavailable.
        """
        v_net.eval()
        v_net.to(device)
        
        X = X.to(device)
        t = t.to(device)
        c = c.to(device)
        
        n_genes = X.shape[1]
        
        # Compute baseline velocity
        with torch.no_grad():
            v_baseline = v_net(X, t, c)  # (n_cells, n_genes)
        
        # Compute importance via finite differences
        eps = 1e-4
        importance = np.zeros(n_genes)
        
        for gene_idx in range(n_genes):
            # Perturb gene expression
            X_perturbed = X.clone()
            X_perturbed[:, gene_idx] += eps
            
            # Forward pass with perturbation
            with torch.no_grad():
                v_perturbed = v_net(X_perturbed, t, c)  # (n_cells, n_genes)
            
            # Compute change in velocity (sum over output genes)
            delta_v = torch.abs(v_perturbed - v_baseline).sum(dim=1)  # (n_cells,)
            
            # Importance = average sensitivity to perturbation
            importance[gene_idx] = delta_v.mean().detach().cpu().item() / eps
        
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
    ) -> dict:
        """
        Full attribution analysis.
        
        Args:
            v_net: velocity network
            X: expression matrix, shape (n_cells, n_genes)
            t: time, shape (n_cells,) or (n_cells, 1)
            c: context, shape (n_cells, context_dim)
            state_labels: state assignments, shape (n_cells,)
            device: 'cpu' or 'cuda'
        
        Returns:
            dict with analysis results
        """
        # Ensure proper shapes
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        
        X_tensor = torch.from_numpy(X).float()
        t_tensor = torch.from_numpy(t).float()
        c_tensor = torch.from_numpy(c).float()
        
        # Compute importances
        print("Computing Jacobian importance...")
        importance_flow = self.compute_jacobian_importance(v_net, X_tensor, t_tensor, c_tensor, device)
        
        print("Computing static importance...")
        importance_static = self.compute_static_importance(X, state_labels)
        
        # Three-way decomposition
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
