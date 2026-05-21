"""Loss functions for unified flow matching."""

import torch
import torch.nn as nn


class CFMLosses(nn.Module):
    """Collection of loss functions for CFM training."""
    
    def __init__(self, lambda_geom=0.1, lambda_context=0.01, lambda_smooth=0.001):
        """
        Args:
            lambda_geom: weight for L_geom
            lambda_context: weight for L_context
            lambda_smooth: weight for L_smooth
        """
        super().__init__()
        self.lambda_geom = lambda_geom
        self.lambda_context = lambda_context
        self.lambda_smooth = lambda_smooth
    
    def l_flow(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        L_flow: MSE between predicted and target velocities along Bézier path.
        
        Args:
            v_pred: predicted velocity, shape (B, D)
            v_target: target (Bézier) velocity, shape (B, D)
            mask: optional mask, shape (B,)
        
        Returns:
            scalar loss
        """
        per_sample = ((v_pred - v_target) ** 2).mean(dim=-1)  # (B,)

        if mask is not None:
            loss = (per_sample * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = per_sample.mean()

        return loss
    
    def l_geom(
        self,
        x_t: torch.Tensor,
        s_t: torch.Tensor,
        v_t: torch.Tensor,
        w_t: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        L_geom: Geometric constraint penalizing velocity in stable regions.
        
        E_t[ w(t) · (1 - s(x_t))² ]
        
        When s(x_t) is high (in transition region), penalty is low.
        When s(x_t) is low (in stable region), we penalize high velocity.
        
        Args:
            x_t: positions, shape (B, D)
            s_t: transition score in [0, 1], shape (B,)
            v_t: velocities, shape (B, D)
            w_t: time-dependent weight, shape (B,)
        
        Returns:
            scalar loss
        """
        # Penalty increases as (1 - s_t)² (higher in stable regions)
        penalty = (1.0 - s_t) ** 2  # (B,)
        
        # Scale by velocity magnitude to discourage motion in stable regions
        velocity_norm = torch.norm(v_t, dim=-1)  # (B,)
        
        loss = penalty * velocity_norm  # (B,)
        
        if w_t is not None:
            loss = (loss * w_t).mean()
        else:
            loss = loss.mean()
        
        return loss
    
    def l_context(
        self,
        v_context_list: list,
    ) -> torch.Tensor:
        """
        L_context: L2 magnitude regularization on context-specific velocity corrections.
        
        This encourages context heads to be sparse — their job is to provide small,
        context-specific adjustments to the shared velocity field, not to dominate it.
        
        v_θ(x, t, c) = v_shared(x, t) + Σ_k α_k(c) * v_context_k(x, t, c)
        
        By regularizing ‖v_context_k‖², we ensure context heads stay small and
        let v_shared capture the dominant dynamics (mixture-of-experts pattern).
        
        Args:
            v_context_list: list of context-specific velocity components, each (B, D)
        
        Returns:
            scalar loss (sum of squared magnitudes across context heads)
        """
        if not v_context_list or len(v_context_list) == 0:
            return torch.tensor(0.0)
        
        # Sum of L2 norms across all context heads and samples
        v_ctx_l2 = torch.stack([v.pow(2).mean() for v in v_context_list]).mean()
        
        return v_ctx_l2
    
    def l_smooth(
        self,
        jacobian: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_smooth: Smoothness regularization using Jacobian.
        
        ‖∂v_θ/∂x‖_F² (Frobenius norm)
        
        To avoid collapsing velocities, we normalize:
        ‖∂v_θ/∂x‖_F² / (‖v_θ‖² + ε)
        
        Args:
            jacobian: Jacobian matrix ∂v/∂x, shape (B, D, D)
        
        Returns:
            scalar loss
        """
        # Frobenius norm per batch element: sqrt(sum of squares)
        jacobian_norm = torch.norm(jacobian, p='fro', dim=(1, 2))  # (B,)
        
        loss = jacobian_norm.mean()
        
        return loss
    
    def l_smooth_normalized(
        self,
        jacobian: torch.Tensor,
        v_norm: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Normalized smoothness: avoid collapse.
        
        Args:
            jacobian: shape (B, D, D)
            v_norm: velocity norm, shape (B,)
            eps: numerical stability
        
        Returns:
            scalar loss
        """
        jacobian_norm = torch.norm(jacobian, p='fro', dim=(1, 2))  # (B,)
        normalized = jacobian_norm / (v_norm + eps)
        
        loss = normalized.mean()
        
        return loss
    
    def total_loss(
        self,
        l_flow_val: torch.Tensor,
        l_geom_val: torch.Tensor,
        l_context_val: torch.Tensor,
        l_smooth_val: torch.Tensor,
    ) -> torch.Tensor:
        """
        Total weighted loss.
        
        L_total = L_flow + λ₁·L_geom + λ₂·L_context + λ₃·L_smooth
        """
        total = (
            l_flow_val
            + self.lambda_geom * l_geom_val
            + self.lambda_context * l_context_val
            + self.lambda_smooth * l_smooth_val
        )
        
        return total


if __name__ == "__main__":
    # Test losses
    losses = CFMLosses(lambda_geom=0.1, lambda_context=0.01, lambda_smooth=0.001)
    
    B, D = 32, 100
    
    # Test L_flow
    v_pred = torch.randn(B, D)
    v_target = torch.randn(B, D)
    l_flow = losses.l_flow(v_pred, v_target)
    print(f"L_flow: {l_flow:.4f}")
    
    # Test L_geom
    x_t = torch.randn(B, D)
    s_t = torch.sigmoid(torch.randn(B))  # [0, 1]
    v_t = torch.randn(B, D)
    l_geom = losses.l_geom(x_t, s_t, v_t)
    print(f"L_geom: {l_geom:.4f}")
    
    # Test L_context
    v_shared = torch.randn(B, D)
    v_context_list = [torch.randn(B, D) for _ in range(3)]
    l_context = losses.l_context(v_shared, v_context_list)
    print(f"L_context: {l_context:.4f}")
    
    # Test L_smooth
    jacobian = torch.randn(B, D, D)
    l_smooth = losses.l_smooth(jacobian)
    print(f"L_smooth: {l_smooth:.4f}")
    
    # Test total
    total = losses.total_loss(l_flow, l_geom, l_context, l_smooth)
    print(f"L_total: {total:.4f}")
