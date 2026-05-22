"""Bézier curve utilities for three-anchor CFM."""

import torch
import numpy as np


def bezier_3point(t: torch.Tensor, x0: torch.Tensor, x_T: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Compute 3-point Bézier curve (quadratic Bézier).
    
    x(t) = (1-t)² x₀ + 2t(1-t) x_T + t² x₁
    
    Args:
        t: time tensor, shape (B,) or scalar in [0, 1]
        x0: anchor P₀ (initial state), shape (B, D) or (D,)
        x_T: anchor P_T (intermediate state), shape (B, D) or (D,)
        x1: anchor P₁ (final state), shape (B, D) or (D,)
    
    Returns:
        x_t: position at time t, same shape as x0
    """
    # Ensure t is proper shape
    if t.dim() == 0:
        t = t.unsqueeze(0)
    
    # Reshape for broadcasting if needed
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)
        x_T = x_T.unsqueeze(0)
        x1 = x1.unsqueeze(0)
    
    # Compute Bézier basis
    one_minus_t = 1.0 - t
    
    # (1-t)²
    coeff0 = one_minus_t ** 2  # (B,)
    # 2t(1-t)
    coeff_T = 2.0 * t * one_minus_t  # (B,)
    # t²
    coeff1 = t ** 2  # (B,)
    
    # Reshape coefficients for broadcasting: (B, 1) to broadcast with (B, D)
    coeff0 = coeff0.unsqueeze(-1)
    coeff_T = coeff_T.unsqueeze(-1)
    coeff1 = coeff1.unsqueeze(-1)
    
    x_t = coeff0 * x0 + coeff_T * x_T + coeff1 * x1
    
    return x_t


def bezier_velocity_3point(t: torch.Tensor, x0: torch.Tensor, x_T: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Compute velocity (derivative) of 3-point Bézier curve.
    
    dx/dt = 2(1-t)(x_T - x₀) + 2t(x₁ - x_T)
    
    Args:
        t: time tensor, shape (B,) or scalar
        x0, x_T, x1: anchors, shape (B, D) or (D,)
    
    Returns:
        v_t: velocity, same shape as x0
    """
    if t.dim() == 0:
        t = t.unsqueeze(0)
    
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)
        x_T = x_T.unsqueeze(0)
        x1 = x1.unsqueeze(0)
    
    one_minus_t = 1.0 - t
    
    # 2(1-t)
    coeff0 = (2.0 * one_minus_t).unsqueeze(-1)
    # 2t
    coeff1 = (2.0 * t).unsqueeze(-1)
    
    v_t = coeff0 * (x_T - x0) + coeff1 * (x1 - x_T)
    
    return v_t


def sample_bezier_path(t_samples: torch.Tensor, x0: torch.Tensor, x_T: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Sample multiple points along Bézier curve.
    
    Args:
        t_samples: time points to sample, shape (T,)
        x0, x_T, x1: anchors
    
    Returns:
        positions: shape (T, B, D)
    """
    positions = []
    for t in t_samples:
        x_t = bezier_3point(t, x0, x_T, x1)
        positions.append(x_t)
    
    return torch.stack(positions, dim=0)


if __name__ == "__main__":
    # Test Bézier curve
    torch.manual_seed(42)
    
    B, D = 4, 10  # batch, dimension
    x0 = torch.randn(B, D)
    x_T = torch.randn(B, D)
    x1 = torch.randn(B, D)
    
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    
    for ti in t:
        x_t = bezier_3point(ti, x0, x_T, x1)
        v_t = bezier_velocity_3point(ti, x0, x_T, x1)
        print(f"t={ti:.2f} | x_t shape: {x_t.shape}, v_t shape: {v_t.shape}")
    
    # Verify endpoints
    x_0_check = bezier_3point(torch.tensor(0.0), x0, x_T, x1)
    x_1_check = bezier_3point(torch.tensor(1.0), x0, x_T, x1)
    print(f"\nEndpoint check:")
    print(f"  x(0) == x0: {torch.allclose(x_0_check, x0, atol=1e-5)}")
    print(f"  x(1) == x1: {torch.allclose(x_1_check, x1, atol=1e-5)}")
