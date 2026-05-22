"""ODE solvers for flow matching inference."""

import torch
import numpy as np
from typing import Callable


class SimpleODESolver:
    """Simple ODE integrator for trajectory reconstruction."""
    
    @staticmethod
    def euler_integration(
        v_net: torch.nn.Module,
        x0: torch.Tensor,
        c: torch.Tensor,
        t_span: np.ndarray = None,
        num_steps: int = 50,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """
        Euler method ODE integration: dx/dt = v_θ(x, t, c)
        
        Args:
            v_net: velocity network
            x0: initial condition, shape (B, D)
            c: context, shape (B, context_dim)
            t_span: time points to evaluate, default linspace(0, 1, num_steps)
            num_steps: number of integration steps
            device: 'cpu' or 'cuda'
        
        Returns:
            trajectory: shape (T, B, D) where T = len(t_span)
        """
        if t_span is None:
            t_span = np.linspace(0, 1, num_steps)
        
        v_net.eval()
        v_net.to(device)
        
        x0 = x0.to(device)
        c = c.to(device)
        
        trajectory = [x0.detach().cpu().numpy()]
        x = x0
        
        dt = t_span[1] - t_span[0] if len(t_span) > 1 else 1.0 / num_steps
        
        with torch.no_grad():
            for t_idx in range(1, len(t_span)):
                t = torch.tensor([[t_span[t_idx - 1]]] * x.shape[0], device=device, dtype=torch.float32)
                
                # Velocity at current time
                v = v_net(x, t, c)  # (B, D)
                
                # Euler step
                x = x + dt * v
                
                trajectory.append(x.detach().cpu().numpy())
        
        # Stack: (T, B, D)
        trajectory = np.stack(trajectory, axis=0)
        
        return torch.from_numpy(trajectory).float()
    
    @staticmethod
    def rk4_integration(
        v_net: torch.nn.Module,
        x0: torch.Tensor,
        c: torch.Tensor,
        t_span: np.ndarray = None,
        num_steps: int = 50,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """
        RK4 (4th order Runge-Kutta) ODE integration.
        
        More accurate than Euler but slower.
        
        Args:
            v_net: velocity network
            x0: initial condition, shape (B, D)
            c: context, shape (B, context_dim)
            t_span: time points to evaluate
            num_steps: number of steps
            device: 'cpu' or 'cuda'
        
        Returns:
            trajectory: shape (T, B, D)
        """
        if t_span is None:
            t_span = np.linspace(0, 1, num_steps)
        
        v_net.eval()
        v_net.to(device)
        
        x0 = x0.to(device)
        c = c.to(device)
        
        trajectory = [x0.detach().cpu().numpy()]
        x = x0
        
        dt = t_span[1] - t_span[0] if len(t_span) > 1 else 1.0 / num_steps
        
        with torch.no_grad():
            for t_idx in range(1, len(t_span)):
                t_curr = t_span[t_idx - 1]
                t = torch.tensor([[t_curr]] * x.shape[0], device=device, dtype=torch.float32)
                
                # RK4 stages
                k1 = v_net(x, t, c)
                
                t_half = torch.tensor([[t_curr + 0.5 * dt]] * x.shape[0], device=device, dtype=torch.float32)
                k2 = v_net(x + 0.5 * dt * k1, t_half, c)
                
                k3 = v_net(x + 0.5 * dt * k2, t_half, c)
                
                t_next = torch.tensor([[t_curr + dt]] * x.shape[0], device=device, dtype=torch.float32)
                k4 = v_net(x + dt * k3, t_next, c)
                
                # RK4 update
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                
                trajectory.append(x.detach().cpu().numpy())
        
        trajectory = np.stack(trajectory, axis=0)
        
        return torch.from_numpy(trajectory).float()


class TrajectoryReconstructor:
    """Reconstruct single-cell trajectories from trained CFM model."""
    
    def __init__(self, cfm_model, device: str = 'cpu', solver: str = 'euler'):
        """
        Args:
            cfm_model: trained CFM model
            device: 'cpu' or 'cuda'
            solver: 'euler' or 'rk4'
        """
        self.cfm_model = cfm_model
        self.device = device
        self.solver = solver
    
    def reconstruct_from_P0(
        self,
        X_p0: np.ndarray,
        c: np.ndarray,
        num_steps: int = 100,
    ) -> dict:
        """
        Reconstruct trajectory starting from P0.
        
        Args:
            X_p0: initial P0 cells, shape (n_cells, n_genes)
            c: context encoding, shape (n_cells, context_dim)
            num_steps: number of integration steps
        
        Returns:
            dict with:
                - trajectory: shape (T, n_cells, n_genes)
                - t_span: time points, shape (T,)
        """
        X_tensor = torch.from_numpy(X_p0).float()
        c_tensor = torch.from_numpy(c).float()
        
        if self.solver == 'rk4':
            trajectory = SimpleODESolver.rk4_integration(
                self.cfm_model,  # Pass CFMModel directly (compatible with v(x, t, c) signature)
                X_tensor,
                c_tensor,
                num_steps=num_steps,
                device=self.device,
            )
        else:  # euler
            trajectory = SimpleODESolver.euler_integration(
                self.cfm_model,  # Pass CFMModel directly
                X_tensor,
                c_tensor,
                num_steps=num_steps,
                device=self.device,
            )
        
        t_span = np.linspace(0, 1, num_steps)
        
        return {
            "trajectory": trajectory.numpy(),
            "t_span": t_span,
        }


if __name__ == "__main__":
    # Test
    import torch.nn as nn
    
    class DummyVNet(nn.Module):
        def forward(self, x, t, c):
            return x * 0.1  # Simple drift
    
    v_net = DummyVNet()
    
    B, D, C = 10, 100, 5
    x0 = torch.randn(B, D)
    c = torch.randn(B, C)
    
    # Test Euler
    traj_euler = SimpleODESolver.euler_integration(v_net, x0, c, num_steps=20)
    print(f"Euler trajectory: {traj_euler.shape}")
    
    # Test RK4
    traj_rk4 = SimpleODESolver.rk4_integration(v_net, x0, c, num_steps=20)
    print(f"RK4 trajectory: {traj_rk4.shape}")
