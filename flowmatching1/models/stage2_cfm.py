"""Stage 2: Core CFM velocity field training."""

import torch
import torch.nn as nn
from torch.optim import Adam
from .components.bezier import bezier_3point, bezier_velocity_3point
from .components.losses import CFMLosses


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for time."""
    
    def __init__(self, dim: int = 32):
        """
        Args:
            dim: embedding dimension (output will be 2*dim/2 = dim)
        """
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed time using sinusoidal encoding.
        
        Args:
            t: time, shape (B,) or (B, 1)
        
        Returns:
            t_emb: embedding, shape (B, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B,) → (B, 1)
        
        # Frequency scaling
        half = self.dim // 2
        freq = torch.arange(half, device=t.device, dtype=torch.float32)
        freq = 1.0 / (10000 ** (freq / half))
        
        # Apply sin and cos
        t_scaled = t * freq  # (B, half)
        t_emb = torch.cat([t_scaled.sin(), t_scaled.cos()], dim=-1)  # (B, dim)
        
        # Project
        return self.proj(t_emb)


class VelocityMLP(nn.Module):
    """Simple MLP to predict velocity field from pre-concatenated input."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = None, n_layers: int = 3, use_layer_norm: bool = True):
        """
        Args:
            input_dim: dimension of pre-concatenated input (e.g., [x, t_emb] or [x, t_emb, c])
            hidden_dim: hidden dimension
            output_dim: output dimension (velocity)
            n_layers: number of hidden layers
            use_layer_norm: whether to add LayerNorm for stability
        """
        super().__init__()
        
        if output_dim is None:
            output_dim = input_dim - 1  # x dimension
        
        # Build MLP layers with optional LayerNorm
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity from pre-concatenated input.
        
        Args:
            x: pre-concatenated input, shape (B, input_dim)
        
        Returns:
            v: velocity, shape (B, output_dim)
        """
        return self.net(x)


class CFMModel(nn.Module):
    """Unified Flow Matching model with factorized velocity field and Bézier anchors."""
    
    def __init__(
        self,
        input_dim: int,
        context_dim: int = 5,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_contexts: int = 1,
    ):
        """
        Args:
            input_dim: dimension of expression data
            context_dim: dimension of context encoding (drug + cell-line + dose)
            hidden_dim: hidden dimension of MLP
            n_layers: number of hidden layers
            n_contexts: number of context-specific heads
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.n_contexts = n_contexts
        
        # Time embedding
        time_emb_dim = 32
        self.time_embedding = SinusoidalTimeEmbedding(dim=time_emb_dim)
        
        # Shared velocity component: v_shared(x, t_emb)
        # Input: [x (D), t_emb (32)] -- no context!
        self.v_shared = VelocityMLP(
            input_dim=input_dim + time_emb_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            n_layers=n_layers,
            use_layer_norm=True,
        )
        
        # Context-specific heads: v_context_k(x, t_emb, c)
        # One head per context type
        self.v_context_heads = nn.ModuleList([
            VelocityMLP(
                input_dim=input_dim + time_emb_dim + context_dim,  # Include time embedding and context
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                n_layers=n_layers,
                use_layer_norm=True,
            )
            for _ in range(n_contexts)
        ])
        
        # Dynamic context weights: α(c) → softmax weights for context heads
        self.alpha_weights = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_contexts),
        )
        
        # Loss function
        self.loss_fn = CFMLosses(lambda_geom=0.1, lambda_context=0.01, lambda_smooth=0.001)
    
    def _forward_internal(self, x: torch.Tensor, t_emb: torch.Tensor, c: torch.Tensor):
        """
        Internal forward using pre-computed time embedding.
        Returns (v, v_ctx_outputs) to avoid recomputing context heads in training_step.
        """
        x_t_emb = torch.cat([x, t_emb], dim=-1)
        v_shared = self.v_shared(x_t_emb)

        alpha = torch.softmax(self.alpha_weights(c), dim=-1)  # (B, n_contexts)

        x_t_emb_c = torch.cat([x, t_emb, c], dim=-1)
        v_context_sum = torch.zeros_like(v_shared)
        v_ctx_outputs = []
        for k, context_head in enumerate(self.v_context_heads):
            v_ctx_k = context_head(x_t_emb_c)
            v_ctx_outputs.append(v_ctx_k)
            v_context_sum = v_context_sum + alpha[:, k:k+1] * v_ctx_k

        return v_shared + v_context_sum, v_ctx_outputs

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity: v_θ(x, t, c) = v_shared(x, t) + Σ_k α_k(c) * v_context_k(x, t, c)

        Args:
            x: expression, shape (B, D)
            t: time, shape (B, 1)
            c: context encoding, shape (B, context_dim)

        Returns:
            v: velocity, shape (B, D)
        """
        t_emb = self.time_embedding(t)
        v, _ = self._forward_internal(x, t_emb, c)
        return v
    
    def training_step(
        self,
        x0: torch.Tensor,
        x_T: torch.Tensor,
        x1: torch.Tensor,
        t_samples: torch.Tensor,
        c: torch.Tensor,
        s_t: torch.Tensor = None,
    ) -> dict:
        """
        Single training step with factorized velocity field.
        
        Args:
            x0: anchor P₀, shape (B, D)
            x_T: anchor P_T, shape (B, D)
            x1: anchor P₁, shape (B, D)
            t_samples: time samples, shape (B,)
            c: context, shape (B, context_dim)
            s_t: transition score, shape (B,) - optional
        
        Returns:
            dict with loss components
        """
        B = x0.shape[0]
        
        # Ensure t_samples is 2D
        if t_samples.dim() == 1:
            t_samples = t_samples.unsqueeze(-1)
        
        # Compute Bézier path and target velocities
        x_t = bezier_3point(t_samples.squeeze(-1), x0, x_T, x1)  # (B, D)
        v_target = bezier_velocity_3point(t_samples.squeeze(-1), x0, x_T, x1)  # (B, D)

        # Single forward pass — reuse t_emb and context head outputs for all losses
        t_emb = self.time_embedding(t_samples)
        v_pred, v_ctx_outputs = self._forward_internal(x_t, t_emb, c)

        # L_flow: MSE between predicted and Bézier target velocities
        l_flow = self.loss_fn.l_flow(v_pred, v_target)

        # L_geom: penalize velocity in stable regions (low transition score)
        l_geom = torch.tensor(0.0, device=x0.device)
        if s_t is not None:
            l_geom = self.loss_fn.l_geom(x_t, s_t, v_pred)

        # L_context: sparsity regularization on context-specific velocity corrections.
        # Keeps v_context_k small so v_shared captures the dominant dynamics;
        # context heads provide drug/cell-line residuals only.
        v_ctx_l2 = torch.stack([v.pow(2).mean() for v in v_ctx_outputs]).mean()
        l_context = v_ctx_l2

        # L_smooth: Hutchinson estimator of ‖∂v/∂x‖_F².
        # E_ε[‖(∂v/∂x)ε‖²] = ‖∂v/∂x‖_F² when ε ~ N(0,I).
        eps = 1e-4
        noise = torch.randn_like(x_t)
        v_pred_eps = self._forward_internal(x_t + eps * noise, t_emb, c)[0]
        jvp = (v_pred_eps - v_pred) / eps
        l_smooth = jvp.pow(2).mean()
        
        # Total loss
        total_loss = self.loss_fn.total_loss(l_flow, l_geom, l_context, l_smooth)
        
        return {
            "total_loss": total_loss,
            "l_flow": l_flow.item(),
            "l_geom": l_geom.item(),
            "l_context": l_context.item(),
            "l_smooth": l_smooth.item(),
        }


class CFMTrainer:
    """Trainer for CFM model."""
    
    def __init__(self, model: CFMModel, learning_rate: float = 1e-3):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.loss_history = []
    
    def train_epoch(self, dataloader, device: str = 'cpu') -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: yields (x0, x_T, x1, t, c, s_t)
            device: 'cpu' or 'cuda'
        
        Returns:
            average loss for epoch
        """
        self.model.train()
        self.model.to(device)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            x0, x_T, x1, t, c, s_t = batch
            
            x0 = x0.to(device)
            x_T = x_T.to(device)
            x1 = x1.to(device)
            t = t.to(device)
            c = c.to(device)
            if s_t is not None:
                s_t = s_t.to(device)
            
            # Forward pass
            loss_dict = self.model.training_step(x0, x_T, x1, t, c, s_t)
            loss = loss_dict["total_loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def eval(self, dataloader, device: str = 'cpu') -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        self.model.to(device)
        
        eval_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x0, x_T, x1, t, c, s_t = batch
                
                x0 = x0.to(device)
                x_T = x_T.to(device)
                x1 = x1.to(device)
                t = t.to(device)
                c = c.to(device)
                if s_t is not None:
                    s_t = s_t.to(device)
                
                loss_dict = self.model.training_step(x0, x_T, x1, t, c, s_t)
                loss = loss_dict["total_loss"]
                
                eval_loss += loss.item()
                n_batches += 1
        
        avg_loss = eval_loss / n_batches
        return avg_loss


if __name__ == "__main__":
    # Simple test
    model = CFMModel(input_dim=2000, context_dim=5)
    
    B = 32  # batch size
    D = 2000  # expression dim
    C = 5  # context dim
    
    x0 = torch.randn(B, D)
    x_T = torch.randn(B, D)
    x1 = torch.randn(B, D)
    t = torch.rand(B, 1)
    c = torch.randn(B, C)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    loss_dict = model.training_step(x0, x_T, x1, t, c)
    print(f"Loss dict: {loss_dict}")
