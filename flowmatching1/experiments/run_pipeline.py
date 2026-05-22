"""Main pipeline runner: connect all stages."""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.synthetic_data_generator import generate_toy_data
from models.stage1_featuremap import Stage1Pipeline
from models.stage2_cfm import CFMModel, CFMTrainer
from models.stage3_attribution import AttributionAnalyzer
from models.ode_solver import TrajectoryReconstructor
from models.context_encoder import ContextEncoder, ConditionVocabulary
from validation.metrics import compute_metrics, ValidationMetrics


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """
    Stage 0: Generate toy data.
    
    Returns:
        adata: AnnData object with expression data
    """
    print("\n" + "="*60)
    print("STAGE 0: Data preparation")
    print("="*60)
    
    adata = generate_toy_data(
        n_drugs=config['data']['n_drugs'],
        n_cell_lines=config['data']['n_cell_lines'],
        n_timepoints=config['data']['n_timepoints'],
        n_cells_per_condition=config['data']['n_cells_per_condition'],
        n_genes=config['data']['n_genes'],
        n_variable_genes=config['data']['n_variable_genes'],
        seed=config['data']['seed'],
    )
    
    print(f"✓ Generated data: {adata.shape}")
    return adata


def stage1_manifold_extraction(adata, config: dict):
    """
    Stage 1: Extract manifold structure.
    
    Returns:
        state_labels, transition_scores, X_pca
    """
    print("\n" + "="*60)
    print("STAGE 1: Manifold extraction")
    print("="*60)
    
    pipeline = Stage1Pipeline(
        n_components=config['stage1']['n_components'],
        n_states=config['stage1']['n_states'],
    )
    
    X = adata.X
    result = pipeline.fit_predict(X)
    
    state_labels = result['state_labels']
    transition_scores = result['transition_scores']
    X_pca = result['X_pca']
    
    # Sort cluster labels by pseudotime (timepoint_id) to ensure temporal ordering
    # KMeans returns arbitrary cluster indices (0, 1, 2), but they may not match
    # the biological time order (P0→P_T→P1). Reorder based on mean timepoint_id.
    cluster_to_pseudotime = {}
    for cluster_id in np.unique(state_labels):
        mask = state_labels == cluster_id
        mean_pseudotime = adata.obs['timepoint_id'].values[mask].mean()
        cluster_to_pseudotime[cluster_id] = mean_pseudotime
    
    # Create mapping from old cluster indices to new temporal order
    sorted_clusters = sorted(cluster_to_pseudotime.keys(), 
                             key=lambda x: cluster_to_pseudotime[x])
    cluster_remap = {old_id: new_id for new_id, old_id in enumerate(sorted_clusters)}
    
    # Apply remapping
    state_labels_remapped = np.array([cluster_remap[label] for label in state_labels])
    
    print(f"✓ State labels (temporal order): {sorted_clusters} → {list(range(len(sorted_clusters)))}")
    print(f"✓ Cluster→Pseudotime mapping: {cluster_to_pseudotime}")
    print(f"✓ State distribution: {np.bincount(state_labels_remapped)}")
    print(f"✓ Transition scores: min={transition_scores.min():.3f}, max={transition_scores.max():.3f}")
    print(f"✓ PCA space: {X_pca.shape}")
    
    return state_labels_remapped, transition_scores, X_pca


def stage2_cfm_training(adata, state_labels, transition_scores, config: dict):
    """
    Stage 2: Train CFM velocity field.
    
    Returns:
        trained_model, velocities
    """
    print("\n" + "="*60)
    print("STAGE 2: CFM training")
    print("="*60)
    
    X = adata.X
    n_cells = X.shape[0]
    
    # Use ContextEncoder for proper condition encoding
    condition_vocab = ConditionVocabulary()
    
    # Build vocabulary from data
    for drug in adata.obs['drug'].unique():
        condition_vocab.add_drug(drug)
    for cell_line in adata.obs['cell_line'].unique():
        condition_vocab.add_cell_line(cell_line)
    
    # Encode conditions
    context_encoder = ContextEncoder(
        n_drugs=condition_vocab.get_n_drugs(),
        n_cell_lines=condition_vocab.get_n_cell_lines(),
        embedding_dim=8,
        dose_embedding_dim=4,
    )
    
    drug_ids = torch.tensor([condition_vocab.get_drug_id(d) for d in adata.obs['drug']], dtype=torch.long)
    cell_line_ids = torch.tensor([condition_vocab.get_cell_line_id(c) for c in adata.obs['cell_line']], dtype=torch.long)
    doses = torch.tensor(adata.obs['timepoint_id'].values, dtype=torch.float32).unsqueeze(-1)
    
    c_tensor = context_encoder(drug_ids, cell_line_ids, doses).detach()
    
    print(f"✓ Context encoding: shape {c_tensor.shape}")
    
    # Create model with factorized velocity field
    n_drugs = condition_vocab.get_n_drugs()
    model = CFMModel(
        input_dim=X.shape[1],
        context_dim=c_tensor.shape[1],
        hidden_dim=config['stage2']['hidden_dim'],
        n_layers=config['stage2']['n_layers'],
        n_contexts=max(1, n_drugs),  # One context head per drug type
    )
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X).float()
    s_tensor = torch.from_numpy(transition_scores).float()
    x0_list = []
    x_T_list = []
    x1_list = []
    t_list = []

    batch_size = config['stage2']['batch_size']
    n_states = config['stage1']['n_states']
    if n_states < 3:
        raise ValueError(f"3-point Bezier requires n_states >= 3, got {n_states}")

    # Use the temporally first, middle, and last clusters as P0 / P_T / P1.
    p0_idx, pT_idx, p1_idx = 0, n_states // 2, n_states - 1
    idx_state = {
        'p0': np.where(state_labels == p0_idx)[0],
        'pT': np.where(state_labels == pT_idx)[0],
        'p1': np.where(state_labels == p1_idx)[0],
    }

    smallest_state = min(len(v) for v in idx_state.values())
    if smallest_state < batch_size:
        raise ValueError(
            f"Smallest state has {smallest_state} cells, batch_size={batch_size}. "
            "Reduce batch_size or increase data."
        )

    # Build batch list: sample independent cells from each state as Bézier anchors.
    # CFM requires distributional sampling (not means) so x0, x_T, x1 come from
    # their respective state populations independently.
    rng = np.random.default_rng(config['data']['seed'])
    n_batches = smallest_state // batch_size
    for b in range(n_batches):
        idx0 = rng.choice(idx_state['p0'], batch_size, replace=False)
        idx_T = rng.choice(idx_state['pT'], batch_size, replace=False)
        idx1 = rng.choice(idx_state['p1'], batch_size, replace=False)
        x0_list.append(X_tensor[idx0])
        x_T_list.append(X_tensor[idx_T])
        x1_list.append(X_tensor[idx1])
        t_list.append(torch.rand(batch_size, 1))

    # Train
    trainer = CFMTrainer(model, learning_rate=config['stage2']['learning_rate'])
    n_cells = X_tensor.shape[0]

    n_epochs = config['stage2']['n_epochs']
    for epoch in range(n_epochs):
        total_loss = 0.0

        for i in range(len(x0_list)):
            x0 = x0_list[i]
            x_T = x_T_list[i]
            x1 = x1_list[i]
            t = t_list[i]

            # Sample context and transition scores to match batch size
            ctx_idx = rng.choice(n_cells, batch_size)
            c_batch = c_tensor[ctx_idx]
            s_batch = s_tensor[ctx_idx]

            loss_dict = model.training_step(x0, x_T, x1, t, c_batch, s_t=s_batch)
            loss = loss_dict['total_loss']

            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            trainer.optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(x0_list)
        
        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")
    
    print(f"✓ Training complete")
    
    return model, context_encoder, condition_vocab


def stage3_attribution(model, adata, state_labels, context_encoder, condition_vocab, config: dict):
    """
    Stage 3: Gene attribution.

    Uses the same ContextEncoder from Stage 2 so attribution is computed
    under the same conditioning the model was trained with.

    Returns:
        attribution_results
    """
    print("\n" + "="*60)
    print("STAGE 3: Gene attribution")
    print("="*60)

    X = adata.X
    n_cells = X.shape[0]

    # Re-encode conditions with the training-time encoder (same as Stage 2)
    drug_ids = torch.tensor(
        [condition_vocab.get_drug_id(d) for d in adata.obs['drug']], dtype=torch.long
    )
    cell_line_ids = torch.tensor(
        [condition_vocab.get_cell_line_id(cl) for cl in adata.obs['cell_line']], dtype=torch.long
    )
    doses = torch.tensor(adata.obs['timepoint_id'].values, dtype=torch.float32).unsqueeze(-1)
    c_tensor = context_encoder(drug_ids, cell_line_ids, doses).detach().numpy()

    # Use per-cell pseudotime: P0→0, P_T→0.5, P1→1. Random t washes out the
    # signal in ∂v/∂x because the Jacobian depends on where on the path we are.
    n_timepoints = config['data']['n_timepoints']
    t = (adata.obs['timepoint_id'].values / max(1, n_timepoints - 1)).astype(np.float32)

    # Build P0 cells + target-condition context for trajectory integration
    n_timepoints = config['data']['n_timepoints']
    p0_mask = state_labels == 0
    X_p0 = X[p0_mask]

    drug_ids_p0 = torch.tensor(
        [condition_vocab.get_drug_id(d) for d in adata.obs.loc[p0_mask, 'drug']], dtype=torch.long
    )
    cell_line_ids_p0 = torch.tensor(
        [condition_vocab.get_cell_line_id(cl) for cl in adata.obs.loc[p0_mask, 'cell_line']], dtype=torch.long
    )
    target_dose_p0 = torch.full((p0_mask.sum(), 1), fill_value=float(n_timepoints - 1), dtype=torch.float32)
    c_p0 = context_encoder(drug_ids_p0, cell_line_ids_p0, target_dose_p0).detach().numpy()

    analyzer = AttributionAnalyzer()
    result = analyzer.analyze(
        v_net=model,
        X=X,
        t=t,
        c=c_tensor,
        state_labels=state_labels,
        device=config['stage2']['device'],
        X_p0=X_p0,
        c_p0=c_p0,
    )
    
    print(f"✓ Gene decomposition:")
    print(f"  - Core drivers: {len(result['decomposition']['core_drivers'])}")
    print(f"  - Kinetic drivers: {len(result['decomposition']['kinetic_drivers'])}")
    print(f"  - Static markers: {len(result['decomposition']['static_markers'])}")
    
    return result


def validation(model, adata, state_labels, context_encoder, condition_vocab, config: dict):
    """
    Stage 4: Validation with ODE-based trajectory reconstruction.
    """
    print("\n" + "="*60)
    print("STAGE 4: Validation")
    print("="*60)
    
    X = adata.X
    
    # Reconstruct trajectories from P0 to P1 using ODE integration
    print("  Reconstructing trajectories via ODE integration...")
    
    reconstructor = TrajectoryReconstructor(
        model,
        device=config['stage2']['device'],
        solver='euler',  # or 'rk4'
    )
    
    # Get P0 / P1 cluster indices from temporally-sorted state labels
    n_states = config['stage1']['n_states']
    p0_state, p1_state = 0, n_states - 1

    # Get P0 cells
    p0_mask = state_labels == p0_state
    X_p0 = X[p0_mask][:100]  # Take first 100 P0 cells
    
    # Get corresponding context (reuse encoder from training)
    drug_ids = torch.tensor(
        [condition_vocab.get_drug_id(adata.obs.loc[p0_mask, 'drug'].iloc[i]) for i in range(len(X_p0))],
        dtype=torch.long
    )
    cell_line_ids = torch.tensor(
        [condition_vocab.get_cell_line_id(adata.obs.loc[p0_mask, 'cell_line'].iloc[i]) for i in range(len(X_p0))],
        dtype=torch.long
    )
    # Use target timepoint_id=1 (P1) instead of source timepoint_id=0 (P0).
    # During ODE integration we are asking "push these cells toward P1", so the
    # context should represent the *target* condition, not the starting condition.
    # Using P0's timepoint_id=0 here caused the velocity field to predict near-zero
    # drift because the model learned low velocity for cells already at t=0.
    n_timepoints = config['data']['n_timepoints']
    target_dose = torch.full((len(X_p0), 1), fill_value=float(n_timepoints - 1), dtype=torch.float32)
    c = context_encoder(drug_ids, cell_line_ids, target_dose).detach().numpy()
    
    # Reconstruct
    result = reconstructor.reconstruct_from_P0(X_p0, c, num_steps=50)
    trajectory = result['trajectory']
    
    print(f"  ✓ Trajectory shape: {trajectory.shape}")
    
    # Validate: compare reconstructed P1 *distribution* with real P1 distribution.
    # Direct element-wise MSE is meaningless because the two sets are not paired.
    p1_mask = state_labels == p1_state
    X_p1 = X[p1_mask]
    X_recon_p1 = trajectory[-1]  # (n_recon, D), final time point

    val_metrics = ValidationMetrics()
    mmd = val_metrics.mmd(X_recon_p1, X_p1)
    w2 = val_metrics.wasserstein_2_approx(X_recon_p1, X_p1)

    # Sanity baseline: MMD between P0 (start) and P1 (target) — should be larger
    # than MMD(reconstruction, P1) if the model actually moved the cells.
    mmd_p0_vs_p1 = val_metrics.mmd(X[p0_mask], X_p1)

    metrics = {
        "mmd_recon_vs_p1": mmd,
        "wasserstein_recon_vs_p1": w2,
        "mmd_p0_vs_p1_baseline": mmd_p0_vs_p1,
        "trajectory_steps": len(trajectory),
    }

    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, (int, float)) else f"  {key}: {val}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments/config.yaml')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("\n" + "="*60)
    print("CFM PIPELINE: Full end-to-end run")
    print("="*60)
    
    # Run pipeline
    adata = prepare_data(config)
    state_labels, transition_scores, X_pca = stage1_manifold_extraction(adata, config)
    model, context_encoder, condition_vocab = stage2_cfm_training(adata, state_labels, transition_scores, config)
    attribution = stage3_attribution(model, adata, state_labels, context_encoder, condition_vocab, config)
    metrics = validation(model, adata, state_labels, context_encoder, condition_vocab, config)
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
