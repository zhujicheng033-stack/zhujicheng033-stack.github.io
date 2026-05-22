# CFM Pipeline

A modular implementation of three-point Bézier Conditional Flow Matching (CFM) for single-cell perturbation analysis.

## Project Structure

```
flowmatching/
├── data/
│   └── synthetic_data_generator.py    # Generate toy perturbation data
├── models/
│   ├── stage1_featuremap.py           # Manifold extraction
│   ├── stage2_cfm.py                  # Core CFM velocity field
│   ├── stage3_attribution.py          # Gene attribution via Jacobian/IG
│   └── components/
│       ├── bezier.py                  # 3-point Bézier curves
│       ├── losses.py                  # Loss functions (L_flow, L_geom, etc.)
│       └── utils.py                   # Utilities (to be added)
├── validation/
│   └── metrics.py                     # MMD, Wasserstein-2, gene recovery
├── experiments/
│   ├── config.yaml                    # Hyperparameters
│   └── run_pipeline.py                # Main runner
└── tests/
    └── (test files)
```

## Quick Start

### 1. Install dependencies

```bash
pip install torch anndata numpy scipy scikit-learn pyyaml pandas
```

### 2. Run end-to-end pipeline

```bash
cd experiments
python run_pipeline.py --config config.yaml
```

This will:
- **Stage 0**: Generate synthetic perturbation data
- **Stage 1**: Extract state labels and transition scores
- **Stage 2**: Train CFM velocity field with Bézier anchors
- **Stage 3**: Compute gene attribution (core drivers, kinetic drivers, static markers)
- **Stage 4**: Validate reconstruction quality

### 3. Test individual modules

```bash
# Test synthetic data generator
python -m data.synthetic_data_generator

# Test Bézier curves
python -m models.components.bezier

# Test CFM model
python -m models.stage2_cfm

# Test Stage 1
python -m models.stage1_featuremap

# Test metrics
python -m validation.metrics
```

## Key Design Choices

### Modularity
- Each stage is **independent** and can be modified without affecting others
- Config file controls hyperparameters (no hardcoding)
- Each module has unit tests via `if __name__ == "__main__"`

### Three-Point Bézier CFM
- Quadratic Bézier curve: `x(t) = (1-t)²x₀ + 2t(1-t)x_T + t²x₁`
- Target velocity: `v(t) = 2(1-t)(x_T - x₀) + 2t(x₁ - x_T)`
- Can capture non-monotonic transitions (e.g., up then down)

### Loss Components
- **L_flow**: MSE between predicted and Bézier target velocities
- **L_geom**: Penalize motion in stable regions using transition score s(t)
- **L_context**: Encourage consistency across drugs/cell-lines
- **L_smooth**: Jacobian regularization to prevent collapse

### Three-Way Gene Decomposition
1. **Core drivers** (G_flow ∩ G_DGV): Captured by both kinetics and DE
2. **Kinetic drivers** (G_flow \ G_DGV): Novel discoveries from flow
3. **Static markers** (G_DGV \ G_flow): Known DE genes

## Configuration

Edit `experiments/config.yaml`:

```yaml
stage1:
  n_components: 20        # PCA dimensions
  n_states: 3             # Number of states (P0, P_T, P1)

stage2:
  hidden_dim: 256         # MLP hidden dimension
  n_layers: 3             # Number of hidden layers
  learning_rate: 0.001
  batch_size: 32
  n_epochs: 50
  lambda_geom: 0.1        # L_geom weight
  lambda_context: 0.01    # L_context weight
  lambda_smooth: 0.001    # L_smooth weight

data:
  n_drugs: 2              # Number of drugs
  n_cell_lines: 2         # Number of cell lines
  n_genes: 2000           # Total genes
  n_cells_per_condition: 500  # Cells per drug×cell_line×timepoint
```

## Next Steps

### Priority 1: Complete L_context
Currently uses simple variance minimization. Implement proper factorization:
```
v_θ = v_shared + Σ α_c · v_context_c
```

### Priority 2: Add real data support
- Load AnnData objects with real scRNA-seq
- Support external validation data

### Priority 3: Advanced metrics
- Integrate scVelo velocity comparison
- Add pathway enrichment for gene sets
- Implement integration with CRISPR screens / L1000

### Priority 4: Visualization
- UMAP of state transitions
- Velocity field quiver plots
- Gene set heatmaps

## References

- Chen et al. "Conditional Flow Matching" (2024)
- Song et al. "Flow Matching for Generative Modeling" (2022)
- Single-cell perturbation methods survey

---

**Status**: Prototype framework ready for experimentation.
**Next Milestone**: Tune on real data and validate against scVelo.
