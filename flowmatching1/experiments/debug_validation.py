"""
debug_validation.py
===================
Three focused checks to confirm the CFM pipeline is actually working:

  1. Velocity sanity check   — is the velocity field outputting non-trivial values?
  2. Gene recovery           — does Stage-3 attribution recover the known ground-truth genes?
  3. PCA trajectory plot     — do ODE-integrated cells visually move from P0 toward P1?

Run from the project root:
    python experiments/debug_validation.py --config experiments/config.yaml
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.synthetic_data_generator import generate_toy_data
from experiments.run_pipeline import (
    load_config,
    prepare_data,
    stage1_manifold_extraction,
    stage2_cfm_training,
    stage3_attribution,
)
from models.ode_solver import TrajectoryReconstructor
from models.context_encoder import ContextEncoder, ConditionVocabulary
from validation.metrics import ValidationMetrics

# ── Ground-truth gene sets (from synthetic_data_generator.py) ──────────────
# gene_0–9  : up-regulated along trajectory
# gene_10–19: down-regulated along trajectory
TRUE_DRIVER_GENES = set(range(20))


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Velocity sanity
# ═══════════════════════════════════════════════════════════════════════════════

def check_velocity(model, adata, state_labels, context_encoder, condition_vocab, config):
    """
    Directly query v_θ(x, t, c) at t=0, 0.5, 1 and report:
      - mean velocity magnitude   (should be >> 0)
      - cosine similarity between v(P0 cells) and (mean_P1 – mean_P0)
        (positive = model pushes in the right direction)
    """
    print("\n" + "═"*60)
    print("CHECK 1 — Velocity field sanity")
    print("═"*60)

    X = adata.X
    n_states = config['stage1']['n_states']
    p0_mask = state_labels == 0
    p1_mask = state_labels == (n_states - 1)

    # Ideal drift direction in gene space
    mean_p0 = X[p0_mask].mean(axis=0)
    mean_p1 = X[p1_mask].mean(axis=0)
    ideal_direction = mean_p1 - mean_p0                    # (D,)
    ideal_norm = np.linalg.norm(ideal_direction) + 1e-8

    # Build context for P0 cells using target dose (P1 timepoint)
    p0_cells = X[p0_mask][:50]
    n_timepoints = config['data']['n_timepoints']

    drug_ids = torch.tensor(
        [condition_vocab.get_drug_id(d) for d in adata.obs.loc[p0_mask, 'drug'].values[:50]],
        dtype=torch.long,
    )
    cell_line_ids = torch.tensor(
        [condition_vocab.get_cell_line_id(cl) for cl in adata.obs.loc[p0_mask, 'cell_line'].values[:50]],
        dtype=torch.long,
    )
    target_dose = torch.full((len(p0_cells), 1), fill_value=float(n_timepoints - 1), dtype=torch.float32)
    c = context_encoder(drug_ids, cell_line_ids, target_dose).detach()

    x_tensor = torch.from_numpy(p0_cells).float()
    model.eval()

    results = {}
    for t_val in [0.0, 0.5, 1.0]:
        t_tensor = torch.full((len(p0_cells), 1), fill_value=t_val, dtype=torch.float32)
        with torch.no_grad():
            v = model(x_tensor, t_tensor, c).numpy()   # (50, D)

        mag = np.linalg.norm(v, axis=1).mean()

        # Cosine similarity between each cell's velocity and ideal direction
        cos_sims = (v @ ideal_direction) / (np.linalg.norm(v, axis=1) + 1e-8) / ideal_norm
        mean_cos = cos_sims.mean()

        results[t_val] = {"magnitude": mag, "cosine_sim": mean_cos}
        print(f"  t={t_val:.1f} | mean‖v‖ = {mag:.4f} | cos_sim(v, P1-P0) = {mean_cos:.4f}")

    # Verdict
    avg_mag = np.mean([r["magnitude"] for r in results.values()])
    avg_cos = np.mean([r["cosine_sim"] for r in results.values()])

    print()
    if avg_mag < 0.01:
        print("  ⚠  FAIL — velocity magnitude near zero. Model is not predicting drift.")
        print("     Possible causes: learning rate too low, loss collapsed, context bug.")
    elif avg_cos < 0.0:
        print("  ⚠  WARN — velocity points *away* from P1. Direction may be inverted.")
    else:
        print(f"  ✓  PASS — velocity is non-trivial (mag={avg_mag:.3f}) and points toward P1 (cos={avg_cos:.3f})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Gene recovery
# ═══════════════════════════════════════════════════════════════════════════════

def check_gene_recovery(attribution_result, adata=None):
    """
    Compare Stage-3 predicted driver genes against the known ground-truth genes.
    Report precision / recall / F1.
    A random baseline at top-N from D=2000 genes gives F1 ≈ 2%–5%.
    """
    print("\n" + "═"*60)
    print("CHECK 2 — Gene recovery vs ground truth")
    print("═"*60)

    decomp = attribution_result["decomposition"]
    pred_core    = set(decomp.get("core_drivers", []))
    pred_kinetic = set(decomp.get("kinetic_drivers", []))
    pred_static  = set(decomp.get("static_markers", []))

    pred_all_dynamic = pred_core | pred_kinetic          # genes attributed to flow

    print(f"  Ground-truth driver genes : {sorted(TRUE_DRIVER_GENES)}")
    print(f"  Predicted core drivers    : {sorted(pred_core)}")
    print(f"  Predicted kinetic drivers : {sorted(pred_kinetic)}")
    print(f"  Predicted static markers  : {sorted(pred_static)}")
    print()

    for label, pred_set in [
        ("core_drivers",         pred_core),
        ("kinetic_drivers",      pred_kinetic),
        ("core + kinetic",       pred_all_dynamic),
    ]:
        metrics = ValidationMetrics.gene_recovery(
            np.array(sorted(pred_set)),
            np.array(sorted(TRUE_DRIVER_GENES)),
        )
        print(f"  [{label}]")
        print(f"    precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  IoU={metrics['iou']:.3f}")

    # Random baseline F1 (hypergeometric expectation)
    # D must reflect the actual number of genes in the data, not a hardcoded value.
    D = attribution_result["importance_flow"].shape[0]
    K = len(TRUE_DRIVER_GENES)
    N = len(pred_all_dynamic) if pred_all_dynamic else 1
    rand_prec = K / D
    rand_rec  = N / D if N < D else 1.0
    rand_f1   = 2 * rand_prec * rand_rec / (rand_prec + rand_rec + 1e-8)
    print(f"\n  Random baseline F1 ≈ {rand_f1:.4f}  (picking {N} genes uniformly from {D})")

    actual_f1 = ValidationMetrics.gene_recovery(
        np.array(sorted(pred_all_dynamic or [0])),
        np.array(sorted(TRUE_DRIVER_GENES)),
    )["f1"]

    print()
    if actual_f1 > 3 * rand_f1:
        print(f"  ✓  PASS — F1={actual_f1:.3f} is {actual_f1/rand_f1:.1f}× above random baseline")
    elif actual_f1 > rand_f1:
        print(f"  ⚠  WEAK — F1={actual_f1:.3f} is only {actual_f1/rand_f1:.1f}× above random. May need more training.")
    else:
        print(f"  ✗  FAIL — F1={actual_f1:.3f} is at or below random baseline ({rand_f1:.3f}).")

    return actual_f1


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — PCA trajectory visualization
# ═══════════════════════════════════════════════════════════════════════════════

def check_pca_trajectory(model, adata, state_labels, context_encoder, condition_vocab, config, out_path):
    """
    Project all cells + ODE trajectory into PCA space and plot.
    The trajectory should smoothly arc from the P0 cloud toward the P1 cloud.
    """
    print("\n" + "═"*60)
    print("CHECK 3 — PCA trajectory visualization")
    print("═"*60)

    X = adata.X
    n_states = config['stage1']['n_states']
    n_timepoints = config['data']['n_timepoints']

    p0_mask = state_labels == 0
    pT_mask = state_labels == (n_states // 2)
    p1_mask = state_labels == (n_states - 1)

    # Fit PCA on all cells
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X)

    X_p0_2d = pca.transform(X[p0_mask])
    X_pT_2d = pca.transform(X[pT_mask])
    X_p1_2d = pca.transform(X[p1_mask])

    # ODE integration: P0 → P1
    X_p0_cells = X[p0_mask][:100]
    drug_ids = torch.tensor(
        [condition_vocab.get_drug_id(d) for d in adata.obs.loc[p0_mask, 'drug'].values[:100]],
        dtype=torch.long,
    )
    cell_line_ids = torch.tensor(
        [condition_vocab.get_cell_line_id(cl) for cl in adata.obs.loc[p0_mask, 'cell_line'].values[:100]],
        dtype=torch.long,
    )
    target_dose = torch.full((len(X_p0_cells), 1), fill_value=float(n_timepoints - 1), dtype=torch.float32)
    c = context_encoder(drug_ids, cell_line_ids, target_dose).detach().numpy()

    reconstructor = TrajectoryReconstructor(model, device=config['stage2']['device'], solver='euler')
    result = reconstructor.reconstruct_from_P0(X_p0_cells, c, num_steps=50)
    trajectory = result["trajectory"]   # (T, n_cells, D)

    # Project trajectory into PCA
    T, B, D = trajectory.shape
    traj_flat = trajectory.reshape(T * B, D)
    traj_2d = pca.transform(traj_flat).reshape(T, B, 2)  # (T, n_cells, 2)
    traj_mean = traj_2d.mean(axis=1)                     # (T, 2) — mean cell position over time

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter of real cells
    ax = axes[0]
    ax.scatter(*X_p0_2d.T, s=5, alpha=0.3, color="#4C72B0", label="P0 (control)")
    ax.scatter(*X_pT_2d.T, s=5, alpha=0.3, color="#55A868", label="P_T (intermediate)")
    ax.scatter(*X_p1_2d.T, s=5, alpha=0.3, color="#C44E52", label="P1 (treated)")

    # Overlay mean trajectory
    ax.plot(traj_mean[:, 0], traj_mean[:, 1], color="black", linewidth=2, label="ODE trajectory (mean)")
    ax.scatter(*traj_mean[0],  s=80, color="blue",  marker="^", zorder=5, label="start")
    ax.scatter(*traj_mean[-1], s=80, color="red",   marker="*", zorder=5, label="end")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA: real cells + ODE trajectory")
    ax.legend(fontsize=7, markerscale=2)

    # Right: only trajectory coloured by time
    ax2 = axes[1]
    times = np.linspace(0, 1, T)
    for t_idx in range(0, T, max(1, T // 20)):
        pts = traj_2d[t_idx]       # (n_cells, 2)
        ax2.scatter(*pts.T, s=3, alpha=0.2, color=plt.cm.plasma(times[t_idx]))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label="integration time t")

    # P1 cloud for reference
    ax2.scatter(*X_p1_2d.T, s=5, alpha=0.2, color="grey", label="real P1")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_title("ODE trajectory coloured by time (vs real P1)")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓  Saved PCA trajectory plot → {out_path}")

    # ── MMD: recon vs P1, and P0 vs P1 baseline ──────────────────────────────
    X_recon_p1 = trajectory[-1]    # final time step
    X_p0_arr   = X[p0_mask]
    X_p1_arr   = X[p1_mask]

    vm = ValidationMetrics()
    mmd_recon   = vm.mmd(X_recon_p1, X_p1_arr)
    mmd_baseline = vm.mmd(X_p0_arr,  X_p1_arr)

    print(f"\n  MMD(recon_P1,  real_P1) = {mmd_recon:.4f}   ← model output")
    print(f"  MMD(P0_start,  real_P1) = {mmd_baseline:.4f}   ← no-transport baseline")
    improvement = (mmd_baseline - mmd_recon) / (mmd_baseline + 1e-8) * 100
    print()
    if mmd_recon < mmd_baseline:
        print(f"  ✓  PASS — reconstruction is {improvement:.1f}% closer to P1 than the baseline")
    else:
        print(f"  ✗  FAIL — reconstruction is not closer to P1 than just using P0.")
        print("     Check the PCA plot — if trajectory doesn't move, the velocity fix may need more epochs.")

    return {"mmd_recon": mmd_recon, "mmd_baseline": mmd_baseline}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/config.yaml")
    parser.add_argument("--plot-dir", default="experiments/debug_plots")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    config = load_config(args.config)

    print("\n" + "═"*60)
    print("DEBUG VALIDATION SCRIPT")
    print("Runs the full pipeline, then applies three targeted checks.")
    print("═"*60)

    # ── Re-run pipeline ───────────────────────────────────────────────────────
    adata = prepare_data(config)
    state_labels, transition_scores, X_pca = stage1_manifold_extraction(adata, config)
    model, context_encoder, condition_vocab = stage2_cfm_training(
        adata, state_labels, transition_scores, config
    )
    attribution = stage3_attribution(
        model, adata, state_labels, context_encoder, condition_vocab, config
    )

    # ── Three checks ──────────────────────────────────────────────────────────
    vel_results = check_velocity(
        model, adata, state_labels, context_encoder, condition_vocab, config
    )
    gene_f1 = check_gene_recovery(attribution, adata=adata)
    traj_metrics = check_pca_trajectory(
        model, adata, state_labels, context_encoder, condition_vocab, config,
        out_path=os.path.join(args.plot_dir, "pca_trajectory.png"),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("SUMMARY")
    print("═"*60)
    avg_mag = np.mean([r["magnitude"] for r in vel_results.values()])
    avg_cos = np.mean([r["cosine_sim"] for r in vel_results.values()])
    print(f"  Velocity magnitude (mean over t) : {avg_mag:.4f}")
    print(f"  Velocity cosine sim  (mean over t): {avg_cos:.4f}")
    print(f"  Gene recovery F1                  : {gene_f1:.4f}")
    print(f"  MMD recon vs P1                   : {traj_metrics['mmd_recon']:.4f}")
    print(f"  MMD P0 vs P1 (baseline)           : {traj_metrics['mmd_baseline']:.4f}")
    print(f"  PCA plot saved to                 : {args.plot_dir}/pca_trajectory.png")
    print("═"*60)


if __name__ == "__main__":
    main()
