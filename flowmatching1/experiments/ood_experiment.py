"""OOD experiment: leave-one-out compositional generalization test.

Tests whether the CFM virtual-cell simulator can predict the full P1
distribution for unseen drug×cell_line combinations by composing
individually-seen drug and cell-line embeddings.

Design
------
  4 conditions = {drug_0, drug_1} × {cell_line_0, cell_line_1}

  For each held-out condition (leave-one-out):
    • Train CFM on the remaining 3 conditions only
    • At test time: start from held-out P0 cells + held-out context
    • Integrate ODE [0→1] to predict P1 distribution
    • Compare MMD(predicted, true P1) against three baselines:
        B1  no-transport  — MMD(P0_held_out, P1_held_out)
                            the trivial "do nothing" baseline
        B2  best-seen     — min_k MMD(P1_seen_k, P1_held_out)
                            nearest seen-condition P1 (static transfer)
        B3  mean-seen     — MMD(mean(P1_seen_k), P1_held_out)
                            average of all seen P1 distributions

Why this proves the virtual-cell claim
---------------------------------------
  The held-out drug and cell_line are each seen individually in training
  (drug_1 appears in drug_1×cell_line_0; cell_line_1 in drug_0×cell_line_1).
  The model must COMPOSE learned embeddings to handle the unseen combination.
  If CFM beats both baselines it is genuinely simulating new conditions rather
  than memorising or interpolating seen distributions.

Usage
-----
  cd /path/to/flowmatching
  python experiments/ood_experiment.py [--config experiments/config.yaml]
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# Allow running from project root or experiments/ directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.synthetic_data_generator import generate_toy_data
from models.stage1_featuremap import Stage1Pipeline
from models.stage2_cfm import CFMModel, CFMTrainer
from models.context_encoder import ContextEncoder, ConditionVocabulary
from models.ode_solver import TrajectoryReconstructor
from validation.metrics import ValidationMetrics


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def condition_mask(adata, drug: str, cell_line: str) -> np.ndarray:
    """Boolean mask for cells belonging to one drug×cell_line condition."""
    return (adata.obs["drug"] == drug).values & (adata.obs["cell_line"] == cell_line).values


def sort_state_labels_by_pseudotime(state_labels: np.ndarray, adata) -> np.ndarray:
    """Reindex KMeans cluster IDs so that cluster 0 = earliest timepoint."""
    cluster_to_pseudo = {}
    for cid in np.unique(state_labels):
        mask = state_labels == cid
        cluster_to_pseudo[cid] = adata.obs["timepoint_id"].values[mask].mean()
    sorted_clusters = sorted(cluster_to_pseudo, key=lambda c: cluster_to_pseudo[c])
    remap = {old: new for new, old in enumerate(sorted_clusters)}
    return np.array([remap[s] for s in state_labels])


# ──────────────────────────────────────────────────────────────────────────────
# Single OOD fold
# ──────────────────────────────────────────────────────────────────────────────

def run_ood_fold(
    adata_full,
    held_out_drug: str,
    held_out_cell_line: str,
    config: dict,
    verbose: bool = True,
) -> dict:
    """
    Run one leave-one-out OOD fold.

    Parameters
    ----------
    adata_full        : AnnData with all 4 conditions
    held_out_drug     : drug name to withhold
    held_out_cell_line: cell line name to withhold
    config            : pipeline config dict
    verbose           : whether to print progress

    Returns
    -------
    dict with keys: mmd_model, mmd_b1_no_transport, mmd_b2_best_seen,
                    mmd_b3_mean_seen, improvement_vs_b1, improvement_vs_b2
    """
    tag = f"[{held_out_drug} × {held_out_cell_line}]"
    if verbose:
        print(f"\n{'='*60}")
        print(f"  OOD fold: held-out = {tag}")
        print(f"{'='*60}")

    # ── Split data ────────────────────────────────────────────────────────────
    ho_mask = condition_mask(adata_full, held_out_drug, held_out_cell_line)
    tr_mask = ~ho_mask

    # AnnData subsets (use .copy() to avoid SettingWithCopyWarning later)
    import anndata as ad
    adata_train = adata_full[tr_mask].copy()
    adata_test  = adata_full[ho_mask].copy()

    if verbose:
        print(f"  Train: {adata_train.n_obs} cells | Test: {adata_test.n_obs} cells")

    X_train = adata_train.X
    X_test  = adata_test.X
    n_timepoints = config["data"]["n_timepoints"]

    # ── Stage 1: manifold extraction on training data ─────────────────────────
    s1 = Stage1Pipeline(
        n_components=config["stage1"]["n_components"],
        n_states=config["stage1"]["n_states"],
    )
    result1 = s1.fit_predict(X_train)
    state_labels_train = sort_state_labels_by_pseudotime(
        result1["state_labels"], adata_train
    )
    transition_scores_train = result1["transition_scores"]

    if verbose:
        print(f"  Stage 1 done | state dist: {np.bincount(state_labels_train)}")

    # ── Stage 2: build context encoder and train CFM on training data ─────────
    condition_vocab = ConditionVocabulary()
    for drug in adata_train.obs["drug"].unique():
        condition_vocab.add_drug(drug)
    for cl in adata_train.obs["cell_line"].unique():
        condition_vocab.add_cell_line(cl)

    # NOTE: held-out drug and cell_line each appear individually in training,
    # so the vocab has entries for them — we can safely call get_drug_id /
    # get_cell_line_id for the held-out condition at inference time.

    context_encoder = ContextEncoder(
        n_drugs=condition_vocab.get_n_drugs(),
        n_cell_lines=condition_vocab.get_n_cell_lines(),
        embedding_dim=8,
        dose_embedding_dim=4,
    )

    drug_ids_tr = torch.tensor(
        [condition_vocab.get_drug_id(d) for d in adata_train.obs["drug"]],
        dtype=torch.long,
    )
    cl_ids_tr = torch.tensor(
        [condition_vocab.get_cell_line_id(c) for c in adata_train.obs["cell_line"]],
        dtype=torch.long,
    )
    doses_tr = torch.tensor(
        adata_train.obs["timepoint_id"].values, dtype=torch.float32
    ).unsqueeze(-1)

    c_train = context_encoder(drug_ids_tr, cl_ids_tr, doses_tr).detach()

    n_states   = config["stage1"]["n_states"]
    batch_size = config["stage2"]["batch_size"]
    n_drugs_tr = condition_vocab.get_n_drugs()

    model = CFMModel(
        input_dim=X_train.shape[1],
        context_dim=c_train.shape[1],
        drug_context_dim=context_encoder.embedding_dim,
        cell_context_dim=context_encoder.embedding_dim,
        hidden_dim=config["stage2"]["hidden_dim"],
        n_layers=config["stage2"]["n_layers"],
    )

    # Build Bézier batches (same logic as run_pipeline.py)
    X_tensor = torch.from_numpy(X_train).float()
    s_tensor = torch.from_numpy(transition_scores_train).float()

    p0_idx, pT_idx, p1_idx = 0, n_states // 2, n_states - 1
    idx_state = {
        "p0": np.where(state_labels_train == p0_idx)[0],
        "pT": np.where(state_labels_train == pT_idx)[0],
        "p1": np.where(state_labels_train == p1_idx)[0],
    }
    smallest_state = min(len(v) for v in idx_state.values())
    if smallest_state < batch_size:
        batch_size = smallest_state // 2
        if verbose:
            print(f"  Warning: reduced batch_size to {batch_size} (smallest state = {smallest_state})")

    rng = np.random.default_rng(config["data"]["seed"])

    # ── Condition-stratified batch construction ────────────────────────────────
    # KEY FIX: bind (x0, xT, x1) and context c to the SAME drug×cell_line condition.
    # Previously context was sampled randomly from all cells → x and c were
    # decorrelated → v_drug/v_cell received only noise as training signal.
    # With stratified sampling, the model sees proper (state, condition) pairs.
    drug_arr = adata_train.obs["drug"].values
    cl_arr   = adata_train.obs["cell_line"].values
    seen_cond_list = [
        (d, cl)
        for d  in np.unique(drug_arr)
        for cl in np.unique(cl_arr)
    ]

    timepoint_arr = adata_train.obs["timepoint_id"].values  # 0=P0, 1=P_T, 2=P1

    cond_batches = []   # list of (x0, xT, x1, t, c, s)
    for drug, cell_line in seen_cond_list:
        cond_mask = (drug_arr == drug) & (cl_arr == cell_line)
        g_idx     = np.where(cond_mask)[0]          # indices into X_train

        # Use actual timepoint_id labels — more reliable than KMeans state labels
        # which are global clusters and may not align with per-condition timepoints
        tp_cond   = timepoint_arr[cond_mask]
        idx_p0_c  = g_idx[tp_cond == 0]
        idx_pT_c  = g_idx[tp_cond == 1]
        idx_p1_c  = g_idx[tp_cond == 2]

        min_cells = min(len(idx_p0_c), len(idx_pT_c), len(idx_p1_c))
        if min_cells < batch_size:
            if verbose:
                print(f"  Skip {drug}×{cell_line}: {min_cells} cells/state < batch_size")
            continue

        # Context: target-dose encoding for this condition (matches inference time)
        drug_id  = condition_vocab.get_drug_id(drug)
        cl_id    = condition_vocab.get_cell_line_id(cell_line)
        d_ids_c  = torch.tensor([drug_id]  * batch_size, dtype=torch.long)
        cl_ids_c = torch.tensor([cl_id]    * batch_size, dtype=torch.long)
        tdose    = torch.full((batch_size, 1), float(n_timepoints - 1), dtype=torch.float32)
        c_cond   = context_encoder(d_ids_c, cl_ids_c, tdose).detach()

        n_cond_batches = min_cells // batch_size
        for _ in range(n_cond_batches):
            i0 = rng.choice(idx_p0_c, batch_size, replace=False)
            iT = rng.choice(idx_pT_c, batch_size, replace=False)
            i1 = rng.choice(idx_p1_c, batch_size, replace=False)
            cond_batches.append((
                X_tensor[i0], X_tensor[iT], X_tensor[i1],
                torch.rand(batch_size, 1),
                c_cond,
                s_tensor[i0],
            ))

    if not cond_batches:
        raise RuntimeError(
            "No valid condition batches — increase n_cells_per_condition or reduce batch_size."
        )
    if verbose:
        print(f"  Stratified batches: {len(cond_batches)} total "
              f"from {len(seen_cond_list)} conditions")

    trainer = CFMTrainer(model, learning_rate=config["stage2"]["learning_rate"])
    n_epochs = config["stage2"]["n_epochs"]

    for epoch in range(n_epochs):
        total_loss  = 0.0
        total_orth  = 0.0
        total_flow  = 0.0
        perm = rng.permutation(len(cond_batches))   # shuffle each epoch

        for idx in perm:
            x0, xT, x1, t, c_batch, s_batch = cond_batches[idx]

            # CFG: randomly null out context (15% drop)
            c_in = torch.zeros_like(c_batch) if rng.random() < 0.15 else c_batch

            loss_dict = model.training_step(x0, xT, x1, t, c_in, s_t=s_batch)
            trainer.optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            trainer.optimizer.step()
            total_loss  += loss_dict["total_loss"].item()
            total_orth  += loss_dict.get("l_orth",  0.0)
            total_flow  += loss_dict.get("l_flow",  0.0)

        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            nb = len(cond_batches)
            print(f"  Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {total_loss/nb:.4f} | "
                  f"l_flow: {total_flow/nb:.4f} | "
                  f"l_orth: {total_orth/nb:.4f}")

    if verbose:
        print("  Training done.")

    # ── ODE inference on held-out condition ───────────────────────────────────
    # Take P0 cells from held-out condition (timepoint_id == 0)
    p0_test_mask = adata_test.obs["timepoint_id"].values == 0
    X_p0_test = X_test[p0_test_mask]

    # Build context for held-out condition targeting P1 timepoint
    # drug and cell_line embeddings were trained on other conditions, but
    # both IDs exist in the vocabulary → compositional generalization.
    n_p0_test = X_p0_test.shape[0]
    drug_ids_test = torch.tensor(
        [condition_vocab.get_drug_id(held_out_drug)] * n_p0_test, dtype=torch.long
    )
    cl_ids_test = torch.tensor(
        [condition_vocab.get_cell_line_id(held_out_cell_line)] * n_p0_test,
        dtype=torch.long,
    )
    # Target dose = final timepoint (n_timepoints - 1) to ask "push to P1"
    target_dose = torch.full((n_p0_test, 1), float(n_timepoints - 1), dtype=torch.float32)
    c_test = context_encoder(drug_ids_test, cl_ids_test, target_dose).detach().numpy()

    reconstructor = TrajectoryReconstructor(
        model, device=config["stage2"]["device"], solver="euler"
    )
    traj_result = reconstructor.reconstruct_from_P0(X_p0_test, c_test, num_steps=50)
    X_pred_p1 = traj_result["trajectory"][-1]  # (n_p0_test, n_genes)

    # True P1 cells from held-out condition
    p1_test_mask = adata_test.obs["timepoint_id"].values == (n_timepoints - 1)
    X_true_p1 = X_test[p1_test_mask]

    # ── Baselines ─────────────────────────────────────────────────────────────
    # B1: no-transport — compare starting P0 to true P1 (naive baseline)
    val = ValidationMetrics()
    mmd_b1 = val.mmd(X_p0_test, X_true_p1)

    # B2: best-seen — for each seen condition's P1, compute MMD to held-out P1
    seen_conditions = [
        (d, c)
        for d in adata_full.obs["drug"].unique()
        for c in adata_full.obs["cell_line"].unique()
        if not (d == held_out_drug and c == held_out_cell_line)
    ]
    mmd_seen = []
    for seen_drug, seen_cl in seen_conditions:
        seen_mask_p1 = (
            condition_mask(adata_full, seen_drug, seen_cl)
            & (adata_full.obs["timepoint_id"].values == (n_timepoints - 1))
        )
        X_seen_p1 = adata_full.X[seen_mask_p1]
        mmd_seen.append(val.mmd(X_seen_p1, X_true_p1))
    mmd_b2 = float(np.min(mmd_seen))

    # B3: mean-seen — average the seen P1 expressions, compare to held-out P1
    all_seen_p1_cells = []
    for seen_drug, seen_cl in seen_conditions:
        seen_mask_p1 = (
            condition_mask(adata_full, seen_drug, seen_cl)
            & (adata_full.obs["timepoint_id"].values == (n_timepoints - 1))
        )
        all_seen_p1_cells.append(adata_full.X[seen_mask_p1])
    X_mean_seen_p1 = np.vstack(all_seen_p1_cells)  # concatenated, not mean of means
    mmd_b3 = val.mmd(X_mean_seen_p1, X_true_p1)

    # ── Model metric ──────────────────────────────────────────────────────────
    mmd_model = val.mmd(X_pred_p1, X_true_p1)

    improvement_vs_b1 = (mmd_b1 - mmd_model) / mmd_b1 * 100  # % reduction
    improvement_vs_b2 = (mmd_b2 - mmd_model) / mmd_b2 * 100

    if verbose:
        print(f"\n  Results for held-out {tag}:")
        print(f"    MMD(model → P1):         {mmd_model:.4f}")
        print(f"    B1 no-transport (P0→P1): {mmd_b1:.4f}  (↓{improvement_vs_b1:.1f}%)")
        print(f"    B2 best-seen P1:         {mmd_b2:.4f}  (↓{improvement_vs_b2:.1f}%)")
        print(f"    B3 mean-seen P1 pool:    {mmd_b3:.4f}")

    return {
        "held_out": f"{held_out_drug}×{held_out_cell_line}",
        "mmd_model": mmd_model,
        "mmd_b1_no_transport": mmd_b1,
        "mmd_b2_best_seen": mmd_b2,
        "mmd_b3_mean_seen": mmd_b3,
        "improvement_vs_b1_pct": improvement_vs_b1,
        "improvement_vs_b2_pct": improvement_vs_b2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OOD leave-one-out experiment")
    parser.add_argument("--config", default="experiments/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n" + "="*60)
    print("OOD EXPERIMENT: leave-one-out compositional generalization")
    print("="*60)
    print(
        "Claim: CFM can predict P1 distribution under UNSEEN drug×cell_line\n"
        "combinations by composing individually-seen condition embeddings.\n"
    )

    # Generate synthetic data once for all folds (same seed → reproducible)
    print("Generating synthetic data (all 4 conditions)...")
    adata_full = generate_toy_data(
        n_drugs=config["data"]["n_drugs"],
        n_cell_lines=config["data"]["n_cell_lines"],
        n_timepoints=config["data"]["n_timepoints"],
        n_cells_per_condition=config["data"]["n_cells_per_condition"],
        n_genes=config["data"]["n_genes"],
        n_variable_genes=config["data"]["n_variable_genes"],
        seed=config["data"]["seed"],
    )

    drugs      = sorted(adata_full.obs["drug"].unique())
    cell_lines = sorted(adata_full.obs["cell_line"].unique())

    # Run all 4 leave-one-out folds
    results = []
    for drug in drugs:
        for cell_line in cell_lines:
            fold_result = run_ood_fold(
                adata_full, drug, cell_line, config, verbose=True
            )
            results.append(fold_result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY: OOD generalization across all 4 held-out conditions")
    print("="*60)

    df = pd.DataFrame(results)
    df = df.rename(columns={
        "held_out":            "Held-out condition",
        "mmd_model":           "Model MMD↓",
        "mmd_b1_no_transport": "B1 no-transport",
        "mmd_b2_best_seen":    "B2 best-seen",
        "mmd_b3_mean_seen":    "B3 mean-seen pool",
        "improvement_vs_b1_pct": "Δ vs B1 (%)",
        "improvement_vs_b2_pct": "Δ vs B2 (%)",
    })
    df_str = df.to_string(index=False, float_format=lambda x: f"{x:.4f}")
    print(df_str)

    # ── Aggregate verdict ─────────────────────────────────────────────────────
    beats_b1 = (df["Δ vs B1 (%)"] > 0).sum()
    beats_b2 = (df["Δ vs B2 (%)"] > 0).sum()
    mean_imp_b1 = df["Δ vs B1 (%)"].mean()
    mean_imp_b2 = df["Δ vs B2 (%)"].mean()

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    print(f"  Model beats B1 (no-transport):  {beats_b1}/{len(results)} folds  "
          f"(avg improvement {mean_imp_b1:+.1f}%)")
    print(f"  Model beats B2 (best-seen P1):  {beats_b2}/{len(results)} folds  "
          f"(avg improvement {mean_imp_b2:+.1f}%)")

    if beats_b1 >= 3 and mean_imp_b1 > 0:
        print("\n  ✅ CFM reliably transports P0 toward the correct P1 distribution")
        print("     even under held-out drug×cell_line combinations.")
    else:
        print("\n  ⚠️  Model does not consistently beat the no-transport baseline.")
        print("     Consider increasing n_epochs, hidden_dim, or n_cells_per_condition.")

    if beats_b2 >= 3 and mean_imp_b2 > 0:
        print("  ✅ CFM outperforms naive condition transfer — genuine compositional")
        print("     generalization beyond copying the most-similar seen condition.")
    else:
        print("  ℹ️  Model does not consistently beat nearest-seen-condition transfer.")
        print("     The virtual-cell claim requires improvement over this baseline.")

    print()
    return df


if __name__ == "__main__":
    main()
