"""Generate toy perturbation scRNA-seq data for pipeline testing."""

import numpy as np
from scipy.stats import multivariate_normal
import anndata as ad


def generate_toy_data(
    n_drugs: int = 2,
    n_cell_lines: int = 2,
    n_timepoints: int = 3,  # P0 (control), P_T (intermediate), P1 (final)
    n_cells_per_condition: int = 500,
    n_genes: int = 2000,
    n_variable_genes: int = 50,
    seed: int = 42,
):
    """
    Generate synthetic perturbation response data.
    
    Structure:
    - P0 (control): baseline state
    - P_T (intermediate): partially perturbed
    - P1 (final): fully perturbed
    
    True signal: ~10 genes up, ~10 genes down along trajectory
    """
    np.random.seed(seed)
    
    drugs = [f"drug_{i}" for i in range(n_drugs)]
    cell_lines = [f"cell_line_{i}" for i in range(n_cell_lines)]
    timepoints = ["P0", "P_T", "P1"]
    
    # Collect all cells and metadata
    X_list = []
    obs_list = []
    
    for drug_idx, drug in enumerate(drugs):
        for cell_line_idx, cell_line in enumerate(cell_lines):
            for time_idx, timepoint in enumerate(timepoints):
                # Mean expression profile: varies by drug + cell-line + time
                base_mean = np.random.randn(n_genes) * 0.5
                
                # Gene module 1: perturbation-induced (up-regulated)
                up_genes = np.arange(10)
                base_mean[up_genes] += 1.0 + drug_idx * 0.2 + time_idx * 0.5
                
                # Gene module 2: perturbation-induced (down-regulated)
                down_genes = np.arange(10, 20)
                base_mean[down_genes] -= 1.0 + drug_idx * 0.2 + time_idx * 0.3
                
                # Cell-line specific signature
                base_mean[20:30] += cell_line_idx * 0.3
                
                # Sample cells from multivariate normal
                cov = np.eye(n_genes) * 0.2
                X_cells = multivariate_normal.rvs(
                    mean=base_mean, cov=cov, size=n_cells_per_condition
                )
                X_cells = np.clip(X_cells, 0, None)  # log counts should be non-negative
                
                X_list.append(X_cells)
                obs_list.extend(
                    [
                        {
                            "drug": drug,
                            "cell_line": cell_line,
                            "timepoint": timepoint,
                            "timepoint_id": time_idx,
                        }
                    ]
                    * n_cells_per_condition
                )
    
    # Combine into AnnData
    X = np.vstack(X_list)
    
    import pandas as pd
    obs_df = pd.DataFrame(obs_list)
    
    adata = ad.AnnData(X=X, obs=obs_df)
    
    # Gene names
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Mark variable genes (true signal + top HVG)
    adata.var["is_variable"] = False
    var_idx = adata.var_names[:n_variable_genes]
    adata.var.loc[var_idx, "is_variable"] = True
    
    print(f"Generated toy data: {adata.shape}")
    print(f"Conditions: {n_drugs} drugs × {n_cell_lines} cell-lines × {n_timepoints} timepoints")
    print(f"Metadata:\n{adata.obs.head()}")
    
    return adata


if __name__ == "__main__":
    adata = generate_toy_data()
    adata.write_h5ad("toy_data.h5ad")
    print("Saved to toy_data.h5ad")
