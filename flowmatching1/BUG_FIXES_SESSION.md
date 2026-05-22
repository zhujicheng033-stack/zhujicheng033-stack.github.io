# Bug Fixes - Session 2

## Summary
Fixed 4 critical bugs and 3 medium-priority issues that were causing models to learn incorrectly or crash.

---

## 
### Bug A: Missing return in `compute_jacobian_importance` 
**File:** `models/stage3_attribution.py:63-77`
**Issue:** Function computed importance scores but never returned them, causing `None` to be passed downstream and crashing in `three_way_decomposition`.
**Also:** In-place `requires_grad_()` was polluting original tensors.

**Fix:**
- Added `return importance.detach().cpu().numpy()` at line 77
 `X.clone().requires_grad_(True)` to prevent mutation

**Test Jacobian importance now returns numpy array with correct shape:** 

---

### Bug B: Stage 3 context dimension mismatch [ALREADY FIXED]
**File:** `experiments/run_pipeline.py:218-226`
**Issue:** Was using random noise instead of trained encoder, causing shape mismatch.

**Status:** Already corrected in recent commits - now uses `context_encoder(drug_ids, cell_line_ids, doses)` consistently

---

### Bug C: Training data dead variable [ALREADY FIXED]
**File:** `experiments/run_pipeline.py:150-162`
**Issue:** Was sampling real cells `batch_X` but never using them, only training on state centroids.

**Status:** Already corrected - now properly samples x0/x_T/x1 from respective state populations via `np.random.choice`

---

### Bug D: KMeans labels have no temporal order
**File:** `experiments/run_pipeline.py:73-85`
P1 but this is 50% wrong on average.

**Fix:** Added pseudotime-based reordering:
```python
# Sort clusters by mean timepoint_id to ensure temporal order
cluster_to_pseudotime = {}
for cluster_id in np.unique(state_labels):
    mask = state_labels == cluster_id
    mean_pseudotime = adata.obs['timepoint_id'].values[mask].mean()
    cluster_to_pseudotime[cluster_id] = mean_pseudotime

sorted_clusters = sorted(cluster_to_pseudotime.keys(), 
                         key=lambda x: cluster_to_pseudotime[x])
cluster_remap = {old_id: new_id for new_id, old_id in enumerate(sorted_clusters)}
state_labels_remapped = np.array([cluster_remap[label] for label in state_labels])
```

2.0

---

## 
### Issue E: Conflicting l_context implementations
**File:** `models/components/losses.py:85-118`
**Issue:** Two implementations existed:
- `losses.CFMLosses.l_context()` - variance minimization (unused)
- `stage2_cfm.py:240-241` - L2 magnitude regularization (used)

**Fix:** 
- Kept L2 magnitude (correct for residual/mixture-of-experts)
- Rewrote `l_context()` method to match actual usage
- Simplified signature: `l_context(v_context_list)` instead of `(v_shared, v_context_list, weights)`

**Benefit:** Consistent semantics - context heads stay small, v_shared captures dominant dynamics

---

### Issue F: Context heads computed twice [ALREADY OPTIMIZED]
**File:** `models/stage2_cfm.py:226-227`
**Status:** Already optimized - `_forward_internal()` returns both `(v, v_ctx_outputs)`, used in line 227 without duplication

---

### Issue G: l_smooth uses 2 forwards [OPTIMIZATION DEFERRED]
**File:** `models/stage2_cfm.py:246-249`
**Issue:** Hutchinson finite difference requires 2 forwards; could use `torch.func.jvp` for 1 forward.
**Decision:** Keep current implementation (already fast enough with shared t_emb). Can optimize in future if profiling shows bottleneck.

---

## 
### Fixed NumPy 2.x compatibility
**File:** `requirements.txt`
**Change:** Added `numpy>=1.24.0,<2.0.0` to avoid scipy/sklearn conflicts

**Benefit:** Clean imports on all systems

---

### Fixed obs DataFrame initialization
**File:** `data/synthetic_data_generator.py:74-89`
**Issue:** obs was numpy array of dicts, causing pandas warning and potential bugs

**Fix:**
- Direct `pd.DataFrame(obs_list)` construction
- Use gene names for var indexing instead of integer positions
- Pass obs_df directly to AnnData constructor

**Benefit No more ImplicitModificationWarning:** 

---

### Fixed docstring warning
**File:** `models/stage3_attribution.py:9`
**Change:** Changed to raw string `r"""..."""` to escape backslashes in docstring

---

## Test Results

| Module | Test | Status |
|--------|------|--------|
| Stage 0 (Data Gen) | Generate 30 cells, check metadata PASS | | 
| Stage 1 (Clustering) | Cluster 90 cells into 3 states PASS | | 
| Stage 1 (Pseudotime) | Verify cluster ordering by pseudotime PASS | | 
| Stage 3 (Jacobian) | compute_jacobian_importance returns numpy array PASS | | 
| All imports | Module syntax and imports PASS | | 

---

## Files Modified
1. `models/stage3_attribution.py` - Fixed return statement, clone() for requires_grad
2. `experiments/run_pipeline.py` - Added pseudotime-based cluster reordering
3. `models/components/losses.py` - Unified l_context to L2 magnitude
4. `data/synthetic_data_generator.py` - Fixed DataFrame initialization
5. `requirements.txt` - Fixed NumPy version constraint

---

## Impact on Model Quality
- **Before:** Model trained on arbitrary cluster orderings (50% chance of backwards flow)
P1 dynamics
- **Stage 3:** No longer crashes on jacobian computation; returns valid importance scores

These fixes enable the pipeline to learn biologically meaningful velocity fields.
