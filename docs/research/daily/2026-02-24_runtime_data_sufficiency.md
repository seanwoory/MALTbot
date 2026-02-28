# 2026-02-24 — Analysis: Data Sufficiency & Runtime Estimation for Matbench CHGNet

## 1. Runtime Estimation (NVIDIA L4 GPU)
**Context:** Matbench `mp_e_form` has ~132,752 structures. A 5-fold CV means each fold uses ~106k for training and ~26k for testing.
**The Bottleneck:** `CrystalGraphConverter` runs on CPU. Converting 130k structures takes approximately **1.5 to 2 hours**. If done *per fold* without caching, graph conversion alone consumes **~8-10 hours** for a full 5-fold CV.

### Estimated Times (Without Graph Caching):
1. **`chgnet_pretrained_infer` (0 epochs, just predict test set)**
   - **Graph Conversion:** ~30 mins per fold (only converting 26k test structures).
   - **L4 GPU Inference:** ~1-2 mins per fold.
   - **Total (5-fold):** **~2.5 hours**.
2. **`chgnet_head_finetune_freeze` (10 epochs, frozen backbone)**
   - **Graph Conversion:** ~1.5 hours per fold (106k train + 26k test).
   - **L4 GPU Training:** ~2-3 mins/epoch × 10 = ~25 mins per fold.
   - **Total (5-fold):** **~10 hours** (80% of time spent on redundant graph conversion).
3. **`chgnet_full_finetune` (20 epochs, full backprop)**
   - **Graph Conversion:** ~1.5 hours per fold.
   - **L4 GPU Training:** ~6-8 mins/epoch × 20 = ~140 mins per fold.
   - **Total (5-fold):** **~19 hours**.

**Critical Conclusion:** Without persisting graph representations to disk, fast iteration (1-2 hr Colab sessions) is **impossible**.

---

## 2. Current Data Sufficiency for "Top-1 Optimization"
**Status:** `results.json` and `RESULTS.md` only record the final scalar metrics (MAE, RMSE, max_error) at the end of the run.
**Verdict:** **Insufficient.** 
To reach Top-1 (MAE 0.0170), we must hunt for gains of 0.001 eV/atom. We cannot optimize a GNN blindly. We must know *when* it overfits, *what* it fails on, and *how* the loss landscape looks. Scalar metrics alone are a "black box."

---

## 3. Missing Data & Artifacts Needed for Deep Optimization
To achieve SOTA, we must extract the following signals from the training loop:

1. **Epoch-Level Metrics (Learning Curves):**
   - `train_loss`, `val_loss`, `val_mae` per epoch.
   - *Why:* To diagnose overfitting, verify LR scheduler behavior, and set proper Early Stopping patience.
2. **Predicted vs. Actual Vectors (`predictions.csv`):**
   - Columns: `id` (or index), `composition`, `y_true`, `y_pred`, `abs_error`.
   - *Why:* To plot error distributions. Are we failing on large unit cells? Specific elements (e.g., Lanthanides)? High-energy metastable states? We need this to design targeted losses or data augmentations.
3. **Model Checkpoints (`best_model_fold_X.pth`):**
   - Save the weights of the best validation epoch.
   - *Why:* We need these weights to perform Test-Time Augmentation (TTA) or to load them into an Ensemble later. We cannot do Track C (Hybrid) without saved weights.
4. **Pre-computed Graph Cache (`.pt` files):**
   - *Why:* To reduce the 10-hour graph conversion overhead to **seconds**.

---

## 4. Instructions for the Coder

**Coder Task: Implement the "Optimization Telemetry & Caching" Pipeline**

1. **Global Graph Caching (Highest Priority):**
   - Modify the dataset loading step. Before the fold loop, convert *all* 132k `pymatgen.Structure` objects to CHGNet graphs once.
   - Save the list of graphs to disk (e.g., `data/mp_e_form_chgnet_graphs.pkl` or `.pt`).
   - If the file exists, `load()` it instead of re-running `CrystalGraphConverter`.
   - *Note:* Handle isolated atoms by dynamically increasing `atom_graph_cutoff` during this initial caching step so it never fails again.

2. **Epoch History Logging:**
   - Inside the training loop (or using CHGNet's `Trainer` hooks), append `train_loss`, `val_loss`, and `val_mae` to a list per epoch.
   - Save this history to `results/daily/YYYY-MM-DD/EXP_NAME/history_fold_{i}.csv`.

3. **Predictions Export:**
   - After testing on a fold, save a CSV file: `results/daily/YYYY-MM-DD/EXP_NAME/preds_fold_{i}.csv`.
   - Include `y_true` and `y_pred`.

4. **Model Checkpointing:**
   - Ensure the `Trainer` saves `best_model.pth` based on validation MAE.
   - Move or rename this file to `results/daily/YYYY-MM-DD/EXP_NAME/model_fold_{i}.pth` at the end of the fold.

*(End of Instructions)*