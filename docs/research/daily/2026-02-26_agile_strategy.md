# 2026-02-26 — Critical Strategic Pivot: Agile Experimentation Protocol (Fast Iteration)

## 1. The Problem: The "5-Fold Trap"
The core reason we haven't produced a finalized optimization loop in 4 days is the insistence on running full 5-fold cross-validation on 132k structures for *every* experiment. 
A 5-hour iteration cycle destroys our ability to explore hyperparameters (LR, weight decay, loss functions, architectural tweaks). Deep learning research requires rapid failure and fast feedback loops, not exhaustive benchmarking at every step.

## 2. The Agile Blueprint: "10% Data, 1 Fold, Fast Feedback"
We must immediately pivot to a pipeline that completes an experiment in **under 20 minutes**. Full 5-fold CV is strictly reserved for the *final* model evaluation.

### Principle 1: Single Fold Prototyping (Fold 0 Only)
- **Action:** For all exploratory runs, we will *only* train and validate on **Fold 0**. 
- **Why:** Cuts runtime by 80% immediately. Fold 0 is statistically representative enough to detect if a learning rate is diverging or a model is overfitting.

### Principle 2: Fractional Datasets (The "Speed Run" Mode)
- **Action:** Introduce a `fractional_data` parameter (e.g., `0.1` or `0.2`) to the training pipeline.
- **Mechanism:** If set to `0.1`, we randomly sample 10% of the training set (~10,600 structures) and 10% of the validation set (~2,600 structures) for Fold 0.
- **Why:** Reduces a 40-minute fold to **4 minutes**. We can verify learning dynamics (loss dropping, no NaNs) almost instantly.

### Principle 3: Aggressive Early Stopping
- **Action:** Implement Early Stopping based on validation MAE.
- **Parameters:** `patience = 3` (for fractional runs) or `patience = 5` (for full Fold 0 runs).
- **Why:** Prevents wasting GPU hours on models that have already plateaued.

### Principle 4: Telemetry is Non-Negotiable
Even on a 10% dataset, the pipeline MUST produce the 4 core artifacts:
1. `history_fold_0.csv` (Train/Val Loss, Val MAE per epoch)
2. `preds_fold_0.csv` (y_true vs y_pred)
3. `best_model_fold_0.pth` (Weights)
4. `results.json` (Summary metrics)

---

## 3. Concrete Instructions for the Coder

**Coder Task: Implement the "Fast Iteration Protocol" in `scripts/run_chgnet_structure.py`**

1. **Implement `fractional_data` Flag:**
   - Add an argument (e.g., `--fraction 0.1` or via YAML config).
   - In the data loading logic for the *training* and *validation* splits, if `fraction < 1.0`, randomly select that percentage of indices before creating the DataLoader.
   - *Crucial:* Set a fixed random seed for the subset selection to ensure reproducible "mini-datasets" across different runs.

2. **Implement `single_fold` Override:**
   - Modify the loop over `task.folds`. Add a flag (e.g., `--fold 0`) that forces the script to *only* run that specific fold and then exit successfully.
   - Do NOT attempt to run the full Matbench `record()` or `to_file()` aggregation if we only ran one fold, as Matbench expects all 5. Save our custom telemetry instead.

3. **Enforce Early Stopping:**
   - Ensure the training loop monitors validation MAE and stops if it does not improve for `patience` epochs.

4. **Workflow Execution Rule (From now on):**
   - **Phase 1 (Debug/Sweep):** `fraction=0.1`, `fold=0`, `epochs=10` -> Runtime: ~5 mins. Use this to sweep LRs, fix bugs, test new layers.
   - **Phase 2 (Promising Candidate):** `fraction=1.0`, `fold=0`, `epochs=30` (with early stopping) -> Runtime: ~40 mins. Use this to confirm the metric is approaching Top-1 territory.
   - **Phase 3 (Final Submission):** `fraction=1.0`, `folds=all`, `epochs=50` -> Runtime: ~5 hours. Run this overnight *only* when Phase 2 succeeds.

*(End of Instructions)*