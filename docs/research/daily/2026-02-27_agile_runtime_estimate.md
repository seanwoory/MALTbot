# 2026-02-27 — Agile Runtime Estimation: Finetuning Models (L4 GPU)

## Context
- **Hardware:** NVIDIA L4 GPU
- **Pipeline:** "Agile Pipeline" (Fold 0 Only, pre-cached graphs)
- **Dataset (`mp_e_form` Fold 0):** ~106k train, ~26k validation.

## Scenario A: Full Fold 0 Data (Fraction = 1.0)
This uses the entire 106k training graphs per epoch.

1. **`chgnet_head_finetune_freeze` (Max 10 Epochs)**
   - **Characteristics:** Backbone is frozen (`requires_grad=False`). Only the readout/AtomRef head computes gradients. Forward pass is fast.
   - **Time per epoch:** ~2-3 minutes.
   - **Total Estimated Time:** **20 - 30 minutes** (less if Early Stopping triggers).

2. **`chgnet_full_finetune` (Max 20 Epochs)**
   - **Characteristics:** Full backpropagation through the entire GNN structure. High compute load.
   - **Time per epoch:** ~6-8 minutes.
   - **Total Estimated Time:** **120 - 160 minutes (~2 to 2.5 hours)** (less if Early Stopping triggers).

## Scenario B: "Speed Run" Fractional Data (Fraction = 0.1)
This uses only 10% of Fold 0 (~10.6k train graphs) for rapid hyperparameter ranking and debugging.

1. **`chgnet_head_finetune_freeze` (Max 10 Epochs)**
   - **Time per epoch:** ~15-20 seconds.
   - **Total Estimated Time:** **3 - 4 minutes**.

2. **`chgnet_full_finetune` (Max 20 Epochs)**
   - **Time per epoch:** ~40-50 seconds.
   - **Total Estimated Time:** **13 - 17 minutes**.

## Summary for User
With graph caching complete, the I/O bottleneck is removed. 
- If you run the **10% fractional speed run** to debug or sweep parameters, both models will finish in under **20 minutes total**.
- If you run the **full 100% Fold 0**, the frozen head model will take about **half an hour**, and the full finetune will take roughly **2 to 2.5 hours**.