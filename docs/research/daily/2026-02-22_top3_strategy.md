# Matbench v0.1 mp_e_form Top-1 Strategy Re-evaluation (2026-02-22)

## 1. Competitive Landscape & SOTA Analysis

The current target is `matbench_mp_e_form` (Formation energy, 132k samples).

### 1.1 Leaderboard Top-5 (Official v0.1)
| Rank | Model | MAE (eV/atom) | Citations/Status |
| :--- | :--- | :--- | :--- |
| **#1** | **coGN** | **0.0170** | [Ruff et al. (2023)](https://arxiv.org/abs/2302.14102) |
| #2 | coNGN | 0.0178 | Same authors |
| #3 | ALIGNN | 0.0215 | [Choudhary et al. (2021)](https://www.nature.com/articles/s41524-021-00650-1) |
| #4 | SchNet | 0.0218 | [Schütt et al. (2018)](https://arxiv.org/abs/1706.08566) |
| #5 | DimeNet++ | 0.0235 | [Klicpera et al. (2020)](https://arxiv.org/abs/2011.14115) |

### 1.2 Unofficial/Modern SOTA (2024-2025)
Recent foundation models (Universal Potentials) have surpassed these:
- **MACE-MPA-0**: Reported MAE **~0.016** eV/atom on Matbench `mp_e_form`. [Batatia et al. (2024)](https://arxiv.org/abs/2401.00096)
- **CHGNet (Original)**: Reported MAE **0.023** (out of the box). [Deng et al. (2023)](https://www.nature.com/articles/s42256-023-00716-3)

### 1.3 Reproducibility Assessment
- **coGN/coNGN**: Code is public (`kgcnn` lib), but exact Matbench-v0.1-specific training configs are buried in the Matbench repo.
- **CHGNet**: **Highest reproducibility**. Pretrained weights and finetuning notebooks are provided officially.
- **MACE**: Very strong, but requires significant setup for Matbench fold loops compared to our current CHGNet-lite scaffold.

---

## 2. Strategy Re-evaluation: Is Full CHGNet the Best Path?

**1. Is it the most efficient?**
Yes. Swapping the "Composition MLP" for the "Structure GNN" within CHGNet is a **model swap, not a library swap**. It leverages the same `pymatgen` structure objects and `chgnet` library. Moving to MACE or ALIGNN would double the implementation time for a ~5% performance gain over a well-tuned CHGNet.

**2. Should we consider MACE/ALIGNN in parallel?**
Only if CHGNet finetuning plateaus above 0.020. MACE is superior for stability/MD, but for static property prediction (like `mp_e_form`), CHGNet is highly optimized for MP-like data.

**3. Review of 'Next 24h' (Seed 44, Epochs 80 on MLP)**:
**Decision: TERMINATE MLP optimization.** 
Optimizing an MLP that is physically limited to 0.10 MAE will not reach Top-1. We must move to the graph engine immediately.

---

## 3. NEW TOP 3 "Quantum Leaps" (Actionable)

### Leap #1: The Graph Engine Swap (Physical Reality)
- **Mechanism**: Shift from chemical formula to 3D crystal graph. This is the difference between MAE 0.11 and MAE 0.02.
- **Complexity**: Medium (Graph caching is the bottleneck).
- **Compute**: Lean (20 epochs on 132k samples takes ~2-4 hours on H100).
- **Coder Instructions**: Implement `FullCHGNetRunner`. Use `CHGNet.load()` and the `Trainer` class from the official repo. Ensure `Structure` objects are converted to graphs *once* and cached.

### Leap #2: Precision Target Alignment (The "Delta" Strategy)
- **Mechanism**: Matbench targets are **Formation Energy**, but CHGNet predicts **Total Energy**.
- **Novelty Angle**: Instead of retraining the whole GNN, we learn a **learned AtomRef (chemical potential)**. Use the pretrained GNN as a "frozen feature extractor" and train a robust head (Huber loss) to map structure features to formation energy.
- **Complexity**: Low.
- **Experiment**: `exp_chgnet_frozen_atomref`. Freeze the backbone, only train the composition-based correction.

### Leap #3: Stochastic Symmetry Averaging (SSA)
- **Mechanism**: Predict MAE by averaging 8 symmetric variants of the structure (rotation + small strain).
- **Novelty Angle**: "Invariance through averaging." This compensates for discretization errors in graph construction.
- **Complexity**: Low (Inference time only).
- **Experiment**: `exp_chgnet_tta8`. Use the best model from Leap #1 and apply 8-way TTA.

---

## 4. Finalized 24h Implementation Plan (Handover to Coder)

### Goal: Establish the "Structure-Based" Baseline
Do not run the old MLP experiments. Today's goal is to see **0.02x MAE**.

1. **Run#1 (PM): Zero-Shot Pretrained CHGNet**
   - **Variable**: `model_name = "chgnet_pretrained"`, `epochs = 0`.
   - **Task**: Load `CHGNet.load()`, predict `e_form` on all 5 folds using the default MP reference energies.
   - **Expected**: MAE ~0.025 - 0.035. (This confirms our structure pipeline is correct).

2. **Run#2 (PM): Frozen Backbone Finetuning**
   - **Variable**: `freeze_backbone = True`, `epochs = 10`.
   - **Task**: Train only the output MLP head and AtomRef.
   - **Expected**: MAE ~0.020 - 0.025.

3. **Run#3 (Night): Full Finetune (5 Folds)**
   - **Variable**: `freeze_backbone = False`, `lr = 1e-4`, `epochs = 20`.
   - **Goal**: Hit the **Top-5** (~0.020) immediately.

### Coder Implementation Requirements:
- Use `chgnet.model.model.CHGNet.load()` for weights.
- Implement a `MatbenchDataset` class that calls `chgnet.graph.converter.CrystalGraphConverter`.
- Use `torch.utils.data.DataLoader` with `batch_size=64` or higher.
- **Crucial**: The `e_form` target in Matbench is per-atom. Ensure `chgnet.predict_structure` output is compared to the correct scale.
- Save results using the fixed float-precision logging (`results.json`).
