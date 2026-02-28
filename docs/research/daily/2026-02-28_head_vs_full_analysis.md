# 2026-02-28 15:32 — CHGNet Finetuning Strategy (Head vs Full Discrepancy)

## Current Status (mp_e_form, fold0, frac=0.1)
*   **`head_finetune_freeze`** (freeze=True, lr=1e-3, epochs=3): Val MAE **1.048 → 0.664 → 0.463**, Test MAE = **0.467**
*   **`full_finetune`** (freeze=False, lr=1e-4, epochs=3): Val MAE **4.520 → 4.056 → 3.348**, Test MAE = **3.281**

## 1) Why is `full_finetune` significantly worse? (Top 3 Hypotheses)
The full finetuning should ideally outperform or match the head-freeze model, given it has more capacity to learn. The fact that it starts with a much higher MAE (4.520 vs 1.048) and drops slowly suggests fundamental issues.

**Hypothesis A: Target/Offset Unlearning (Catastrophic Forgetting)**
*   **Reasoning:** CHGNet was pre-trained on total energies. The target `mp_e_form` is formation energy. In `head_finetune_freeze`, the pre-trained features are preserved, and only the final layer learns the mapping (or AtomRef offset) to formation energy. In `full_finetune`, backpropagating the loss through the entire network unfreezes everything, potentially destroying the carefully learned pre-trained representations before the final layer can learn the target offset.

**Hypothesis B: Inappropriate Learning Rate / Optimizer State**
*   **Reasoning:** `lr=1e-4` might be too high for the pre-trained backbone. Deep layers might be experiencing massive gradient updates relative to their optimal pre-trained weights, leading to instability (though the loss is decreasing, it started from a broken state). It needs discriminative learning rates or a much smaller LR (e.g., 1e-5) for the backbone.

**Hypothesis C: Weight Initialization / Loading Issue in `full_finetune`**
*   **Reasoning:** Notice that `full_finetune` starts Epoch 1 Val MAE at **4.520**. This is suspiciously close to the zero-shot inference MAE (~4.75). However, `head_finetune_freeze` starts at **1.048**. If both loaded the exact same weights and just varied `requires_grad`, the *initial* loss before the first step should be identical. The fact that `head_finetune_freeze` gets to 1.048 in Epoch 1 means it adapted incredibly fast in just one epoch, whereas `full_finetune` struggled.

---

## 2) Minimal Experiments to Validate the Hypotheses (1 Variable Each)

**Experiment 1: Test Hypothesis B (Learning Rate)**
*   **Goal:** See if a smaller learning rate prevents the backbone from "breaking" during full finetuning.
*   **Config:** `full_finetune`, `fold0`, `fraction=0.1`, `epochs=3`, `bs=8`.
*   **Change (1 variable):** `lr = 1e-5` (decrease from 1e-4).
*   **Expected Result:** If the initial Val MAE is lower than 4.520 or drops much faster, the previous LR was destroying the backbone.

**Experiment 2: Test Hypothesis A/C (Head-Start / 2-Stage Training)**
*   **Goal:** Provide the network with the correct final-layer offset *before* unfreezing the backbone.
*   **Config:** `full_finetune`, `fold0`, `fraction=0.1`, `epochs=3`, `bs=8`.
*   **Change (1 conceptual variable):** Load the weights from the *best checkpoint* of the successful `head_finetune_freeze` run (MAE 0.467) as the starting point, then run `full_finetune` with a small `lr=5e-5`.
*   **Expected Result:** If it starts with MAE < 0.467 and improves, it proves that catastrophic forgetting of the target offset was the issue.

---

## 3) Strategy towards SOTA (Next 24h: 3-Run Plan)

Given that `head_finetune_freeze` is already working incredibly well (0.467 in just 3 epochs), the most efficient path to SOTA is to build upon it, rather than fighting the unstable `full_finetune` from scratch.

### The "Progressive Unfreezing" Strategy (Low Risk, High Reward)

**Run 1: Push `head_finetune_freeze` to its limit (Baseline Setup)**
*   **Why:** 3 epochs is too short. We need to see where it plateaus.
*   **Settings:** `head_finetune_freeze`, `fold0`, `frac=0.1`, **`epochs=10`**, `bs=8`, `lr=1e-3`, `early_patience=3`.
*   **Gate:** Save the best checkpoint. If MAE drops below 0.3, this is a very strong signal.

**Run 2: 2-Stage Finetuning (Implementation constraint: manual weight transfer)**
*   **Why:** Once the head has learned the formation energy mapping (from Run 1), we gently unfreeze the rest of the network to fine-tune the structural representations.
*   **Settings:** `full_finetune`, `fold0`, `frac=0.1`, `epochs=5`, `bs=8`.
*   **Crucial Step:** **Load the checkpoint from Run 1**. Set `lr=1e-5` (very small, to not break the backbone).
*   **Gate:** If MAE drops further than Run 1, this confirms the 2-stage approach is the winning architecture.

**Run 3: (If 2-Stage is hard to implement immediately) Head-Freeze with Cosine Annealing**
*   **Why:** If Coder cannot easily implement checkpoint loading between different configs today, maximize the head-only learning.
*   **Settings:** `head_finetune_freeze`, `fold0`, `frac=0.1`, `epochs=15`, `bs=8`, `lr=1e-3` but **add CosineAnnealingLR scheduler**.
*   **Gate:** Compare against Run 1.

**Implementation note for Coder:** Focus on making checkpoint loading seamless between `head_freeze` and `full_finetune` modes. This "Linear Probing then Fine-Tuning (LP-FT)" technique is standard practice when transferring foundation models to new tasks.