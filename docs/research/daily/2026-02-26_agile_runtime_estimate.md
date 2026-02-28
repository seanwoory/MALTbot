# 2026-02-26 — Agile Runtime Estimation: `chgnet_pretrained_infer` (Fold 0 Only)

## 1. Graph Caching Time (Fold 0)
- **Scope:** Fold 0 uses ~106k structures for training and ~26k for testing. 
- **Time:** Since this is the zero-shot inference run (`epochs: 0`), the script only needs to convert and cache the **26k test structures**.
- **Estimate:** Converting 26k structures to CHGNet graphs on CPU takes roughly **15 to 20 minutes**.

## 2. GPU Inference Time (Fold 0)
- **Scope:** A single forward pass on the 26k cached test graphs using the L4 GPU.
- **Estimate:** With a reasonable batch size (e.g., 64-128), inference on an L4 is extremely fast, taking only **1 to 2 minutes**.

## 3. Total Expected Runtime
- **Total Time:** 15-20 mins (caching) + 1-2 mins (inference) + overhead.
- **User Expectation:** You should expect the results for the first model to appear on GitHub in **approximately 20 minutes**. 
- *Note:* Future runs on Fold 0 will skip the 20-minute caching step and finish in just minutes!