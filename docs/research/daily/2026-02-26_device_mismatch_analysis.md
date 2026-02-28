# 2026-02-26 — Analysis: PyTorch Device Mismatch Error (chgnet_full_finetune)

## 1. Root Cause Analysis (Device Mismatch)
**The Error:** `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`
**Context:** This occurred during the forward pass after implementing the global graph cache. The graphs are loaded from disk (`.pt` files) and passed to the model.
**Why it happens:** 
When PyTorch loads tensors or custom objects from disk using `torch.load()`, they are typically loaded onto the CPU by default (unless specified otherwise). In a standard PyTorch training loop, batch data (like `X, y`) is explicitly moved to the GPU using `X.to(device)`. 
However, CHGNet's `CrystalGraph` (or a batch of graphs) is a complex custom object containing multiple internal tensors (e.g., `atom_features`, `edge_index`, `bond_features`, etc.). A generic `.to(device)` call on a list of graphs or a standard PyTorch `DataLoader` collation will not automatically recursively move all internal tensors of custom objects to the GPU. Consequently, the model weights are on `cuda:0`, but the input graph features remain on `cpu`.

## 2. Proposed Fix
To fix this, we need to explicitly move the graph objects to the correct device right before feeding them into the model in the training loop. If we are using CHGNet's internal structures, `CrystalGraph` or `GraphBatch` usually implement a `.to(device)` method. If using a custom PyTorch loop, the batch of graphs must be iterated and explicitly transferred.

## 3. Instructions for the Coder

**Coder Task: Fix Device Mismatch in `scripts/run_chgnet_structure.py`**

1. **Locate the Forward Pass:**
   Find the training and validation loops in `scripts/run_chgnet_structure.py` where the data is yielded from the `DataLoader` and passed to the model (e.g., `outputs = model(graphs)`).

2. **Explicitly Transfer Graphs to Device:**
   Before the forward pass, ensure every graph in the batch is moved to the target device (e.g., `cuda`).
   - If `graphs` is a list of `CrystalGraph` objects:
     ```python
     graphs = [g.to(device) for g in graphs]
     targets = targets.to(device)
     ```
   - If using a CHGNet `GraphBatch` object or a specific collate function, ensure you call `.to(device)` on the batched object.

3. **Check the Collate Function:**
   Ensure that the `DataLoader` uses the correct `collate_fn` (e.g., `collate_graphs` from CHGNet) to properly batch the loaded `.pt` files.

4. **Verify Target Tensors:**
   Make sure the `e_form` target tensors (`y_true`) are also moved to the same device as the model (`cuda`) before calculating the loss.

By explicitly routing the loaded graph components to the GPU, the mismatch between the `cuda:0` model weights and the `cpu` input data will be resolved.