# 2026-02-26 — Analysis: L4 GPU Runtime Anomaly (I/O Bottleneck)

## 1. Root Cause Analysis (Extreme Runtime on L4)
**The Problem:** The `chgnet_head_finetune_freeze` run (10 epochs, frozen backbone) has taken over 7 hours without completing, despite using an NVIDIA L4 GPU.
**The Cause:** This is a classic **I/O Bottleneck (Storage Thrashing)**.
- Previously, to solve the Out-Of-Memory (OOM) crash, we instructed the Coder to implement a "Disk-First" lazy-loading strategy. This involved saving each of the ~132,000 graphs as an individual `.pt` file.
- During training, the PyTorch `DataLoader` must open, read, deserialize, and close a file for *every single graph* in every batch, across every epoch.
- **The Math:** 132k files × 10 epochs = **1.32 million individual file read operations**.
- Google Colab's file system (especially if the cache was accidentally placed on Google Drive instead of local `/content` SSD, or due to inherent overhead of Python's file I/O) cannot handle this volume of micro-reads efficiently. The GPU is likely sitting idle at 0% utilization while waiting for the CPU/Disk to fetch the next batch.

## 2. Proposed Solution: Fast I/O Caching
Reading 130,000 tiny files per epoch is the wrong paradigm for deep learning. We need a format optimized for fast sequential or chunked reads.

**Option A: In-Memory Tensor Stacking (If fits in RAM)**
- Instead of keeping heavy Python `CrystalGraph` objects in memory, we extract their underlying PyTorch tensors (nodes, edges) and stack them into a single massive `torch.Tensor`. This is what `PyTorch Geometric` does (`InMemoryDataset`). It is highly efficient but might still hit the 16GB RAM limit if not careful.

**Option B: LMDB / HDF5 (Memory-Mapped Storage)**
- Store all graphs in a single database file (like LMDB or HDF5) that supports lightning-fast memory-mapped reads. The OS handles caching the active parts in RAM.

**Option C: Chunked `.pt` Files (The Pragmatic Middle Ground)**
- Instead of 132,000 files (1 graph per file), we save **chunks of 1,000 graphs** per file.
- The PyTorch `Dataset` loads a chunk into RAM, yields the 1,000 graphs iteratively, and then loads the next chunk.
- This reduces file I/O operations by 1,000x (from 1.3 million to just 1,320 reads).

**Decision:** We will proceed with **Option C (Chunked Caching)**. It is the easiest to implement quickly in PyTorch without requiring complex external database libraries like LMDB.

## 3. Next Steps (Instructions for Coder)
The current 1-graph-per-file approach must be replaced with a chunked caching system.

1. **Modify Graph Caching:**
   - Group the 132,752 `pymatgen.Structure` objects into chunks of 1,000.
   - Convert each chunk into a list of CHGNet graphs.
   - Save the list as a single file: `chunk_001.pt`, `chunk_002.pt`, etc.
2. **Modify the Dataset/DataLoader:**
   - The Dataset should load an entire `chunk_*.pt` into memory only when an index within that chunk is requested.
   - Maintain an LRU (Least Recently Used) cache of 1-2 chunks in memory to prevent reloading the same chunk multiple times if the DataLoader shuffles data. (Or pre-shuffle indices *within* chunks to keep loading sequential).
   - Alternatively, use PyTorch's `IterableDataset` which is perfectly suited for reading chunked files sequentially.

*(End of Report)*