# 2026-02-25 — Analysis: Colab OOM Crash during Graph Conversion

## 1. Root Cause Analysis (OOM / SIGKILL)
- **The Error:** The Colab session crashed with `returncode -9` (SIGKILL). This is the classic signature of the Linux Out-Of-Memory (OOM) killer terminating a process that consumed all available system RAM.
- **Why it happened:** The previous instruction to the Coder was to perform "Global Graph Caching" by converting all 132,752 `pymatgen.Structure` objects into CHGNet graphs *at once* and storing them in a single list before saving. 
- **The Math:** A single `pymatgen.Structure` is relatively lightweight, but a converted `chgnet.graph.CrystalGraph` (which includes node features, edge indices, edge features, and potentially large dense tensors for distances/angles) consumes significantly more memory. 
  - 132k structures × ~100-500 KB per graph = **13-65 GB of RAM required**.
  - Standard Colab instances (even with L4 GPUs) typically provide 12.7 GB to 16 GB of system RAM. Holding all 132k graphs in memory simultaneously guarantees an OOM crash.

## 2. Proposed Memory-Efficient Strategy
We must abandon the "load-everything-into-RAM" approach and move to a **Disk-Backed, Chunked or Lazy-Loading Strategy**.

**Option A: Chunked Caching to Disk (Recommended)**
Instead of converting all structures into one list, we process them in chunks (e.g., 10,000 structures at a time) and save each chunk to disk immediately.
- *Pros:* Controls peak RAM usage perfectly.
- *Cons:* Slightly more complex I/O logic.

**Option B: Lazy Conversion (On-the-fly via PyTorch Dataset)**
Convert structures to graphs inside the `__getitem__` method of the PyTorch `Dataset`, perhaps with a local disk cache (e.g., `LMDB` or simply saving individual `.pt` files per graph).
- *Pros:* Zero upfront RAM overhead.
- *Cons:* Slower training loop if I/O becomes the bottleneck (though reading small `.pt` files from Colab's local SSD is usually fast enough).

**Decision:** We will use **Option A (Chunked Caching) combined with a custom PyTorch Dataset** that loads these chunks on demand, or **saving individual graphs as `.pt` files** to a cache directory. Saving individual files is highly parallelizable and robust.

## 3. Instructions for the Coder

**Coder Task: Fix OOM in `scripts/run_chgnet_structure.py`**

The current implementation loads all structures, tries to convert them all to graphs in memory, and then crashes. You must rewrite the data processing pipeline to be memory safe.

**Implement the "Disk-First" Graph Cache Strategy:**

1. **Create a Cache Directory:**
   - Define a cache path, e.g., `data/chgnet_graph_cache/matbench_mp_e_form/`.
   - Ensure this directory exists.

2. **Iterative/Chunked Conversion (Avoid RAM bloat):**
   - Iterate through the 132,752 `pymatgen.Structure` objects *one by one* (or in small batches of 1,000).
   - Check if the graph file already exists (e.g., `data/chgnet_graph_cache/.../graph_{index}.pt`).
   - If it does not exist:
     - Convert the single structure using `CrystalGraphConverter(atom_graph_cutoff=6.0, ...)` (wrapped in the `try-except` block for isolated atoms as discussed previously).
     - **Immediately save the graph to disk** using `torch.save(graph, file_path)` or CHGNet's internal graph save method.
     - **Delete the graph object from memory** (`del graph`).
   - By doing this iteratively, peak memory usage for conversion is practically zero.

3. **Update the PyTorch Dataset (`__getitem__`):**
   - The Dataset class should no longer hold a list of graphs in memory.
   - It should only hold a list of file paths (e.g., `['graph_0.pt', 'graph_1.pt', ...]`) and the corresponding targets (`y_true`).
   - In the `__getitem__(self, idx)` method:
     - Load the graph from disk: `graph = torch.load(self.filepaths[idx])`
     - Return `(graph, self.targets[idx])`.

4. **Colab Specifics:**
   - Ensure the cache directory is on the local disk (`/content/data/...`) rather than Google Drive to ensure fast read speeds during training.

*(End of Instructions)*