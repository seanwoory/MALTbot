# Daily Colab Run (Batch)

Canonical notebook: `notebooks/MALTbot_2.ipynb`

## Setup
- Colab runtime: **GPU**
- Colab Secrets: add `GH_TOKEN` (repo push scope)

## One-click daily batch
1. Open `notebooks/MALTbot_2.ipynb`
2. Edit only the **CONFIG** cell:
   - `DATE`
   - `BATCH_RUN_NAME`
   - `TASK`
   - `EXPERIMENTS` (list of experiment names from `configs/experiments/*.yaml`)
   - `GH_PUSH`
3. Run all cells

Example experiment list: `baseline_chgnet`, `mlp_pretrained_infer_fallback`, `mlp_head_finetune_freeze`, `chgnet_ensemble3`, `chgnet_seed43`, `chgnet_seed44`, `chgnet_lr_schedule`, `chgnet_target_transform`, `chgnet_ema`, `chgnet_epochs80_seed43`.

Expected runtime (GPU, rough):
- `baseline_chgnet`: medium
- `mlp_pretrained_infer_fallback`: short-medium (MLP fallback, NOT CHGNet pretrained)
- `mlp_head_finetune_freeze`: short (partial fold0 quick gate, MLP)
- `chgnet_ensemble3`: long (3-seed averaging)

Naming policy: `chgnet_pretrained_infer`, `chgnet_head_finetune_freeze` are reserved for future true-CHGNet implementations and are currently disabled.

## Output behavior
- Each experiment writes:
  - `results/daily/<DATE>/<BATCH_RUN_NAME>/<exp_name>/results.json`
- Each experiment appends one line to `RESULTS.md`
- Successful runs write numeric MAE (`metric_value`) using explicit fold MAE aggregation and/or `FINAL_METRIC_MAE` marker fallback.
- Disabled / not-implemented experiments are logged as `status=skipped` and `METRIC=SKIPPED`

## Git push behavior
- Branch: `colab-<DATE>-<BATCH_RUN_NAME>` (slash-free)
- Never pushes to `main`
- Uses `GH_TOKEN` from `google.colab.userdata.get('GH_TOKEN')`
- Fallback: hidden prompt (`getpass`) if secret is missing
- Token is never printed

For branch watchers, query prefix as `colab-<DATE>-` (example: `branches/all?query=colab-2026-02-21-`).

If push fails (auth/network), rerun the token cell first, then rerun preflight + push cells.

Backward compatibility: old branches like `colab/<DATE>/<BATCH>` may still exist. If needed, open PR manually from GitHub UI (branch selector) or URL-encode slashes in compare links.