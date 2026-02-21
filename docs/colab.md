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

## Output behavior
- Each experiment writes:
  - `results/daily/<DATE>/<BATCH_RUN_NAME>/<exp_name>/results.json`
- Each experiment appends one line to `RESULTS.md`
- Successful runs write numeric MAE (`metric_value`) using explicit fold MAE aggregation and/or `FINAL_METRIC_MAE` marker fallback.
- Disabled / not-implemented experiments are logged as `status=skipped` and `METRIC=SKIPPED`

## Git push behavior
- Branch: `colab/<DATE>/<BATCH_RUN_NAME>`
- Never pushes to `main`
- Uses `GH_TOKEN` from `google.colab.userdata.get('GH_TOKEN')`
- Fallback: hidden prompt (`getpass`) if secret is missing
- Token is never printed
