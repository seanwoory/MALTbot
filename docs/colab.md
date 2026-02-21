# Daily Colab Run

Canonical notebook: `notebooks/MALTbot_2.ipynb`

## Setup
- Colab runtime: **GPU**
- Colab Secrets: add `GH_TOKEN` (repo push scope)

## Daily flow
1. Open `notebooks/MALTbot_2.ipynb`
2. Edit only the **CONFIG** cell (`DATE`, `RUN_NAME`, `TASK`, `SEED`, `MODEL_CONFIG`, `NOTE`)
3. Run all cells
4. Notebook writes:
   - `results/daily/<DATE>/<RUN_NAME>/results.json`
   - one line appended to `RESULTS.md`
5. Notebook pushes to branch: `colab/<DATE>/<RUN_NAME>`
6. Open printed compare URL and create PR (never push to `main`)

## Token handling
- Primary: `google.colab.userdata.get('GH_TOKEN')`
- Fallback: hidden prompt (`getpass`) if secret missing
- Token is never printed
