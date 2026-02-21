# MALTbot

## Documentation

- [Matbench Ops Rules](docs/MATBENCH_OPS.md)
- [Colab notebook: CHGNet finetune track (`matbench_mp_e_form`)](notebooks/route_b_chgnet_matbench_mp_e_form_colab.ipynb)

## CHGNet finetune track (`matbench_mp_e_form`)

Minimal, reproducible experiment scaffold for Matbench v0.1:

- Runner: `scripts/run_chgnet_mp_e_form.py`
- Config: `configs/chgnet_mp_e_form.yaml`
- Output artifact: `results/daily/YYYY-MM-DD/<run_name>/results.json`
- Scoring: `matbench` `record()` API (`task.scores` written to results JSON)

## Quick run (Colab)

Use the notebook above and run all cells.

It handles:
- GPU check
- clone/pull (rerun-safe)
- pinned dependency install (Colab Python 3.12)
- Drive mount
- training/evaluation run
- result discovery
- PR-only branch commit/push using `GH_TOKEN` from Colab Secrets

## PR-only results update (CLI)

```bash
python scripts/run_chgnet_mp_e_form.py --config configs/chgnet_mp_e_form.yaml

DATE_KST="$(TZ=Asia/Seoul date +%Y%m%d)"
RUN_TAG="chgnet-mp-e-form"
BRANCH="bot/${DATE_KST}-${RUN_TAG}"

git checkout -B "${BRANCH}"
git add RESULTS.md results/daily/**
git commit -m "results: ${RUN_TAG} ${DATE_KST}"
git push -u origin "${BRANCH}"
```

Do not push directly to `main`; open a PR from the branch.
