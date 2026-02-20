# MALTbot

## Docs

- [Matbench Ops Rules](docs/MATBENCH_OPS.md)

## Route B (CHGNet) — `matbench_mp_e_form`

This repo includes a minimal, reproducible Route-B runner:

- Script: `scripts/run_chgnet_mp_e_form.py`
- Config: `configs/chgnet_mp_e_form.yaml`
- Output: `results/daily/YYYY-MM-DD/<run_name>/results.json`

> Notes:
> - The runner uses `matbench` `record()` scoring for official task metrics.
> - Current implementation is a lightweight PyTorch composition-feature baseline under the CHGNet route name (easy to run on Colab). Replace internals with full CHGNet finetuning as next step.

## Colab quickstart (Run all)

```bash
# 1) Clone
!git clone https://github.com/seanwoory/MALTbot.git
%cd MALTbot

# 2) Install deps
!pip install -U pip
!pip install torch matbench pymatgen pyyaml tqdm

# 3) Run
!python scripts/run_chgnet_mp_e_form.py --config configs/chgnet_mp_e_form.yaml

# 4) Inspect output
!find results -name results.json | sort
```

## PR-only flow (no `gh` auth required)

```bash
# Commit results on your branch and push
!git checkout -b bot/chgnet-mp-e-form-$(date +%Y%m%d)
!git add results/daily/** RESULTS.md
!git commit -m "results: route-B chgnet mp_e_form daily run"
!git push -u origin HEAD
```

Then open GitHub in browser and create a PR from your branch to `main`.

## Optional: package results from Google Drive

```bash
# Example: copy Drive-produced results.json into repo path
!bash scripts/package_drive_results.sh \
  /content/drive/MyDrive/MALTbot-results/chgnet_run/results.json \
  chgnet_mp_e_form_route_b
```
