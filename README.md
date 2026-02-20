# MALTbot

## Docs

- [Matbench Ops Rules](docs/MATBENCH_OPS.md)

## CHGNet Fine tuning

This repo includes a minimal, reproducible runner:

- Script: `scripts/run_chgnet_mp_e_form.py`
- Config: `configs/chgnet_mp_e_form.yaml`
- Output: `results/daily/YYYY-MM-DD/<run_name>/results.json`
- Colab notebook: [`notebooks/route_b_chgnet_matbench_mp_e_form_colab.ipynb`](notebooks/route_b_chgnet_matbench_mp_e_form_colab.ipynb)

> Notes:
> - The runner uses `matbench` `record()` scoring for official task metrics.
> - Current implementation is a lightweight PyTorch composition-feature baseline under the CHGNet route name (easy to run on Colab). Replace internals with full CHGNet finetuning as next step.

## Colab Run-all 셀 플랜 (Route-B / `matbench_mp_e_form`)

아래 블록을 **순서대로 각각 Colab 셀**에 붙여넣고 `Run all` 하세요.
(재실행에도 최대한 안전하게 동작하도록 작성됨)

### 셀 0) GPU 확인

```python
import torch, platform
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

### 셀 1) 저장소 clone/pull (재실행 대응)

```bash
set -euo pipefail
REPO_DIR="/content/MALTbot"
REPO_URL="https://github.com/seanwoory/MALTbot.git"

if [ -d "${REPO_DIR}/.git" ]; then
  echo "[info] Repo exists. Pull latest main..."
  cd "${REPO_DIR}"
  git fetch origin
  git checkout main || true
  git pull --ff-only origin main
else
  echo "[info] Cloning repo..."
  git clone "${REPO_URL}" "${REPO_DIR}"
  cd "${REPO_DIR}"
fi

pwd
git rev-parse --short HEAD
```

### 셀 2) 의존성 설치 (Colab Python 3.12 호환 핀)

```bash
set -euo pipefail
python -m pip install -U pip setuptools wheel
python -m pip install \
  "numpy==1.26.4" \
  "scipy==1.11.4" \
  "pandas==2.2.2" \
  "scikit-learn==1.5.2" \
  "pymatgen==2024.8.9" \
  "matbench==0.6" \
  "torch==2.5.1" \
  "pyyaml==6.0.2" \
  "tqdm==4.66.5"
```

### 셀 3) Google Drive 마운트

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
```

### 셀 4) 실행

```bash
set -euo pipefail
cd /content/MALTbot
python scripts/run_chgnet_mp_e_form.py --config configs/chgnet_mp_e_form.yaml
```

### 셀 5) 결과 파일 확인

```bash
set -euo pipefail
cd /content/MALTbot
find results -type f -name "results.json" | sort
```

### 셀 6) PR 전용 커밋/푸시 (main 직접 푸시 금지)

> Colab 좌측 **Secrets**에 `GH_TOKEN`을 추가해두세요.
> 아래 코드는 토큰을 출력하지 않습니다.

```python
from google.colab import userdata
import os

token = userdata.get('GH_TOKEN')
if not token:
    raise ValueError("Colab Secret 'GH_TOKEN'이 비어 있습니다.")
os.environ['GH_TOKEN'] = token
print("GH_TOKEN loaded from Colab Secrets")
```

```bash
set -euo pipefail
cd /content/MALTbot

# Git identity (필요 시 수정)
git config user.name "colab-bot"
git config user.email "colab-bot@users.noreply.github.com"

DATE_KST="$(TZ=Asia/Seoul date +%Y%m%d)"
RUN_TAG="routeb-chgnet-mp-e-form"
BRANCH="bot/${DATE_KST}-${RUN_TAG}"

# 브랜치 생성/전환 (재실행 대응)
git checkout -B "${BRANCH}"

# 결과 파일만 스테이징
if [ -f RESULTS.md ]; then
  git add RESULTS.md
fi
# shellcheck disable=SC2016
git add results/daily/** || true

if git diff --cached --quiet; then
  echo "[info] Commit할 변경사항이 없습니다."
else
  git commit -m "results: ${RUN_TAG} ${DATE_KST}"

  # 토큰을 URL에 직접 노출하지 않고 http header로 push
  git -c http.https://github.com/.extraheader="AUTHORIZATION: bearer ${GH_TOKEN}" \
    push -u origin "${BRANCH}"

  echo "[done] Pushed branch: ${BRANCH}"
  echo "PR 생성: https://github.com/seanwoory/MALTbot/compare/main...${BRANCH}?expand=1"
fi
```

주의: 이 플로우는 **PR 전용**입니다. `main` 브랜치에 직접 push하지 마세요.

## Optional: package results from Google Drive

```bash
# Example: copy Drive-produced results.json into repo path
!bash scripts/package_drive_results.sh \
  /content/drive/MyDrive/MALTbot-results/chgnet_run/results.json \
  chgnet_mp_e_form_route_b
```
