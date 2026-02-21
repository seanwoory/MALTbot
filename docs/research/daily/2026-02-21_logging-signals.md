# 2026-02-21 — Matbench 회귀 최적화를 위한 로깅 신호(Logging Signals) 제안

## 질문
- Matbench 회귀 모델 최적화에서 **MAE만** 로깅하면 충분한가?
- 결론: **불충분**. MAE는 “최종 성능”만 보여주고, *왜 좋아졌는지/왜 망가졌는지*를 설명해주지 않음.

아래는 **Colab + 기존 `run_experiment.py` 파이프라인에 쉽게 붙일 수 있는**(스칼라/짧은 요약 중심) 최소 로깅 스키마 제안.

---

## 1) Minimal high-value logging schema (beyond MAE)

### Must-have (오버헤드 낮고, 의사결정에 필수)
> 목적: “실험이 좋아졌는지”뿐 아니라 “과적합/불안정/비용”을 즉시 판단.

#### A. 성능 지표(평가)
- **RMSE**
- **Max error** (또는 `max_abs_error`)
- **Abs error percentiles**: p50, p90, p95, p99
- **(가능하면) per-fold metrics** + **aggregate(mean/std across folds)**

왜 도움이 되나?
- RMSE는 큰 오차(outlier)에 더 민감 → MAE가 비슷해도 tail이 나빠진 경우 탐지.
- Max error/상위 퍼센타일(p95/p99)은 *“리스크(최악 사례)”*를 보여줌. 구조/전처리 불안정성, 모델 발산 징후를 빨리 잡음.
- fold별 기록은 특정 fold에서만 망가지는 “데이터 분할 민감도”를 확인.

#### B. 학습 안정성/일반화 신호
- **best_epoch** (val 기준)
- **train_loss@best**, **val_loss@best** (또는 train/val MAE@best)
- **generalization_gap** = (val_metric - train_metric) @ best
- **early_stopped** (bool) + **patience**

왜 도움이 되나?
- best_epoch가 너무 이르면 underfit, 너무 늦으면 과적합/학습률 스케줄 문제.
- gap은 “성능 개선이 진짜 일반화인지(=val 개선)” vs “train만 과하게 맞춘 건지”를 판별.

#### C. 런 비용/자원(lean budget 최적화)
- **wall_time_sec** (전체)
- **train_time_sec_per_epoch** (평균 or median)
- **num_params** (가능하면)
- **hardware summary**: `device`(cpu/cuda), `gpu_name`, `mixed_precision`(on/off)

왜 도움이 되나?
- 리더보드에서는 성능만 보지만, 연구/개발에서는 “시간 대비 이득”이 중요.
- 동일 MAE라면 더 싼 설정(배치/모델/precision)을 선택.

#### D. 재현성/추적에 필수인 메타
- **run_id** (timestamp+hash)
- **task_name** (e.g., matbench_mp_e_form)
- **seed**
- **git_commit** (가능하면) / 코드 버전
- **hyperparams (flat dict)**: lr, wd, batch_size, epochs, scheduler, loss, augmentation flags 등
- **target_transform**: none / standardize(mean,std) / log 등 + (mean,std 값)

왜 도움이 되나?
- 같은 MAE라도 “어떤 설정”이었는지 재현/비교가 불가능하면 최적화가 느려짐.


### Nice-to-have (오버헤드 중~상, 있으면 분석/논문화에 강함)
> 목적: “왜” 좋아지는지 설명하고, 논문용 분석(오류 구조)을 뽑기 쉬움.

#### E. 예측 분포/편향 진단
- **pred_mean, pred_std, pred_min, pred_max** (test 예측)
- **target_mean, target_std** (test target은 접근 불가한 경우 제외; 대신 train/val만)
- **residual_mean** (mean(y_pred - y_true)) on val (가능할 때)

도움
- 예측이 특정 범위에 “눌리는” 현상(예: 분산 붕괴) 탐지.

#### F. 에러-타깃 의존성(짧은 요약)
- **abs_error_by_target_quantile**: 예) y_true를 5분위로 나눠 각 bin의 MAE

도움
- 특정 에너지 구간(예: 불안정 상)에서만 성능이 나쁜지 확인 → loss/샘플 가중치/transform 방향 제시.

#### G. 불확실성/앙상블 관련
- (앙상블 시) **ensemble_size**
- **disagreement stats**: 예) 예측 표준편차의 mean/p90
- (가능하면) **correlation(disagreement, abs_error)** on val

도움
- 불확실성 기반 샘플 가중치/active learning/증류 설계의 근거.

#### H. 학습 곡선 요약(전체 로그 대신)
- 마지막 N epoch의 val metric trend: `val_mae_last5 = [..]` 같은 **짧은 배열**

도움
- 발산/진동(학습률 과다) vs 안정 수렴을 빠르게 파악.

---

## 2) `results.json`에 추가할 정확한 필드(제안)

> 기존에 `results.json`이 “한 run 결과”를 담고 있다고 가정하고, 스칼라/짧은 요약만 추가.

### Must-have: JSON schema (핵심)
```json
{
  "run_id": "2026-02-21T10:16:00Z_ab12cd",
  "task": "matbench_mp_e_form",
  "seed": 0,
  "hyperparams": {
    "lr": 0.0003,
    "wd": 0.0001,
    "batch_size": 64,
    "epochs": 50,
    "scheduler": "cosine",
    "loss": "huber",
    "augmentation": "none|ssa",
    "amp": true
  },
  "target_transform": {
    "name": "standardize",
    "train_mean": -0.123,
    "train_std": 0.456
  },
  "fit": {
    "best_epoch": 23,
    "early_stopped": true,
    "patience": 10,
    "train_mae_at_best": 0.0123,
    "val_mae_at_best": 0.0181,
    "generalization_gap_mae": 0.0058
  },
  "metrics": {
    "folds": [
      {
        "fold": 0,
        "mae": 0.0180,
        "rmse": 0.050,
        "max_abs_error": 3.2,
        "abs_err_p50": 0.008,
        "abs_err_p90": 0.030,
        "abs_err_p95": 0.040,
        "abs_err_p99": 0.120
      }
    ],
    "aggregate": {
      "mae_mean": 0.0182,
      "mae_std": 0.0004,
      "rmse_mean": 0.051,
      "max_abs_error_max": 3.8,
      "abs_err_p95_mean": 0.041
    }
  },
  "compute": {
    "device": "cuda",
    "gpu_name": "T4",
    "wall_time_sec": 7420,
    "sec_per_epoch": 155,
    "num_params": 412525
  }
}
```

### Nice-to-have: JSON 추가 필드
- `metrics.folds[*].pred_std` (예측 분산)
- `error_slices`: `mae_by_target_quantile` 같은 짧은 dict
- `ensemble`: `{ "size": 3, "disagreement_mean": ..., "disagreement_p90": ... }`
- `fit.val_mae_last5`: 마지막 5 epoch의 val MAE 리스트(길이 5)

---

## 3) `RESULTS.md`에 무엇을 남길까?

### Must-have (짧고 강하게)
- 실험 1줄 요약(한 run = 한 줄)
  - `run_id`, `task`, `mae_mean ± std`, `rmse_mean`, `p95`, `max_abs_error`, `wall_time`, 핵심 변경점 1개

예시:
- `2026-02-21_ab12cd | mp_e_form | MAE 0.0182±0.0004 | RMSE 0.051 | p95 0.041 | max 3.8 | 2.1h | +SSA(TTA=4)`

### Nice-to-have
- “Top-3 runs so far” 섹션에 best 3개만 간단 비교(오염 방지: 긴 표 금지)

---

## 4) Colab + 기존 파이프라인 호환성 가이드
- **스칼라 위주**로 유지: fold별 metric도 p50/p90/p95/p99 정도만.
- 원본 예측 벡터 전체 저장 금지(용량/프라이버시/재현성 문제).
- `run_experiment.py`에서 이미 fold loop가 있다면:
  - 예측값으로 `abs_err = |y_pred - y_true|`만 계산 → percentile 추출은 `np.percentile`로 즉시 가능.
  - 시간은 `time.time()` 차이.

---

## 결론
- MAE는 최종 판단 지표이지만, **(1) tail(퍼센타일/최대오차), (2) 안정성(gap/best_epoch), (3) 비용(time/hw)** 없이는 최적화가 느려지고 “왜 좋아졌는지”를 설명하기 어렵다.
- 위 Must-have만 추가해도, 다음 주 5~7개 실험을 *훨씬 빠르게* 수렴시킬 수 있음.
