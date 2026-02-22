# 2026-02-21 (mp_e_form) — Next 24h Strategy Update

## 확인한 입력 (레포에서 직접 확인)
- `RESULTS.md` 최신 라인: 2026-02-21 daily10 실험들이 기록되어 있으나 다수가 **METRIC=None/placeholder** 또는 **SKIPPED**로 남아 있음.  
  - 근거: `/Users/sungwoo/Desktop/Career/git/MALTbot/RESULTS.md` (tail)
- `results/daily/2026-02-21/daily10/baseline_chgnet` 최신 baseline 결과는 **top-level 결과 파일에는 metric이 비어 있고**, 날짜 폴더에 실제 점수가 저장되어 있음.
  - top-level: `results/daily/2026-02-21/daily10/baseline_chgnet/results.json` → `metric_value: null`
  - scored: `results/daily/2026-02-21/daily10/2026-02-21/baseline_chgnet/results.json` → **MAE mean=0.1112187908**
- 요청된 quick-test의 `METRIC=0.110898` 문자열은 현재 repo 내 텍스트 검색(단순 grep 기준)에서는 **직접 발견되지 않았음**. 다만 baseline의 fold-mean MAE=0.1112로 매우 근접해서, quick-test가 (a) 일부 fold/서브셋, (b) 반올림/다른 metric 계산 경로였을 가능성이 높음.

---

## 1) 오늘 결과 요약

### 유효했던 것(= 실제로 값이 달라졌고, 정보가 생긴 것)
- **seed 변화는 유의미한 변동을 만들었다.**
  - seed=42 (baseline): MAE mean **0.1112188** (std 0.00334)
  - seed=43 (`chgnet_seed43`): MAE mean **0.1087697** (std 0.00098)
  - 차이: **~0.00245 MAE 개선** (상대적으로 꽤 큼)
  - 근거 파일:
    - baseline scored: `results/daily/2026-02-21/daily10/2026-02-21/baseline_chgnet/results.json`
    - seed43 scored: `results/daily/2026-02-21/daily10/2026-02-21/chgnet_seed43/results.json`

### SKIPPED / 무의미했던 것(= 실행은 됐지만 실질적으로 동일 결과 or 미구현)
- **`chgnet_target_transform`, `chgnet_lr_schedule`, `chgnet_ema`는 현재 ‘placeholder’ 상태로 보임.**
  - 세 결과 파일의 score가 baseline과 **완전히 동일** (MAE mean=0.1112187908 등 전부 일치)
  - 따라서 오늘은 “레버 성능 검증”이 아니라, “파이프라인이 레버를 실제로 적용하지 않고 있다”는 사실만 확인.
  - 근거 파일:
    - `results/daily/2026-02-21/daily10/2026-02-21/chgnet_target_transform/results.json`
    - `results/daily/2026-02-21/daily10/2026-02-21/chgnet_lr_schedule/results.json`
    - `results/daily/2026-02-21/daily10/2026-02-21/chgnet_ema/results.json`
- `alignn_baseline`, `kgcnn_schnet`, `kgcnn_dimenetpp`, `ensemble_3seed`, `tta4`는 **SKIPPED (미구현)**로 기록됨.
  - 근거: `RESULTS.md` 최신 라인

### 해석(중요)
- 현재 `baseline_chgnet`은 이름과 달리 파일 내부 모델 설명이 **"Route-B CHGNet-lite (composition MLP baseline)"**로 되어 있어, 우리가 목표로 하는 “full CHGNet finetune”과는 거리(성능 갭)가 큼.
- 따라서 **다음 24h의 최우선은 ‘레버 튜닝’이 아니라 ‘레버가 실제로 켜지는 파이프라인/모델’ 확보**.

---

## 2) 내일 AM/PM 2-run 계획 (각 run에서 바꾸는 변수 1개)

> 전제: 지금 당장 레버(EMA/target transform/scheduler)가 적용되지 않는 상태이므로, 내일 2-run은 “성능 개선”보다 **원인 분리/분산 추정/다음 개발 우선순위 결정**이 목적.

- **Run A (AM): seed sweep 확장**
  - 변경 변수 1개: `seed=44` (baseline 설정 동일)
  - 목적: seed 분산 스케일 추정(최소 3개 seed 확보) → 이후 모든 실험에서 “개선인지 노이즈인지” 판별 기준 마련
  - 기대 방향: MAE가 0.108~0.112 사이에서 움직일 가능성 (seed43가 유독 좋았는지 확인)

- **Run B (PM): epochs 변경 (조기수렴/과적합 체크)**
  - 변경 변수 1개: `epochs 40 → 80` (seed는 43 고정 추천)
  - 목적: 현재 베이스라인(MLP)이 underfit인지, 더 학습하면 내려가는지 확인
  - 기대 방향:
    - MAE가 내려가면: 학습 부족/스케줄 필요 신호
    - MAE가 오히려 올라가면: 과적합/정규화(wd/dropout) 우선

---

## 3) Top-1 장기 전략 기준 — 다음 1주일 실험 큐(최대 6개) 업데이트

> 순서 기준: (1) 성능 갭을 줄이는 데 필요한 ‘기능 구현’, (2) 구현 후 바로 점수를 올릴 가능성이 높은 레버

1. **Full CHGNet finetune 경로 구현(진짜 Route-B)**
   - 현재는 CHGNet-lite MLP라 Top-1(0.0170)과 갭이 너무 큼.
2. **Target transform(표준화) 실제 적용 + inverse transform 검증**
   - 지금 placeholder였으므로, 적용되면 안정성/성능 개선 가능성 높음.
3. **LR schedule(cosine + warmup) 실제 적용**
   - 큰 데이터에서 수렴/일반화에 도움.
4. **EMA 적용(옵션) + 평가 경로 분리(EMA weights로 eval)**
   - 소폭 개선 + 분산 감소용.
5. **Seed ensemble(3 seeds) 파이프라인 구현 + 제출 포맷(ensemble 평균) 확립**
   - leaderboard gain이 가장 확실한 레버 중 하나.
6. **TTA(테스트 타임 augmentation) x4/x8 최소 구현**
   - 계산은 늘지만, 분산 감소로 MAE 개선 가능.

---

## 4) 왜 이게 다음 최선인지 (근거 3줄)
1. 오늘 데이터에서 실제로 변한 건 **seed뿐**이었고, EMA/스케줄/타깃변환은 **적용되지 않아** 최적화 신호가 없었다.
2. 현재 모델이 `CHGNet-lite (composition MLP)`라서 Top-1(0.0170 eV/atom)과의 격차가 너무 커 **튜닝보다 모델/파이프라인 교체가 우선**이다.
3. 따라서 다음 24h는 (i) seed 분산 기준 확보, (ii) 학습 길이 민감도 확인으로 **다음 주 구현/실험 우선순위를 수치로 확정**하는 게 효율적이다.
