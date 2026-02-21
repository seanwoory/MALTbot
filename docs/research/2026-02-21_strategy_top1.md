# 2026-02-21 — Matbench v0.1 Top-1 전략 노트 (Researcher)

## TL;DR (결정)
- **1차 타깃은 `matbench_mp_e_form` 유지**가 합리적: 현재 Top-1이 **MAE 0.0170 eV/atom**까지 내려가 있고(아주 빡센데), 대규모(132k)라서 *“전처리/학습 프로토콜/앙상블/증류”* 같은 방법론 기여를 논문화하기 좋음.
- 다만 **`matbench_mp_gap`은 Top-1이 0.1559로 상대적으로 여유**가 있어, *빠른 Top-1 확보*가 최우선이면 mp_gap이 더 현실적일 수 있음.

---

## 1) Competitive landscape

### 1.1 리더보드 Top-1/Top-5 (mp_e_form)
- 리더보드(Per-task): **matbench_mp_e_form**
  - 링크: https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_e_form/
- Top-5 (mean MAE, eV/atom)
  - **#1 coGN — 0.0170**
  - **#2 coNGN — 0.0178**
  - **#3 ALIGNN — 0.0215**
  - **#4 SchNet (kgcnn v2.1.0) — 0.0218**
  - **#5 DimeNet++ (kgcnn v2.1.0) — 0.0235**
  - (상위권이 0.02 전후라서, 단일 모델로 “그냥 CHGNet fine-tune”만으로 #1을 넘기는 건 확률이 높지 않음)

### 1.2 리더보드 Top-1/Top-5 (mp_gap)
- 리더보드(Per-task): **matbench_mp_gap**
  - 링크: https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_gap/
- Top-5 (mean MAE, eV)
  - **#1 coGN — 0.1559**
  - **#2 DeeperGATGNN — 0.1694**
  - **#3 coNGN — 0.1697**
  - **#4 ALIGNN — 0.1861**
  - **#5 MegNet (kgcnn v2.1.0) — 0.1934**

### 1.3 “재현 가능성(코드/하이퍼파라미터/가중치)” 관점 스냅샷
> Matbench의 강점은 **각 알고리즘 제출물에 대한 ‘Full Benchmark Data’ 페이지**가 존재하고, 거기에 소프트웨어 요구사항과 fold별 파라미터가 포함된다는 점. 하지만 *모든 제출이 ‘원저자 코드 그대로’는 아니고*, 일부는 “공식 matbench 프로토콜에 맞추기 위한 수정”이 들어갈 수 있음.

- **coGN/coNGN (현 Top-1/Top-2 on mp_e_form)**
  - Full benchmark data (coGN): https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_coGN/
  - Full benchmark data (coNGN): https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_coNGN/
  - 두 페이지 공통으로:
    - “**Raw data download and example notebook available on the matbench repo**” 문구가 있음 → *실행 가능한 레퍼런스가 공개되어 있다는 신호*
    - “Software Requirements”로 `kgcnn==3.0.0` + `graphlist` 특정 커밋 설치가 명시됨(재현에 유리)
  - 재현성 평가(내 판단):
    - **중간~높음**: 실행 노트북/의존성 핀(특정 커밋) 덕에 *환경만 맞추면* 재현 가능성이 높음.
    - 다만 실제로 “한 번에 그대로 재현되는지”는 GPU/seed/전처리 미세차에 민감할 가능성.

- **ALIGNN**
  - mp_e_form Top-3 / mp_gap Top-4에 있고, 공개 repo가 있음: https://github.com/usnistgov/alignn
  - 재현성 평가: **중간**
    - 공개 코드로 구현은 가능하지만, DGL 버전/환경 이슈가 종종 있고, matbench 프로토콜에 맞춘 정확한 설정을 재현해야 점수가 나옴.

- **SchNet / DimeNet++ / MegNet (kgcnn 기반 제출)**
  - 리더보드에 `kgcnn v2.1.0`로 명시됨(모델/구현체가 특정됨) → 재현성 자체는 상대적으로 좋은 편
  - mp_e_form 리더보드 링크(상단 참고): https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_e_form/

- **CHGNet (우리가 고려 중)**
  - CHGNet repo: https://github.com/CederGroupHub/chgnet
  - pretrained 로드 + fine-tune 예제 노트북(공식 제공): https://raw.githubusercontent.com/CederGroupHub/chgnet/main/examples/fine_tuning.ipynb
  - 재현성 평가: **높음(우리 파이프라인 내)**
    - 코드/체크포인트/예제 모두 공개되어 있고, 우리가 matbench split에 맞춰 학습/record만 잘 붙이면 됨.
    - 단, “리더보드 Top-1 넘는지”는 별개의 문제.

---

## 2) 리더보드 이득 + 논문 기여 둘 다 가능한 Attack Vector (2~3개)

### Vector A — **Cross-model ensemble → single-student distillation** (가장 “현실적인 Top-1 도구” + 논문화 가능)
- 핵심 아이디어
  - 여러 강한 teacher(예: CHGNet/ALIGNN/coGN 계열 중 재현 가능한 것)를 **앙상블**해서 리더보드 점수 최적화
  - 동시에 그 앙상블을 **단일 student(예: CHGNet-lite 또는 CHGNet 동일 아키텍처)**로 distill
  - 논문 기여 포인트: *“구조 ML에서 앙상블의 성능을 유지하면서 추론 비용을 줄이는 distillation 프로토콜”* + *불확실성(teacher disagreement) 활용한 샘플 가중치*
- 왜 MAE가 내려가나(메커니즘)
  - 앙상블은 일반적으로 **분산 감소**로 MAE 감소
  - distillation은 **soft target(teacher 평균)**로 학습해서 label noise/불연속성 완화
- 구현 복잡도 (MALTbot Colab)
  - **중간**
  - 필요한 것: fold별로 여러 모델 예측을 저장하고 평균낸 뒤, 그 평균을 student의 학습 타깃으로 사용
- 계산 예산 적합성
  - **lean-friendly(조건부)**
  - “teacher를 몇 개 돌리느냐”가 비용. 하지만 teacher를 *2종*만 써도 distillation의 이득은 종종 발생.
- 구체 실험 플랜(1일 2-run 스타일)
  - AM baseline: **CHGNet 단독 fine-tune** (현재 파이프라인)
  - PM 1-shot variant: **(CHGNet seed ensemble 3개) 평균 → 그 평균으로 student 1개 distill**
  - 기대 방향: MAE ↓, 특히 outlier/edge case에서 max_error도 줄 가능성

### Vector B — **Multi-task / multi-property finetuning within Matbench v0.1 (no external data)**
- 핵심 아이디어
  - 외부데이터 없이도 Matbench v0.1에는 구조 기반 회귀 task가 여러 개 존재
  - 한 모델 backbone을 공유하고, 여러 task 헤드를 붙여 **공유 표현 학습** 후 mp_e_form에 집중 fine-tune
  - 논문 기여 포인트: *“Matbench 내부 멀티태스크가 특정 타깃(mp_e_form/mp_gap)에 주는 이득”* + “어떤 조합이 도움이 되는지” 분석
- 왜 MAE가 내려가나(메커니즘)
  - mp_e_form은 데이터가 크지만, **일반화된 구조 표현(결합/국소환경/주기성)**을 학습하는 데 다른 물성도 regularizer로 작동
  - 특히 mp_gap 같이 전자구조 연관 타깃은 표현을 풍부하게 만들어 e_form에도 도움이 될 수 있음(가설)
- 구현 복잡도
  - **중간~높음**
  - CHGNet가 기본적으로 e/f/s/m 멀티타깃을 지원하긴 하지만(공식 노트북), Matbench의 다양한 property를 멀티헤드로 넣으려면 데이터 로딩/헤드 설계가 필요
  - (Matbench run/record API는 표준화돼 있음) https://matbench.materialsproject.org/How%20To%20Use/2run/
- 계산 예산 적합성
  - **중간**
  - 멀티태스크는 학습 스텝이 늘지만, “pretrain(짧게) + mp_e_form finetune”으로 budget을 통제 가능
- 구체 실험 플랜
  - AM baseline: mp_e_form 단일 task CHGNet
  - PM variant: **(mp_e_form + mp_gap) 2-task 멀티헤드로 3~5 epoch pretrain → mp_e_form만 10~20 epoch finetune**
  - 기대 방향: MAE 소폭↓(0.0005~0.002 수준이라도 Top-1 경쟁에서 큼)

### Vector C — **Symmetry/augmentation: “stochastic symmetry averaging (SSA)” + consistency loss**
- 핵심 아이디어
  - 결정 구조는 주기성/대칭이 강한데, 그래프 변환/neighbor cutoff가 미세하게 달라지면 예측이 흔들릴 수 있음
  - 학습 때 (i) 작은 lattice strain/atomic perturb를 주고, (ii) 예측이 크게 변하지 않도록 **consistency loss**를 추가
  - 추론 때도 여러 augmentation 샘플의 예측을 평균(테스트타임 augmentation) → 앙상블처럼 분산 감소
  - 논문 기여 포인트: *“주기 결정에서 그래프 전처리의 불안정성을 regularize하는 SSA/consistency”*
- 왜 MAE가 내려가나
  - 모델이 neighbor list 경계/셀 표현 변화에 덜 민감해짐 → 일반화 및 안정화
  - 테스트타임 평균은 분산 감소
- 구현 복잡도
  - **낮음~중간**
  - 입력 `Structure`에 작은 perturb/strain을 주는 함수만 추가하면 됨(단, 물리적으로 말이 되는 범위에서)
- 계산 예산 적합성
  - **lean-friendly**
  - training augmentation은 거의 비용 증가 없음
  - test-time averaging은 k배 비용(예: k=4~8로 제한)
- 구체 실험 플랜
  - AM baseline: 기존 CHGNet
  - PM variant: **training에서 perturb/strain 1개만 추가 + inference에서 4-sample 평균**
  - 기대 방향: MAE ↓, 특히 max_error tail 감소 가능

---

## 3) Attack vector별 실행 설계(Colab 기준) — 요약 표

| Vector | 메커니즘(왜 MAE↓) | 구현 난이도 | 예산 적합 | 1-shot 변형 정의(오늘 PM) | 기대 방향 |
|---|---|---:|---:|---|---|
| A 앙상블→증류 | 분산↓ + soft target로 noise 완화 | 중 | 중 | seed-ensemble 평균을 student가 따라가도록 distill | ↓↓↓ |
| B 멀티태스크 | 공유 표현 + regularization | 중~고 | 중 | (e_form+gap) 짧은 멀티태스크 pretrain 후 e_form finetune | ↓ |
| C SSA/consistency | 전처리/neighbor 불안정성 완화 + TTA 평균 | 저~중 | 고 | 작은 perturb/strain + TTA 4-sample 평균 | ↓↓ |

---

## 4) 다음 주 “최선의 단일 계획” (5~7 experiments, ablation 포함)

### 목표
- 1) **리더보드 성능을 실제로 끌어올릴 수 있는 레버**를 확인하고
- 2) 동시에 **arXiv용으로 ‘기여’가 되는 실험 축**을 남김(단순 튜닝이 아닌 프로토콜/방법론).

### 권장: 6개 실험(최대 7개)로 압축
> 각 실험은 *바꾸는 축이 명확*하고, 결과가 좋으면 곧바로 논문화 스토리로 연결되게 설계.

1. **E0 (Baseline)**: CHGNet single-model finetune (현재 파이프라인 고정점)
   - 목적: 이후 모든 개선의 기준
   - 참고(공식 fine-tune 예시): https://raw.githubusercontent.com/CederGroupHub/chgnet/main/examples/fine_tuning.ipynb

2. **E1 (Vector C-abl 1)**: Training augmentation만 추가(perturb/strain), TTA 없음
   - 목적: augmentation이 단독으로 주는 이득 측정

3. **E2 (Vector C-abl 2)**: E1 + **TTA 4-sample 평균**
   - 목적: 테스트타임 평균의 분산 감소 효과 분리 측정

4. **E3 (Vector A-abl 1)**: seed ensemble 3개 평균(증류 없음)
   - 목적: “순수 앙상블”이 어디까지 MAE를 내리는지(Top-1 현실성 판단)

5. **E4 (Vector A-abl 2)**: E3의 앙상블 평균을 teacher로 **single student distillation**
   - 목적: 성능 유지 vs 속도/단일 모델화. 논문 기여 핵심.

6. **E5 (Vector B-abl)**: (mp_e_form + mp_gap) 2-task 멀티헤드 pretrain(짧게) → mp_e_form finetune
   - 목적: 멀티태스크가 실제로 도움 되는지 빠르게 판별
   - mp_gap 리더보드 참고: https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_gap/

7. (옵션, 시간/예산 허용 시) **E6 (Hybrid)**: E4(distilled student) + E2(TTA)
   - 목적: “최종 제출용” 점수 최대화. (논문에서는 ablation 뒤 최종 성능으로 제시)

### 평가/결정 규칙
- E1/E2에서 이득이 명확(예: mean MAE가 baseline 대비 일관되게 감소)하면, SSA/consistency는 논문 기여 축으로 확정.
- E3에서 앙상블만으로 Top-1 근처(0.017x)에 접근하면, E4로 “단일 모델화”를 논문 핵심으로.
- 멀티태스크(E5)가 효과 없으면 과감히 제외(복잡도 대비 ROI 낮음).

---

## 부록: 왜 mp_e_form이 논문/리더보드 양쪽에 더 ‘재밌는가’
- mp_e_form Top-1이 0.0170으로 매우 낮아서(상위권 간 격차가 아주 작음) **분산 감소/일반화 안정화 기법**이 성능을 좌우할 여지가 큼.
  - mp_e_form 리더보드: https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_e_form/
- coGN/coNGN이 “전처리와 아키텍처의 강한 결합”을 강조하고 있음 → 우리가 “모델 자체가 아니라 **학습/추론 프로토콜**”로 따라잡거나 넘기는 스토리가 가능.
  - coGN full data: https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_coGN/
  - coNGN full data: https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_coNGN/

