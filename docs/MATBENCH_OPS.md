# Matbench Ops (v0)

이 문서는 sungwoo와 선생님(main)이 **Matbench v0.1 Top-5~1등** 목표를 위해 합의한 운영 규칙을 고정한다.

## 0) 기본 원칙
- 목표: Matbench v0.1에서 **Top-5 이내(2순위)**, 최종적으로 **1등(1순위)**.
- 산출물: GitHub 레포(매일 업데이트) + arXiv 논문(피어리뷰 없이 공개).
- 역할: **코드는 coder만 작성**. main(선생님)은 설계/리뷰/오케스트레이션.

---

## 1) Quality Gate (강제)
Colab에서 GPU 시간을 태우는 변경(노트북/설치/푸시/로깅/런너)은 아래를 **반드시** 통과해야 한다.
1) 로컬/CI에서 최소 문법 검증: `python -m compileall` 통과
2) Colab end-to-end 스모크(최소 1-run): `baseline_chgnet` 1회 실행으로 아래가 모두 만족
   - `results/daily/YYYY-MM-DD/<batch>/<exp>/results.json` 생성
   - `results.json`에 numeric `metric_value` 저장
   - `RESULTS.md`에 `METRIC=<float>`로 1줄 append
   - push 성공(브랜치 생성) + PR 링크가 정상적으로 열림

실행 중 버그를 발견하면:
- 즉시 패치하고(코드는 coder),
- 다시 위 스모크를 통과한 뒤에만 추가 실험을 진행한다.

## 2) Daily PR 규격 (최소 세트)
매일 PR(또는 daily 결과 반영)은 아래 3종을 기준으로 한다.

### (A1) 결과 아티팩트(작게)
- 경로: `results/daily/YYYY-MM-DD/<run_name>/results.json`
- results.json 필수 포함 항목:
  - `task.scores` (matbench가 산출한 score dict)
  - 모델명/피처명/학습 레시피 요약
  - 하이퍼파라미터
  - seed
  - commit hash(가능하면)
- 금지:
  - 큰 `preds.csv`, 대형 checkpoint를 기본 git에 커밋하지 않기

### (A2) RESULTS.md 한 줄
- 포맷(권장):
  - `날짜 | 태스크 | 모델 | MAE | 링크(results.json) | 메모(짧게)`
- 예:
  - `2026-02-20 | matbench_mp_e_form | structure_simple+HGB | 0.49346 | results/daily/.../results.json | baseline`

### (A3) 코드 변화가 있었으면 코드도 같이
- 포함 후보:
  - `scripts/` 변경분
  - `configs/` 변경분
  - `README/USAGE` 변경분
- 규칙:
  - **매일 무조건 코드가 바뀌어야 한다는 강박은 금지** (재현성 런/seed 반복일 수 있음)
  - 단, **주 단위로는 코드가 반드시 진화**해야 함 (파이프라인/피처/모델/재현성 개선)

---

## 2) Weekly(또는 기능 추가 시) PR 규격 (권장)
- 단일 엔트리포인트:
  - `scripts/run_matbench.py` (또는 동급의 1-command runner)
- 환경 고정:
  - `requirements.txt`(버전 핀) 또는 `environment.yml`
- 문서화:
  - `docs/` (실험 프로토콜, 결과 해석, 재현 방법)

요약: `RESULTS.md`만으로는 부족하며, **결과 JSON + RESULTS.md + (변경된 코드)**가 논문용 최소 구성이다.

---

## 3) Researcher 운영 규칙(강제)
"만연한 모델만" 검증하는 것을 방지하기 위한 품질 게이트.

### 매일 리포트에 반드시 포함
1) 오늘의 1순위 후보 1개
   - 근거 링크(논문/코드), 코드 공개 여부, 적용 난이도
2) 대안 후보 1개
3) 보류(리스크) 후보 1개

### 조사 범위 (최신/비보편 포함)
- 프리트레인/대형 모델 계열(예: Skala/Fair 계열)
- CHGNet / M3GNet 등 실제 성능 좋은 최신 라인
- Matbench v0.1에서 **실제로 상위권 점수**를 낸 구현(재현 가능성/코드 공개 포함)
- 외부 데이터 전략(MP/OC20/Materials Project 등) — 라이선스/재현성 포함 평가

---

## 4) 책임/권한
- main(선생님): 요구사항/통과 기준 정의, 결과 해석, 다음 실험 설계, 품질 게이트.
- coder: 코드/YAML/스크립트 구현 및 PR 제출(변경 요약, 실행 커맨드, 결과 포함).
- sungwoo: Colab 실행(평일 퇴근 후 1회 가능), PR merge 의사결정.
