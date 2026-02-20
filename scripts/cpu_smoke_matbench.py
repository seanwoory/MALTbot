# scripts/cpu_smoke_matbench.py
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from matbench.bench import MatbenchBenchmark
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="matbench_steels", help="matbench task name")
    ap.add_argument("--out", required=True, help="path to write results.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mb = MatbenchBenchmark(autoload=False)
    task = next(t for t in mb.tasks if t.dataset_name == args.task)
    task.load()

    # 매우 가벼운 baseline: composition(string) 길이 1개 피처 + Ridge
    # 목적: "E2E로 돌아가는지" 검증 (성능 목적 아님)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0, random_state=args.seed)),
    ])

    # matbench는 record()로 스코어 계산하는 패턴이 안정적
    for fold in task.folds:
        train_df = task.get_train_and_val_data(fold, as_type="df")
        test_df = task.get_test_data(fold, as_type="df")

        # steels는 보통 composition 컬럼이 있음(없으면 즉시 실패 -> smoke test 목적 달성)
        if "composition" not in train_df.columns:
            raise KeyError(f"'composition' not found. columns={list(train_df.columns)}")

        X_train = train_df["composition"].astype(str).str.len().values.reshape(-1, 1)
        y_train = train_df["target"].values if "target" in train_df.columns else train_df.iloc[:, -1].values

        X_test = test_df["composition"].astype(str).str.len().values.reshape(-1, 1)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        task.record(fold, pred)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "cpu_smoke",
        "task": args.task,
        "model": "len(composition)+Ridge",
        "seed": args.seed,
        "scores": task.scores,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
