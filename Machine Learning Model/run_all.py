#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — Run all 5 training scripts sequentially and merge metrics.
"""

import json, time, subprocess, sys
from pathlib import Path

BASE_OUT = Path(__file__).resolve().parent.parent / "optimized"
SCRIPTS = ["train_esn.py", "train_svm.py", "train_rf.py", "train_lstm.py", "train_gru.py"]

if __name__ == "__main__":
    t0 = time.time()
    script_dir = Path(__file__).resolve().parent

    for script in SCRIPTS:
        print(f"\n{'#'*70}")
        print(f"  Running {script} ...")
        print(f"{'#'*70}\n")
        result = subprocess.run(
            [sys.executable, str(script_dir / script)],
            cwd=str(script_dir),
        )
        if result.returncode != 0:
            print(f"  ❌ {script} failed with code {result.returncode}")
            sys.exit(1)

    # Merge all round_metrics.json into one all_round_metrics.json
    print(f"\n{'='*70}")
    print("  Merging metrics ...")
    all_metrics = {}
    for subdir in sorted(BASE_OUT.iterdir()):
        rm_file = subdir / "round_metrics.json"
        if rm_file.exists():
            data = json.loads(rm_file.read_text(encoding="utf-8"))
            all_metrics.update(data)

    out_file = BASE_OUT / "all_round_metrics.json"
    out_file.write_text(json.dumps(all_metrics, indent=2, default=float), encoding="utf-8")
    print(f"  Saved: {out_file}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ALL DONE  (total: {elapsed/60:.1f} min)")
    print(f"{'='*70}")
    for name, data in sorted(all_metrics.items(), key=lambda x: -x[1]["overall_accuracy"]):
        import numpy as np
        ra = data["round_accuracies"]
        print(f"    {name:20s} {data['overall_accuracy']:.2%}  "
              f"(per-round: {np.mean(ra):.2%} ± {np.std(ra):.2%})")
    print(f"{'='*70}")
