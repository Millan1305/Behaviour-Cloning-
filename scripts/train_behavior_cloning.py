#!/usr/bin/env python3
"""Train the supervised Behavior Cloning network (CPU is fine; ~10-60 s on the sample data).

  python3 scripts/train_behavior_cloning.py [--epochs 200] [--sources both|user|sample]
"""
import argparse
import json
import time

from _bootstrap import ROOT, need

need("numpy")
need("torch")
import preprocess_dataset  # noqa: E402
from robot_cloning.model import save_checkpoint, train_model  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", choices=["user", "sample", "both"], default="both")
    ap.add_argument("--out", default=str(ROOT / "models" / "bc_model.pt"))
    a = ap.parse_args()
    try:
        arrs, episodes, _ = preprocess_dataset.run(a.sources)
    except FileNotFoundError as e:
        raise SystemExit(f"ERROR: {e}")
    t0 = time.time()
    model, meta, _ = train_model(arrs["X"], arrs["Y"], arrs["A"], arrs["EP"], a.epochs, a.lr, 128, a.val_frac,
                                 a.seed, a.hidden)
    save_checkpoint(a.out, model, meta)
    (ROOT / "models" / "training_report.json").write_text(json.dumps(
        {k: meta[k] for k in ("n_train", "n_val", "val_joint_rmse_rad", "val_action_acc")} | {"epochs": a.epochs,
         "train_seconds": round(time.time() - t0, 1), "sources": a.sources}, indent=2))
    print(f"\nSaved model -> {a.out}\nValidation: joint RMSE {meta['val_joint_rmse_rad']:.3f} rad, "
          f"action accuracy {meta['val_action_acc'] * 100:.1f}%  (train {meta['n_train']} / val {meta['n_val']} frames, split by episode)")
