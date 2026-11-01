#!/usr/bin/env python3
"""Evaluate models/best.pt on the held-out TEST split and write results/test_metrics.json.

Use the test split only once, after you have finished tuning on val, otherwise the
numbers are optimistic.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "best.pt")
    ap.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    m = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, device=args.device,
                  conf=0.001, iou=0.6, plots=True, project=str(ROOT / "results"),
                  name=f"{args.split}_eval", exist_ok=True)

    report = {
        "split": args.split,
        "weights": str(args.weights),
        "overall": {"precision": float(m.box.mp), "recall": float(m.box.mr),
                    "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)},
        "per_class": {},
    }
    for i, cls_idx in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        report["per_class"][m.names[int(cls_idx)]] = {
            "precision": float(p), "recall": float(r), "mAP50": float(ap50), "mAP50_95": float(ap)}

    out = ROOT / "results" / f"{args.split}_metrics.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved {out}. Confusion matrix and curves are in results/{args.split}_eval/")


if __name__ == "__main__":
    main()
