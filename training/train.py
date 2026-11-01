#!/usr/bin/env python3
"""Train YOLO11 on the prepared microplastic dataset and copy the best weights to models/best.pt.

Example (RTX 3050 6 GB):
    python training/train.py --model yolo11s.pt --imgsz 640 --epochs 100 --batch 16
If you hit "CUDA out of memory": lower --batch (8, then 4) or use --model yolo11n.pt.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11n.pt / yolo11s.pt / yolo11m.pt, or a .yaml to train from scratch")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' for the first GPU, 'cpu' for CPU")
    ap.add_argument("--workers", type=int, default=4, help="data-loader workers (use 2 if Windows complains)")
    ap.add_argument("--patience", type=int, default=30, help="early-stop after N epochs without improvement")
    ap.add_argument("--fraction", type=float, default=1.0, help="use only this fraction of the training set (smoke tests)")
    ap.add_argument("--name", default="yolo11_microplastic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "best.pt",
                    help="where to copy the best weights (use a new name to keep an older model)")
    # Augmentation / loss knobs. The defaults reproduce the first run; the README retrain uses stronger values.
    ap.add_argument("--hsv-h", type=float, default=0.0, help="hue jitter (Nile Red colour can carry polymer information, keep small)")
    ap.add_argument("--hsv-s", type=float, default=0.3, help="saturation jitter")
    ap.add_argument("--hsv-v", type=float, default=0.3, help="brightness jitter (helps with session-to-session lighting changes)")
    ap.add_argument("--cls", type=float, default=0.5, help="classification loss weight (raise to focus on telling polymers apart)")
    args = ap.parse_args()

    from ultralytics import YOLO

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run training/prepare_dataset.py first.")

    model = YOLO(args.model)
    model.train(
        data=str(args.data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=args.device, workers=args.workers, patience=args.patience, fraction=args.fraction,
        project=str(ROOT / "results"), name=args.name, exist_ok=True, seed=args.seed, plots=True,
        # Fluorescence microscopy: particles have no up/down, and hue can carry polymer information,
        # so rotate/flip freely but do not shift hue.
        flipud=0.5, fliplr=0.5, hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v, degrees=0.0, mosaic=1.0,
        cls=args.cls,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit("Training finished but best.pt was not found.")
    dest = args.out if args.out.is_absolute() else ROOT / args.out
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(best, dest)
    print(f"\nBest weights copied to {dest}\nNext: python training/evaluate.py --weights {dest}")


if __name__ == "__main__":  # required on Windows (multiprocessing)
    main()