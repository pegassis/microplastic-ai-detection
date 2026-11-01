#!/usr/bin/env python3
"""Prepare the Nile Red microplastic dataset (sombsuk/dataset_microplastic) for YOLO training.

Input : a local clone of the dataset repo (image zips + Alldataset_annotation.zip)
Output: dataset/{train,val,test}/{images,labels}, dataset/data.yaml, dataset/manifest.csv

What it does
  * extracts the zips into a temporary work folder
  * matches every image to its YOLO label file (by polymer class + file name)
  * validates / cleans labels (malformed lines, wrong class id, out-of-range boxes)
  * removes exact duplicate images
  * splits into train / val / test

Splitting
  Every image in this dataset shows ONE polymer, and images come from separate
  capture sessions (one zip = one session). Frames from the same session look
  very alike, so a plain random split leaks near-duplicates into val/test and
  inflates the scores. The default `--split-mode batch` therefore keeps whole
  sessions (zips) together. Classes with fewer than 3 sessions fall back to a
  contiguous split inside a session; this is reported so you know which numbers
  are less trustworthy.

LICENSE: the dataset is CC BY-NC-ND 4.0. Use it for non-commercial work, cite the
paper, and do not redistribute the dataset or derived copies of it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["ABS", "Nylon", "PE", "PET", "PS", "PVC"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANN_ZIP = "Alldataset_annotation.zip"
CLASS_RE = re.compile(r"^(ABS|Nylon|PE|PET|PS|PVC)\b")
SPLITS = ("train", "val", "test")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def class_from_name(name: str):
    m = CLASS_RE.match(name)
    return m.group(1) if m else None


def extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)  # zipfile sanitises absolute paths and '..'


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_label(text: str, expected_id: int):
    """Return (clean YOLO lines, Counter of problems)."""
    out, bad = [], Counter()
    for raw in text.splitlines():
        p = raw.split()
        if not p:
            continue
        if len(p) != 5:
            bad["malformed"] += 1
            continue
        try:
            c = int(float(p[0]))
            x, y, w, h = map(float, p[1:])
        except ValueError:
            bad["malformed"] += 1
            continue
        if c != expected_id:
            bad["class_mismatch"] += 1
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            bad["out_of_range"] += 1
            continue
        x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
        x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
        if (x2 - x1) <= 0.002 or (y2 - y1) <= 0.002:
            bad["tiny_box"] += 1
            continue
        out.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return out, bad


def _cut(items, ratios):
    n = len(items)
    n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
    n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
    return items[: n - n_val - n_test], items[n - n_val - n_test: n - n_test], items[n - n_test:]


def _pick_batches(pool: dict, target: float, reserve: int):
    """Choose whole batches whose total size is close to `target`, keeping >= `reserve` batches left."""
    chosen, acc = [], 0
    while len(pool) > reserve and acc < target * 0.75:
        name = min(pool, key=lambda n: abs(acc + len(pool[n]) - target))
        chosen.append(name)
        acc += len(pool.pop(name))
    return chosen


def assign_splits(items, mode, ratios, rng):
    """items: dicts with 'batch' and 'stem' (all one class). Sets item['split'] and item['note']."""
    if mode == "random" or len({i["batch"] for i in items}) == 1:
        items = items[:]
        rng.shuffle(items)
        parts = _cut(items, ratios)
        for split, part in zip(SPLITS, parts):
            for it in part:
                it["split"], it["note"] = split, "random split"
        return

    by_batch = defaultdict(list)
    for it in items:
        by_batch[it["batch"]].append(it)
    total = len(items)

    if len(by_batch) == 2:  # biggest session -> train, smaller one cut contiguously into val/test
        (big, _), (small, _) = sorted(by_batch.items(), key=lambda kv: -len(kv[1]))
        for it in by_batch[big]:
            it["split"], it["note"] = "train", "whole session"
        rest = sorted(by_batch[small], key=lambda i: natural_key(i["stem"]))
        half = len(rest) // 2
        for k, it in enumerate(rest):
            it["split"] = "val" if k < half else "test"
            it["note"] = "contiguous half of one session (less independent)"
        return

    pool = dict(by_batch)
    val_b = _pick_batches(pool, total * ratios[1], reserve=2)
    test_b = _pick_batches(pool, total * ratios[2], reserve=1)
    for n in val_b:
        for it in by_batch[n]:
            it["split"], it["note"] = "val", "whole session"
    for n in test_b:
        for it in by_batch[n]:
            it["split"], it["note"] = "test", "whole session"
    for n in pool:
        for it in by_batch[n]:
            it["split"], it["note"] = "train", "whole session"


def safe_name(cls: str, stem: str, used: set) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    if not base.lower().startswith(cls.lower()):
        base = f"{cls}_{base}"
    name, k = base, 1
    while name in used:
        k += 1
        name = f"{base}_{k}"
    used.add(name)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path, help="folder containing the dataset zips")
    ap.add_argument("--out", type=Path, default=Path("dataset"), help="output dataset folder")
    ap.add_argument("--work", type=Path, default=None, help="scratch folder for extraction (~1 GB free needed)")
    ap.add_argument("--split-mode", choices=["batch", "random"], default="batch")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-class", type=int, default=0, help="debug: limit images per class (0 = all)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing dataset folder")
    args = ap.parse_args()

    repo, out = args.repo.resolve(), args.out.resolve()
    ann_zip = repo / ANN_ZIP
    if not ann_zip.exists():
        sys.exit(f"Cannot find {ANN_ZIP} in {repo}")
    if abs(sum(args.ratios) - 1) > 1e-6:
        sys.exit("--ratios must sum to 1")
    if any((out / s).exists() for s in SPLITS):
        if not args.force:
            sys.exit(f"{out} already contains a dataset. Use --force to rebuild it.")
        for s in SPLITS:
            shutil.rmtree(out / s, ignore_errors=True)

    rng = random.Random(args.seed)
    own_tmp = args.work is None
    work = Path(tempfile.mkdtemp(prefix="mp_prep_")) if own_tmp else args.work
    work.mkdir(parents=True, exist_ok=True)

    try:
        # 1) extract
        print("Extracting archives (this takes a minute)...")
        images = []  # (cls, batch, path)
        for zp in sorted(repo.glob("*.zip"), key=lambda p: natural_key(p.name)):
            if zp.name == ANN_ZIP:
                continue
            cls = class_from_name(zp.stem)
            if cls is None:
                print(f"  skip {zp.name}: cannot tell the polymer from the name")
                continue
            dest = work / "img" / zp.stem
            extract(zp, dest)
            for p in sorted(dest.rglob("*"), key=lambda p: natural_key(p.name)):
                if p.suffix.lower() in IMG_EXT and "__MACOSX" not in p.parts:
                    images.append((cls, zp.stem, p))
        ann_dir = work / "ann"
        extract(ann_zip, ann_dir)
        labels = {}
        for p in ann_dir.rglob("*.txt"):
            if p.name.lower() == "classes.txt" or "__MACOSX" in p.parts:
                continue
            cls = next((part for part in p.relative_to(ann_dir).parts[:-1] if part in CLASSES), None) \
                or class_from_name(p.stem)
            if cls:
                labels[(cls, p.stem)] = p
        print(f"  {len(images)} images, {len(labels)} label files found")

        # 2) match + validate
        stats, problems = Counter(), Counter()
        seen_hash, used_names = set(), set()
        per_class = defaultdict(list)
        for cls, batch, img in images:
            if args.max_per_class and len(per_class[cls]) >= args.max_per_class:
                continue
            lbl = labels.get((cls, img.stem))
            if lbl is None:
                stats["image without label (skipped)"] += 1
                continue
            lines, bad = clean_label(lbl.read_text(errors="ignore"), CLASSES.index(cls))
            problems.update(bad)
            if not lines:
                stats["no valid boxes after cleaning (skipped)"] += 1
                continue
            h = md5(img)
            if h in seen_hash:
                stats["exact duplicate image (skipped)"] += 1
                continue
            seen_hash.add(h)
            per_class[cls].append(dict(cls=cls, batch=batch, stem=img.stem, img=img, lines=lines,
                                       name=safe_name(cls, img.stem, used_names), split=None, note=""))
        unused_labels = len(labels) - sum(len(v) for v in per_class.values())
        print("\nData checks")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        for k, v in problems.items():
            print(f"  label lines dropped ({k}): {v}")
        print(f"  label files not matched to a kept image: {unused_labels}")

        # 3) split
        for cls in CLASSES:
            if per_class[cls]:
                assign_splits(per_class[cls], args.split_mode, args.ratios, rng)
            else:
                print(f"  WARNING: no usable images for class {cls}")

        # 4) write
        rows = []
        for split in SPLITS:
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for cls in CLASSES:
            for it in per_class[cls]:
                ext = it["img"].suffix.lower()
                shutil.copy2(it["img"], out / it["split"] / "images" / f"{it['name']}{ext}")
                (out / it["split"] / "labels" / f"{it['name']}.txt").write_text("\n".join(it["lines"]) + "\n")
                rows.append([f"{it['name']}{ext}", cls, it["batch"], it["split"], len(it["lines"]), it["note"]])
        with open(out / "manifest.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "class", "source_batch", "split", "boxes", "note"])
            w.writerows(rows)
        yaml_text = (f"path: {out.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                     f"nc: {len(CLASSES)}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
        (out / "data.yaml").write_text(yaml_text)

        # 5) report
        print(f"\nSplit mode: {args.split_mode}")
        print(f"{'class':<7}" + "".join(f"{s + ' img/box':>16}" for s in SPLITS))
        for cls in CLASSES:
            cells = []
            for s in SPLITS:
                its = [i for i in per_class[cls] if i["split"] == s]
                cells.append(f"{len(its)}/{sum(len(i['lines']) for i in its)}")
            print(f"{cls:<7}" + "".join(f"{c:>16}" for c in cells))
            if any(not [i for i in per_class[cls] if i['split'] == s] for s in SPLITS):
                print(f"  WARNING: {cls} is missing from at least one split")
            weak = {i["note"] for i in per_class[cls] if "less independent" in i["note"] or i["note"] == "random split"}
            if weak:
                print(f"  note: {cls} could not be split by whole session ({', '.join(sorted(weak))})")
        print(f"\nWrote {out}\n  data.yaml, manifest.csv\n"
              "Reminder: dataset is CC BY-NC-ND 4.0 - non-commercial, cite the paper, do not redistribute.")
    finally:
        if own_tmp:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
