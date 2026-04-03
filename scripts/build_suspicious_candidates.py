from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_needed(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description="Build suspicious candidate pool from error analysis CSV.")
    parser.add_argument("--predictions-csv", type=Path,
                        default=Path("reports/error_analysis/test_predictions.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/raw/openimages_v7/test/remapped/suspicious_candidates"))
    parser.add_argument("--include-fp", action="store_true", help="Include all false positives")
    parser.add_argument("--include-low-confidence", action="store_true",
                        help="Include all low-confidence samples")
    parser.add_argument("--include-fn-below-conf", type=float, default=0.55,
                        help="Include false negatives only if confidence < threshold")
    parser.add_argument("--max-files", type=int, default=300, help="Maximum number of files to copy")

    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(args.output_dir)

    selected = []
    seen = set()

    with args.predictions_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row["filepath"]
            conf = float(row["confidence"])
            is_fp = int(row["is_false_positive"]) == 1
            is_fn = int(row["is_false_negative"]) == 1
            is_low = int(row["is_low_confidence"]) == 1

            keep = False

            if args.include_fp and is_fp:
                keep = True

            if args.include_low_confidence and is_low:
                keep = True

            if is_fn and conf < args.include_fn_below_conf:
                keep = True

            if keep and filepath not in seen:
                selected.append(row)
                seen.add(filepath)

            if len(selected) >= args.max_files:
                break

    for row in selected:
        src = Path(row["filepath"])
        dst_name = (
            f"{src.stem}"
            f"__true-{row['true_label']}"
            f"__pred-{row['pred_label']}"
            f"__conf-{float(row['confidence']):.3f}"
            f"{src.suffix}"
        )
        dst = args.output_dir / dst_name
        copy_if_needed(src, dst)

    print(f"Copied {len(selected)} files to: {args.output_dir}")


if __name__ == "__main__":
    main()
