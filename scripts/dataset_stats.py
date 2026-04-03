from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

def inspect_image(path: Path) -> Tuple[bool, Tuple[int, int] | None]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return True, img.size # (width, height)
    except UnidentifiedImageError:
        return False, None

def summarise_sizes(sizes: List[Tuple[int, int]]) -> Dict[str, float]:
    if not sizes:
        return {
            "min_width": 0,
            "max_width": 0,
            "avg_width": 0,
            "min_height": 0,
            "max_height": 0,
            "avg_height": 0,
        }

    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "avg_width": sum(widths) / len(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_height": sum(heights) / len(heights),
    }

def collect_stats(data_root: Path, splits: List[str], class_names: List[str]) -> List[Dict]:
    rows: List[Dict] = []

    for split in splits:
        split_root = data_root / split / "remapped"

        for class_name in class_names:
            class_dir = split_root / class_name
            image_paths = list_images(class_dir)

            valid_sizes: List[Tuple[int, int]] = []
            broken_count = 0

            for path in image_paths:
                ok, size = inspect_image(path)
                if ok and size is not None:
                    valid_sizes.append(size)
                else:
                    broken_count += 1

            size_summary = summarise_sizes(valid_sizes)
            rows.append({
                "split": split,
                "class_name": class_name,
                "num_images": len(image_paths),
                "num_valid_images": len(valid_sizes),
                "num_broken_images": broken_count,
                **size_summary
            })

    return rows

def print_stats_table(rows: List[Dict]) -> None:
    print("\n=== Dataset Stats ===")
    for row in rows:
        print(
            f"[{row['split']:<10}] "
            f"{row['class_name']:<10} "
            f"count={row['num_images']:<5} "
            f"valid={row['num_valid_images']:<5} "
            f"broken={row['num_broken_images']:<3} "
            f"avg_size=({row['avg_width']:.1f}, {row['avg_height']:.1f})"
        )
    print("\n=== Split Totals ===")
    totals = defaultdict(int)
    for row in rows:
        totals[row['split']] += row['num_images']

    for split, total in totals.items():
        print(f"{split:<10} total={total}")

def save_csv(rows: List[Dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "split", "class_name", "num_images", "num_valid_images", "num_broken_images",
        "min_width", "max_width", "avg_width", "min_height", "max_height", "avg_height"
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dataset statistics for remapped image folders.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/openimages_v7"),
                        help="Root directory containing train/validation/test subfolders")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        help="Dataset splits to inspect")
    parser.add_argument("--classes", nargs="+", default=["safe", "unsafe"],
                        help="Class names under remapped/")
    parser.add_argument("--output-csv", type=Path, default=Path("reports/dataset_stats.csv"),
                        help="Where to save CSV summary")
    args = parser.parse_args()
    rows = collect_stats(
        data_root=args.data_root,
        splits=args.splits,
        class_names=args.classes,
    )
    print_stats_table(rows)
    save_csv(rows, args.output_csv)
    print(f"\nSaved stats CSV to: {args.output_csv}")