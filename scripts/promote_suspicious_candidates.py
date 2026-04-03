from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Optional


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_original_filename(candidate_name: str) -> str:
    marker = "__true-"
    if marker in candidate_name:
        prefix, rest = candidate_name.split(marker, 1)
        original_stem = prefix
        ext = Path(candidate_name).suffix
        return f"{original_stem}{ext}"
    return candidate_name


def build_metadata_index(metadata_csv: Path) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    with metadata_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exported_path = row.get("exported_remapped_path", "")
            if not exported_path:
                continue
            key = Path(exported_path).name
            index[key] = row
    return index


def find_original_source(
    candidate_file: Path,
    metadata_index: Dict[str, Dict],
    split_root: Path,
) -> Optional[Path]:
    original_name = parse_original_filename(candidate_file.name)

    row = metadata_index.get(original_name)
    if row:
        remapped_label = row.get("remapped_label", "").strip()
        exported_remapped_path = row.get("exported_remapped_path", "").strip()
        if remapped_label and exported_remapped_path:
            src = Path(exported_remapped_path)
            if src.exists():
                return src

    for cls in ["safe", "unsafe"]:
        candidate = split_root / cls / original_name
        if candidate.exists():
            return candidate

    return None


def promote_candidates_for_split(
    split: str,
    data_root: Path,
    metadata_root: Path,
    dry_run: bool,
    copy_instead_of_move: bool,
    keep_candidate_files: bool,
) -> None:
    split_root = data_root / split / "remapped"
    candidates_dir = split_root / "suspicious_candidates"
    suspicious_dir = split_root / "suspicious"
    metadata_csv = metadata_root / f"openimages_{split}_metadata.csv"

    if not candidates_dir.exists():
        print(f"[WARN] No suspicious_candidates directory for split='{split}': {candidates_dir}")
        return

    ensure_dir(suspicious_dir)

    metadata_index = {}
    if metadata_csv.exists():
        metadata_index = build_metadata_index(metadata_csv)
    else:
        print(f"[WARN] Metadata CSV not found for split='{split}': {metadata_csv}")

    candidate_files = [p for p in candidates_dir.iterdir() if p.is_file()]

    promoted = 0
    missing_source = 0
    skipped_existing = 0

    print(f"\n=== Promoting suspicious candidates for split='{split}' ===")
    print(f"Candidates dir: {candidates_dir}")
    print(f"Suspicious dir: {suspicious_dir}")
    print(f"Num candidate files: {len(candidate_files)}")

    for candidate_file in candidate_files:
        source_path = find_original_source(candidate_file, metadata_index, split_root)

        if source_path is None:
            print(f"[WARN] Could not find original source for candidate: {candidate_file.name}")
            missing_source += 1
            continue

        target_path = suspicious_dir / source_path.name

        if target_path.exists():
            print(f"[INFO] Target already exists, skipping duplicate: {target_path.name}")
            skipped_existing += 1
            if not keep_candidate_files and not dry_run:
                candidate_file.unlink(missing_ok=True)
            continue

        print(f"[PROMOTE] {source_path} -> {target_path}")

        if not dry_run:
            if copy_instead_of_move:
                shutil.copy2(source_path, target_path)
                source_path.unlink(missing_ok=True)
            else:
                shutil.move(str(source_path), str(target_path))

            if not keep_candidate_files:
                candidate_file.unlink(missing_ok=True)

        promoted += 1

    print("\n[SUMMARY]")
    print(f"Promoted: {promoted}")
    print(f"Missing source: {missing_source}")
    print(f"Skipped existing: {skipped_existing}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Promote suspicious candidate images into the final suspicious class and remove them from safe/unsafe."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/openimages_v7"),
                        help="Root containing train/validation/test/remapped")
    parser.add_argument("--metadata-root", type=Path, default=Path("data/metadata"),
                        help="Root containing openimages_<split>_metadata.csv",)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        help="Splits to process")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without moving files")
    parser.add_argument("--copy-instead-of-move", action="store_true",
                        help="Copy into suspicious then delete original, instead of direct move")
    parser.add_argument("--keep-candidate-files", action="store_true",
                        help="Keep files in suspicious_candidates after promotion")
    return parser.parse_args()


def main():
    args = parse_args()

    for split in args.splits:
        promote_candidates_for_split(
            split=split,
            data_root=args.data_root,
            metadata_root=args.metadata_root,
            dry_run=args.dry_run,
            copy_instead_of_move=args.copy_instead_of_move,
            keep_candidate_files=args.keep_candidate_files,
        )


if __name__ == "__main__":
    main()