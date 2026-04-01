import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fiftyone as fo
import fiftyone.zoo as foz


# Class mappings
DEFAULT_CLASS_MAP = {
    "safe": [
        "Cat",
        "Dog",
        "Book",
        "Laptop",
        "Bicycle"
    ],
    "unsafe": [
        "Knife",
        "Axe",
        "Hammer",
        "Fire",
        "Smoke"
    ],
}


# util functions
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def sanitise_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)

def flatten_class_map(class_map: Dict[str, List[str]]) -> Dict[str, str]:
    reverse = {}
    for remapped_label, original_classes in class_map.items():
        for cls in original_classes:
            if cls in reverse:
                raise ValueError(f"Class '{cls}' is in multiple remapped labels")
            reverse[cls] = remapped_label
    return reverse


def extract_positive_labels_from_sample(sample: fo.Sample):
    labels = set()

    if "positive_labels" not in sample.field_names:
        return labels

    value = sample["positive_labels"]
    if value is None:
        return labels

    if isinstance(value, fo.Classification):
        if value.label:
            labels.add(value.label)

    elif isinstance(value, fo.Classifications):
        for clf in value.classifications:
            if clf.label:
                labels.add(clf.label)

    return labels

def extract_negative_labels_from_sample(sample: fo.Sample):
    labels = set()

    if "negative_labels" not in sample.field_names:
        return labels

    value = sample["negative_labels"]
    if value is None:
        return labels

    if isinstance(value, fo.Classification):
        if value.label:
            labels.add(value.label)

    elif isinstance(value, fo.Classifications):
        for clf in value.classifications:
            if clf.label:
                labels.add(clf.label)
    return labels

def choose_remapped_label(
    original_labels: Set[str],
    reverse_class_map: Dict[str, str],
) -> Optional[Tuple[str, List[str]]]:
    """
    If an image aimed multiple candidate classes:
    - first collect all aimed original labels
    - if matched to one remapped label, then accept
    - if it conflicts happened after mapping
       (e.g. simultaneously matching safe and unsafe),
       then return None
    """
    matched_original = [label for label in original_labels if label in reverse_class_map]
    if not matched_original:
        return None

    remapped_set = {reverse_class_map[label] for label in matched_original}
    if len(remapped_set) != 1:
        return None

    remapped_label = next(iter(remapped_set))
    return remapped_label, matched_original

def copy_image(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)

# Main logic
def build_dataset(
    split: str,
    output_root: Path,
    max_samples: int,
    shuffle: bool,
    launch_app: bool,
    overwrite: bool,
    dataset_name: Optional[str],
) -> None:
    class_map = DEFAULT_CLASS_MAP
    reverse_class_map = flatten_class_map(class_map)

    # Original classes set used for FiftyOne to download
    candidate_classes = sorted(
        {cls for classes in class_map.values() for cls in classes}
    )

    if dataset_name is None:
        dataset_name = f"open-images-v7-{split}-content-safety-candidates"

    if overwrite:
        try:
            fo.delete_dataset(dataset_name)
        except Exception:
            pass

    print(f"[INFO] Loading Open Images V7 split='{split}' into FiftyOne...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["classifications"],
        classes=candidate_classes,
        max_samples=max_samples,
        shuffle=shuffle,
        dataset_name=dataset_name,
    )
    print(f"[DEBUG] Loaded FiftyOne dataset name: {dataset.name}")

    print(f"[INFO] Loaded dataset: {dataset.name}")
    print(f"[INFO] Number of samples: {len(dataset)}")

    # Local output directories
    original_root = output_root / "data" / "raw" / "openimages_v7" / split / "original"
    remapped_root = output_root / "data" / "raw" / "openimages_v7" / split / "remapped"
    metadata_root = output_root / "data" / "metadata"

    ensure_dir(original_root)
    ensure_dir(remapped_root / "safe")
    ensure_dir(remapped_root / "suspicious")
    ensure_dir(remapped_root / "unsafe")
    ensure_dir(metadata_root)

    metadata_path = metadata_root / f"openimages_{split}_metadata.csv"

    rows = []
    accepted = 0
    skipped_conflict = 0
    skipped_unmatched = 0

    print(f"[INFO] Exporting samples to local folder structure...")

    for sample in dataset.iter_samples(progress=True):
        filepath = Path(sample.filepath)
        positive_labels = extract_positive_labels_from_sample(sample)
        negative_labels = extract_negative_labels_from_sample(sample)

        decision = choose_remapped_label(positive_labels, reverse_class_map)
        sample_id = str(sample.id)
        filename = sanitise_filename(f"{sample_id}_{filepath.name}")

        if decision is None:
            # could be failed to match or matching conflict
            matched_original = [label for label in positive_labels if label in reverse_class_map]
            if matched_original:
                skipped_conflict += 1
                rows.append({
                    "sample_id": sample_id,
                    "filepath": str(filepath),
                    "split": split,
                    "positive_labels": "|".join(sorted(positive_labels)),
                    "negative_labels": "|".join(sorted(negative_labels)),
                    "num_positive_labels": len(positive_labels),
                    "matched_positive_labels": "|".join(sorted(matched_original)),
                    "remapped_label": "",
                    "status": "skipped_conflict",
                    "exported_positive_path": "",
                    "exported_remapped_path": "",
                })
            else:
                skipped_unmatched += 1
                rows.append({
                    "sample_id": sample_id,
                    "filepath": str(filepath),
                    "split": split,
                    "positive_labels": "|".join(sorted(positive_labels)),
                    "negative_labels": "|".join(sorted(negative_labels)),
                    "num_positive_labels": len(positive_labels),
                    "matched_original_labels": "",
                    "remapped_label": "",
                    "status": "skipped_unmatched",
                    "exported_original_path": "",
                    "exported_remapped_path": "",
                })
            continue

        remapped_label, matched_original = decision

        # For preserving the source of original classes
        # copy the image to original/<original_class>/
        # If matched multiple original classes with same remapped label,
        # put in the primary matching class
        primary_original = sorted(matched_original)[0]

        original_dst = original_root / primary_original / filename
        remapped_dst = remapped_root / remapped_label / filename

        copy_image(filepath, original_dst)
        copy_image(filepath, remapped_dst)

        rows.append({
            "sample_id": sample_id,
            "filepath": str(filepath),
            "split": split,
            "positive_labels": "|".join(sorted(positive_labels)),
            "negative_labels": "|".join(sorted(negative_labels)),
            "num_positive_labels": len(positive_labels),
            "matched_original_labels": "|".join(sorted(matched_original)),
            "remapped_label": remapped_label,
            "status": "accepted",
            "exported_original_path": str(original_dst),
            "exported_remapped_path": str(remapped_dst),
        })
        accepted += 1

    print(f"[INFO] Writing metadata to: {metadata_path}")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "filepath",
                "split",
                "positive_labels",
                "negative_labels",
                "num_positive_labels",
                "matched_original_labels",
                "remapped_label",
                "status",
                "exported_original_path",
                "exported_remapped_path",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n[SUMMARY]")
    print(f"Accepted: {accepted}")
    print(f"Skipped (conflict): {skipped_conflict})")
    print(f"Skipped (unmatched): {skipped_unmatched}")
    print(f"FiftyOne dataset name: {dataset.name}")
    print(f"Metadata file: {metadata_path}")
    print(f"Remapped folders: {remapped_root}")

    if launch_app:
        print("[INFO] Launching FiftyOne App...")
        session = fo.launch_app(dataset)
        session.wait()


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Open Images V7 into FiftyOne and export a local content-safety-style folder structure."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Open Images split to load"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=".",
        help="Project root where data/ will be created"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8000,
        help="Maximum number of samples to download"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the samples before selection"
    )
    parser.add_argument(
        "--launch-app",
        action="store_true",
        help="Launch FiftyOne App after loading"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing FiftyOne dataset with same name first"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional FiftyOne dataset name"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        split=args.split,
        output_root=Path(args.output_root).resolve(),
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        launch_app=args.launch_app,
        overwrite=args.overwrite,
        dataset_name=args.dataset_name,
    )
