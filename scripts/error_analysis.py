from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_needed(src: Path, dest: Path) -> None:
    ensure_dir(dest.parent)
    if not dest.exists():
        shutil.copyfile(src, dest)


def main():
    parser = argparse.ArgumentParser(description="Run error analysis on saved binary classifier")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/openimages_v7"),
                        help="Root containing train/validation/test/remapped")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"], help="Which split to analyse")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/resnet18_binary_champion/best_model.pt"),
                        help="Path to saved checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/error_analysis"),
                        help="Directory to save analysis outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.60,
                        help="Max softmax confidence below which a sample is considered low-confidence")
    parser.add_argument("--max-copy-per-group", type=int, default=200,
                        help="Maximum images to copy for each group folder")

    args = parser.parse_args()
    device = torch.device("mps" if torch.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    split_dir = args.data_root / args.split / "remapped"
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    transform = get_eval_transform(args.image_size)
    dataset = datasets.ImageFolder(split_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False
    )

    print("Class to idx from dataset:", dataset.class_to_idx)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_to_idx_ckpt = checkpoint["class_to_idx"]

    if dataset.class_to_idx != class_to_idx_ckpt:
        raise ValueError(
            f"Dataset class_to_idx {dataset.class_to_idx} does not match checkpoint {args.checkpoint}"
        )

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    model = build_model(num_classes=len(dataset.classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # dataset.samples gives (filepath, target)
    samples = dataset.samples
    all_rows: List[Dict] = []
    with torch.no_grad():
        sample_offset = 0
        for inputs, targets in loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            preds = preds.cpu().tolist()
            confs = confs.cpu().tolist()
            probs = probs.cpu().tolist()
            targets = targets.tolist()

            for i in range(batch_size):
                filepath, target_from_samples = samples[sample_offset+i]
                assert target_from_samples == targets[i]

                true_idx = targets[i]
                pred_idx = preds[i]

                row = {
                    "filepath": filepath,
                    "filename": Path(filepath).name,
                    "split": args.split,
                    "true_idx": true_idx,
                    "true_label": idx_to_class[true_idx],
                    "pred_idx": pred_idx,
                    "pred_label": idx_to_class[pred_idx],
                    "confidence": confs[i],
                    "prob_safe": probs[i][dataset.class_to_idx["safe"]]
                    if "safe" in dataset.class_to_idx else None,
                    "prob_unsafe": probs[i][dataset.class_to_idx["unsafe"]]
                    if "unsafe" in dataset.class_to_idx else None,
                    "is_correct": int(true_idx == pred_idx),
                    "is_false_positive": int(
                        idx_to_class[true_idx] == "safe"
                        and idx_to_class[pred_idx] == "unsafe"
                    ),
                    "is_false_negative": int(
                        idx_to_class[true_idx] == "unsafe"
                        and idx_to_class[pred_idx] == "safe"
                    ),
                    "is_low_confidence": int(confs[i] < args.low_confidence_threshold)
                }
                all_rows.append(row)

            sample_offset += batch_size

    ensure_dir(args.output_dir)

    predictions_csv = args.output_dir / f"{args.split}_predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
        writer.writeheader()
        writer.writerows(all_rows)

    false_positives = [r for r in all_rows if r["is_false_positive"] == 1]
    false_negatives = [r for r in all_rows if r["is_false_negative"] == 1]
    low_confidence = [r for r in all_rows if r["is_low_confidence"] == 1]

    summary = {
        "split": args.split,
        "num_samples": len(all_rows),
        "num_false_positives": len(false_positives),
        "num_false_negatives": len(false_negatives),
        "num_low_confidence": len(low_confidence),
        "low_confidence_threshold": args.low_confidence_threshold,
    }

    with (args.output_dir / f"{args.split}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    def export_group(rows: List[Dict], group_name: str):
        group_dir = args.output_dir / args.split / group_name
        ensure_dir(group_dir)
        csv_path = args.output_dir / args.split / f"{group_name}.csv"
        ensure_dir(csv_path.parent)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

        for row in rows[:args.max_copy_per_group]:
            src = Path(row["filepath"])
            stem = src.stem
            dest_name = (
                f"{stem}"
                f"__true-{row['true_label']}"
                f"__pred-{row['pred_label']}"
                f"__conf-{row['confidence']:.3f}"
                f"{src.suffix}"
            )
            dest = group_dir / dest_name
            copy_if_needed(src, dest)

    if false_positives:
        export_group(false_positives, "false_positives")
    if false_negatives:
        export_group(false_negatives, "false_negatives")
    if low_confidence:
        export_group(low_confidence, "low_confidence")

    print("\n=== Error Analysis Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved predictions CSV to: {predictions_csv}")
    print(f"Saved outputs under: {args.output_dir / args.split}")


if __name__ == "__main__":
    main()