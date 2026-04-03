from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)


def get_data_transforms(image_size: int) -> Dict[str, transforms.Compose]:
    """
    Augmentation and preprocessing for training and evaluation.
    """
    return {
        "train": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        "eval": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    }


def build_datasets(data_root: Path, image_size: int) -> Tuple[ImageFolder,
                                                              ImageFolder,
                                                              ImageFolder]:
    tfms = get_data_transforms(image_size)

    train_dir = data_root / "train" / "remapped"
    val_dir = data_root / "validation" / "remapped"
    test_dir = data_root / "test" / "remapped"

    train_dataset = datasets.ImageFolder(train_dir, transform=tfms["train"])
    val_dataset = datasets.ImageFolder(val_dir, transform=tfms["eval"])
    test_dataset = datasets.ImageFolder(test_dir, transform=tfms["eval"])

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(
    train_dataset: ImageFolder,
    val_dataset: ImageFolder,
    test_dataset: ImageFolder,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(train_dataset) -> torch.Tensor:
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    weights = len(targets) / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    model.eval()

    all_targets = []
    all_preds = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="binary",
        zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm
    }


def train_one_epoch(
    model,
    loader,
    optimiser,
    criterion,
    device: str,
):
    """
    Training for one epoch.
    Returns average training loss for this epoch.
    """
    model.train()

    running_loss = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimiser.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return running_loss / total_samples


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    args_dict: Dict,
    best_val_metrics: Dict
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "args": args_dict,
        "best_val_metrics": best_val_metrics
    }, save_path)


def main():
    parser = argparse.ArgumentParser(description="Train a baseline ResNet18 on safe/unsafe data.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/openimages_v7"))
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", type=str, default="resnet18_binary_champion")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset = build_datasets(args.data_root, args.image_size)
    print("Class to idx:", train_dataset.class_to_idx)

    if len(train_dataset.classes) != 2:
        raise ValueError(
            f"Expected binary classification with 2 classes, "
            f"got {len(train_dataset.classes)}: {train_dataset.classes}"
        )

    train_loader, val_loader, test_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = build_model(num_classes=2, pretrained=not args.no_pretrained).to(device)

    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset).to(device)
        print(f"Using class weights: {class_weights.cpu().numpy().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimiser = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_val_metrics = {}

    history = []
    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_metrics = val_metrics
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    test_metrics = evaluate(model, test_loader, device)

    print("\n=== Best Validation Metrics ===")
    print(json.dumps(best_val_metrics, indent=2))

    print("\n=== Test Metrics ===")
    print(json.dumps(test_metrics, indent=2))

    save_dir = args.save_dir / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / "best_model.pt"
    metrics_path = save_dir / "metrics.json"
    history_path = save_dir / "history.json"

    save_checkpoint(
        save_path=checkpoint_path,
        model=model,
        class_to_idx=train_dataset.class_to_idx,
        args_dict=vars(args),
        best_val_metrics=best_val_metrics,
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({
            "best_validation": best_val_metrics,
            "test": test_metrics,
        }, f, indent=2)

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved checkpoint to: {checkpoint_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved history to: {history_path}")


if __name__ == "__main__":
    main()