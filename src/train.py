"""Training loops for the NEU steel-defect classifier.

Two-stage transfer learning:
  1. Linear probe: backbone frozen, only the classifier head trains.
  2. Fine-tune: the architecture's last block(s) unfrozen, low LR, fresh optimizer
     built over the now-trainable parameters (a fresh optimizer per stage avoids
     silently unfrozen params that never receive gradient updates).
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import Config
from src.dataset import build_loaders, load_datasets
from src.models import DefectClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


@dataclass
class EpochMetrics:
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    train_seconds: float
    val_seconds: float


def fit(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, checkpoint_path, log_prefix=""):
    history: List[EpochMetrics] = []
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        t1 = time.perf_counter()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        t2 = time.perf_counter()
        scheduler.step(val_loss)

        train_seconds, val_seconds = t1 - t0, t2 - t1
        history.append(EpochMetrics(train_loss, train_acc, val_loss, val_acc, train_seconds, val_seconds))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

        print(f"{log_prefix}[{epoch}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
              f"(train {train_seconds:.1f}s, val {val_seconds:.1f}s)")

    return history


def run_two_stage_training(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_id = cfg.run_id()

    ds = load_datasets(cfg)
    train_loader, val_loader, test_loader = build_loaders(ds, cfg.batch_size)

    criterion = nn.CrossEntropyLoss()

    # --- Stage 1: linear probe ---
    model = DefectClassifier(cfg.architecture, num_classes=len(ds.classes)).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    stage1_ckpt = cfg.checkpoint_dir / f"{run_id}_stage1_best.pth"
    t0 = time.perf_counter()
    history_stage1 = fit(
        model, train_loader, val_loader, optimizer, scheduler, criterion, device,
        cfg.epochs_stage1, stage1_ckpt, log_prefix="[stage1] ",
    )
    stage1_seconds = time.perf_counter() - t0

    # --- Stage 2: fine-tune ---
    model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
    model.unfreeze_finetune_block()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    stage2_ckpt = cfg.checkpoint_dir / f"{run_id}_stage2_finetuned.pth"
    t0 = time.perf_counter()
    history_stage2 = fit(
        model, train_loader, val_loader, optimizer, scheduler, criterion, device,
        cfg.epochs_stage2, stage2_ckpt, log_prefix="[stage2] ",
    )
    stage2_seconds = time.perf_counter() - t0

    timing = {
        "stage1_seconds": stage1_seconds,
        "stage2_seconds": stage2_seconds,
        "grand_total_seconds": stage1_seconds + stage2_seconds,
    }
    print(f"[timing] stage1 {stage1_seconds:.1f}s | stage2 {stage2_seconds:.1f}s | "
          f"grand total {timing['grand_total_seconds']:.1f}s")

    history_path = cfg.checkpoint_dir / f"{run_id}_history.json"
    history_path.write_text(json.dumps({
        "architecture": cfg.architecture,
        "run_label": cfg.run_label,
        "seed": cfg.seed,
        "classes": ds.classes,
        "stage1": [asdict(m) for m in history_stage1],
        "stage2": [asdict(m) for m in history_stage2],
        "timing": timing,
    }, indent=2))

    return {
        "classes": ds.classes,
        "test_loader": test_loader,
        "history_stage1": history_stage1,
        "history_stage2": history_stage2,
        "stage1_checkpoint": stage1_ckpt,
        "stage2_checkpoint": stage2_ckpt,
        "history_path": history_path,
        "timing": timing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default=None, help="Overrides Config.architecture")
    parser.add_argument("--run-label", default="", help="Suffix (e.g. '_1') to distinguish repeated runs")
    args = parser.parse_args()

    cfg = Config()
    if args.architecture:
        cfg.architecture = args.architecture
    cfg.run_label = args.run_label

    run_two_stage_training(cfg)


if __name__ == "__main__":
    main()
