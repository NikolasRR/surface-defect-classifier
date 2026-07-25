"""Central configuration for the NEU steel-defect classifier pipeline.

Change ARCHITECTURE to any key in src.models.ARCHITECTURES to swap the backbone
("engine") without touching any other module.
"""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "files" / "data"
TRAIN_DIR = DATA_ROOT / "train" / "images"
VALIDATION_DIR = DATA_ROOT / "validation" / "images"
CHECKPOINT_DIR = REPO_ROOT / "src" / "checkpoints"


@dataclass
class Config:
    architecture: str = "resnet18"
    seed: int = 42

    image_size: int = 224
    batch_size: int = 32

    # Stage 1: linear probe (backbone frozen, only the classifier head trains)
    epochs_stage1: int = 20
    lr_stage1: float = 1e-4

    # Stage 2: fine-tune (last block unfrozen, low LR)
    epochs_stage2: int = 10
    lr_stage2: float = 1e-5

    # Validation folder is split 50/50 into val/test (stratified by class).
    val_fraction_of_validation: float = 0.5

    train_dir: Path = TRAIN_DIR
    validation_dir: Path = VALIDATION_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
