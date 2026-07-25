"""Dataset loading for the NEU steel-defect classifier.

files/data/train/images/<class>/       -> training set, used as-is.
files/data/validation/images/<class>/  -> stratified-split 50/50 into val/test
(in-memory via Subset, no files copied to disk), since the dataset ships without
a dedicated test split.
"""

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


@dataclass
class Datasets:
    train: Subset
    val: Subset
    test: Subset
    classes: list


def load_datasets(cfg) -> Datasets:
    train_tf, eval_tf = build_transforms(cfg.image_size)

    train_ds = datasets.ImageFolder(root=str(cfg.train_dir), transform=train_tf)

    # Same folder loaded twice with different transforms: eval-only version is
    # split into val/test, so neither one ever gets train-time augmentation.
    validation_ds = datasets.ImageFolder(root=str(cfg.validation_dir), transform=eval_tf)

    indices = list(range(len(validation_ds)))
    labels = [validation_ds.samples[i][1] for i in indices]
    val_indices, test_indices = train_test_split(
        indices,
        test_size=1 - cfg.val_fraction_of_validation,
        stratify=labels,
        random_state=cfg.seed,
    )

    val_ds = Subset(validation_ds, val_indices)
    test_ds = Subset(validation_ds, test_indices)

    return Datasets(train=train_ds, val=val_ds, test=test_ds, classes=train_ds.classes)


def build_loaders(ds: Datasets, batch_size: int):
    train_loader = DataLoader(ds.train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds.val, batch_size=batch_size)
    test_loader = DataLoader(ds.test, batch_size=batch_size)
    return train_loader, val_loader, test_loader
