"""Evaluation utilities: classification report, confusion matrix, top-2 accuracy,
and a live (not hardcoded) stage1-vs-stage2 accuracy comparison.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        y_pred.extend(outputs.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred


def classification_report_for(model, loader, classes, device):
    y_true, y_pred = _predict(model, loader, device)
    return classification_report(y_true, y_pred, target_names=classes)


def classification_report_dict(model, loader, classes, device):
    y_true, y_pred = _predict(model, loader, device)
    return classification_report(y_true, y_pred, target_names=classes, output_dict=True)


def confusion_matrix_plot(model, loader, classes, device):
    y_true, y_pred = _predict(model, loader, device)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


@torch.no_grad()
def top2_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        top2 = torch.topk(outputs, 2, dim=1).indices
        correct += (top2 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def compare_stage1_vs_stage2(stage1_model, stage2_model, loader, device, split_name="test"):
    acc_stage1 = accuracy(stage1_model, loader, device)
    acc_stage2 = accuracy(stage2_model, loader, device)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Stage 1 (linear probe)", "Stage 2 (fine-tuned)"], [acc_stage1, acc_stage2],
                   color=["#4C72B0", "#DD8452"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Stage 1 vs Stage 2 accuracy ({split_name} set)")
    for bar, value in zip(bars, [acc_stage1, acc_stage2]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center")
    fig.tight_layout()
    return fig, acc_stage1, acc_stage2
