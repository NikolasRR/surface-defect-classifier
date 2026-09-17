"""Generates a multi-page PDF performance report for a trained NEU defect classifier.

Requires that `python -m src.train` has already been run for the target architecture
(reads its saved checkpoints and training-history JSON from cfg.checkpoint_dir; does not
retrain anything itself).

Usage:
    python -m src.report [--architecture resnet18] [--output src/reports/resnet18_report.pdf]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages

from src.config import Config
from src.dataset import build_loaders, load_datasets
from src.evaluate import (
    accuracy,
    classification_report_dict,
    confusion_matrix_plot,
    measure_inference_time,
    top2_accuracy,
)
from src.models import DefectClassifier
from src.train import set_seed

PAGE_SIZE = (8.5, 11)


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.2f}s"

    seconds = round(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _load_history(cfg: Config) -> dict:
    path = cfg.checkpoint_dir / f"{cfg.run_id()}_history.json"
    if not path.exists():
        raise FileNotFoundError(f"No training history at {path}. Run `python -m src.train` first.")
    return json.loads(path.read_text())


def _load_checkpoints(cfg: Config, num_classes: int, device):
    run_id = cfg.run_id()
    stage1_ckpt = cfg.checkpoint_dir / f"{run_id}_stage1_best.pth"
    stage2_ckpt = cfg.checkpoint_dir / f"{run_id}_stage2_finetuned.pth"
    if not stage1_ckpt.exists() or not stage2_ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoints for '{run_id}'. Run `python -m src.train` first.")

    stage1_model = DefectClassifier(cfg.architecture, num_classes=num_classes).to(device)
    stage1_model.load_state_dict(torch.load(stage1_ckpt, map_location=device))

    stage2_model = DefectClassifier(cfg.architecture, num_classes=num_classes).to(device)
    stage2_model.unfreeze_finetune_block()
    stage2_model.load_state_dict(torch.load(stage2_ckpt, map_location=device))

    return stage1_model, stage2_model


def _title_page(pdf, cfg: Config, ds, history: dict, test_acc_stage2: float):
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.axis("off")

    ax.text(0.5, 0.93, "NEU Steel-Defect Classifier", ha="center", fontsize=22, weight="bold")
    ax.text(0.5, 0.885, "Performance Report", ha="center", fontsize=15, color="#444444")
    ax.text(0.5, 0.85, datetime.now().strftime("Generated %Y-%m-%d %H:%M"),
            ha="center", fontsize=9, color="gray")

    ax.text(0.5, 0.77, f"Architecture: {cfg.run_id()}", ha="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.735, f"Final test accuracy (fine-tuned): {test_acc_stage2:.1%}",
            ha="center", fontsize=13, color="#2E7D32", weight="bold")

    ax.text(0.06, 0.65, "Dataset", fontsize=12, weight="bold")
    per_class = len(ds.train) // len(ds.classes)
    dataset_rows = [
        ["Split", "Img/class", "Total"],
        ["Train", str(per_class), str(len(ds.train))],
        ["Validation", str(len(ds.val) // len(ds.classes)), str(len(ds.val))],
        ["Test", str(len(ds.test) // len(ds.classes)), str(len(ds.test))],
    ]
    t1 = ax.table(cellText=dataset_rows, cellLoc="center", bbox=[0.06, 0.42, 0.42, 0.20])
    t1.auto_set_font_size(False)
    t1.set_fontsize(8.5)
    t1.auto_set_column_width(list(range(3)))
    for j in range(3):
        t1[0, j].set_facecolor("#4C72B0")
        t1[0, j].set_text_props(color="white", weight="bold")

    ax.text(0.54, 0.65, "Hyperparameters", fontsize=12, weight="bold")
    hp_rows = [
        ["Setting", "Value"],
        ["Epochs (S1 / S2)", f"{len(history['stage1'])} / {len(history['stage2'])}"],
        ["LR (S1 / S2)", f"{cfg.lr_stage1:g} / {cfg.lr_stage2:g}"],
        ["Batch size", str(cfg.batch_size)],
        ["Image size", f"{cfg.image_size}x{cfg.image_size}"],
        ["Seed", str(cfg.seed)],
    ]
    t2 = ax.table(cellText=hp_rows, cellLoc="center", bbox=[0.54, 0.42, 0.40, 0.20])
    t2.auto_set_font_size(False)
    t2.set_fontsize(8.5)
    t2.auto_set_column_width([0, 1])
    for j in range(2):
        t2[0, j].set_facecolor("#4C72B0")
        t2[0, j].set_text_props(color="white", weight="bold")

    ax.text(0.08, 0.35, "Classes", fontsize=12, weight="bold")
    ax.text(0.08, 0.30, "\n".join(f"- {c}" for c in ds.classes), fontsize=10, va="top")

    pdf.savefig(fig)
    plt.close(fig)


def _training_curves_page(pdf, history: dict):
    stage1, stage2 = history["stage1"], history["stage2"]
    n1 = len(stage1)
    epochs = list(range(1, n1 + len(stage2) + 1))

    train_loss = [m["train_loss"] for m in stage1] + [m["train_loss"] for m in stage2]
    val_loss = [m["val_loss"] for m in stage1] + [m["val_loss"] for m in stage2]
    train_acc = [m["train_acc"] for m in stage1] + [m["train_acc"] for m in stage2]
    val_acc = [m["val_acc"] for m in stage1] + [m["val_acc"] for m in stage2]

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=PAGE_SIZE)

    for ax, train_series, val_series, ylabel, title in [
        (ax_loss, train_loss, val_loss, "Loss", "Loss per epoch"),
        (ax_acc, train_acc, val_acc, "Accuracy", "Accuracy per epoch"),
    ]:
        ax.plot(epochs, train_series, label="train", color="#4C72B0")
        ax.plot(epochs, val_series, label="val", color="#DD8452")
        if n1 and len(stage2):
            ax.axvline(n1 + 0.5, color="gray", linestyle="--", linewidth=1)
            ax.annotate("fine-tune starts", xy=(n1 + 0.5, ax.get_ylim()[1]),
                        xytext=(3, -3), textcoords="offset points",
                        fontsize=8, color="gray", va="top")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Training curves (stage 1 + stage 2)", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _timing_page(pdf, history: dict, inference: dict):
    timing = history.get("timing", {})
    stage1, stage2 = history["stage1"], history["stage2"]

    stage1_s = timing.get("stage1_seconds", sum(m["train_seconds"] + m["val_seconds"] for m in stage1))
    stage2_s = timing.get("stage2_seconds", sum(m["train_seconds"] + m["val_seconds"] for m in stage2))
    grand_total_s = timing.get("grand_total_seconds", stage1_s + stage2_s)

    avg_train1 = sum(m["train_seconds"] for m in stage1) / len(stage1) if stage1 else 0.0
    avg_val1 = sum(m["val_seconds"] for m in stage1) / len(stage1) if stage1 else 0.0
    avg_train2 = sum(m["train_seconds"] for m in stage2) / len(stage2) if stage2 else 0.0
    avg_val2 = sum(m["val_seconds"] for m in stage2) / len(stage2) if stage2 else 0.0

    fig = plt.figure(figsize=PAGE_SIZE)
    fig.suptitle("Training timing", fontsize=14, weight="bold")

    ax_table = fig.add_axes([0.05, 0.66, 0.9, 0.24])
    ax_table.axis("off")

    rows = [
        ["Section", "Total time", "Avg / epoch (train)", "Avg / epoch (val)"],
        [f"Stage 1 - linear probe ({len(stage1)} epochs)", _format_duration(stage1_s),
         _format_duration(avg_train1), _format_duration(avg_val1)],
        [f"Stage 2 - fine-tune ({len(stage2)} epochs)", _format_duration(stage2_s),
         _format_duration(avg_train2), _format_duration(avg_val2)],
        ["Grand total (stage 1 + stage 2)", _format_duration(grand_total_s), "-", "-"],
    ]
    table = ax_table.table(cellText=rows, cellLoc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(4)))
    for j in range(4):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", weight="bold")
    for j in range(4):
        table[3, j].set_text_props(weight="bold")

    fig.text(0.5, 0.60, "Inference latency (fine-tuned model, one test image at a time)",
              ha="center", fontsize=11, weight="bold")
    fig.text(0.5, 0.555, f"{inference['mean_seconds_per_image'] * 1000:.1f} ms / image",
              ha="center", fontsize=20, color="#2E7D32", weight="bold")
    fig.text(0.5, 0.515,
              f"média sobre {inference['num_images']} imagens de teste "
              f"(min {inference['min_seconds_per_image'] * 1000:.1f} ms, "
              f"max {inference['max_seconds_per_image'] * 1000:.1f} ms, "
              f"total {_format_duration(inference['total_seconds'])})",
              ha="center", fontsize=9, color="#666666")

    n1 = len(stage1)
    epochs = list(range(1, n1 + len(stage2) + 1))
    train_seconds = [m["train_seconds"] for m in stage1] + [m["train_seconds"] for m in stage2]
    val_seconds = [m["val_seconds"] for m in stage1] + [m["val_seconds"] for m in stage2]

    ax_chart = fig.add_axes([0.10, 0.07, 0.80, 0.36])
    ax_chart.bar(epochs, train_seconds, label="train", color="#4C72B0")
    ax_chart.bar(epochs, val_seconds, bottom=train_seconds, label="val", color="#DD8452")
    if n1 and len(stage2):
        ax_chart.axvline(n1 + 0.5, color="gray", linestyle="--", linewidth=1)
        ax_chart.annotate("fine-tune starts", xy=(n1 + 0.5, ax_chart.get_ylim()[1]),
                           xytext=(3, -3), textcoords="offset points",
                           fontsize=8, color="gray", va="top")
    ax_chart.set_xlabel("Epoch")
    ax_chart.set_ylabel("Seconds")
    ax_chart.set_title("Wall-clock duration per epoch (train + val, stacked)")
    ax_chart.legend()

    pdf.savefig(fig)
    plt.close(fig)


def _comparison_page(pdf, acc1: float, acc2: float, top2: float, split_name: str = "test"):
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.suptitle("Stage 1 vs Stage 2 comparison", fontsize=14, weight="bold")

    ax = fig.add_axes([0.15, 0.42, 0.7, 0.38])
    bars = ax.bar(["Stage 1\n(linear probe)", "Stage 2\n(fine-tuned)"], [acc1, acc2],
                   color=["#4C72B0", "#DD8452"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy on {split_name} set")
    for bar, value in zip(bars, [acc1, acc2]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.3f}",
                ha="center", fontsize=11, weight="bold")

    fig.text(0.5, 0.28, "Top-2 accuracy (fine-tuned model, test set)",
              ha="center", fontsize=12, weight="bold")
    fig.text(0.5, 0.20, f"{top2:.1%}", ha="center", fontsize=26, color="#2E7D32", weight="bold")
    fig.text(0.5, 0.14,
              "Fraction of test images where the correct class was among the model's\n"
              "top 2 most likely predictions (not just its single best guess).",
              ha="center", fontsize=9, color="#666666")

    pdf.savefig(fig)
    plt.close(fig)


def _classification_report_page(pdf, report_dict: dict, classes: list):
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.axis("off")
    ax.set_title("Classification report (test set, fine-tuned model)",
                 fontsize=13, weight="bold", pad=20)

    rows = [["Class", "Precision", "Recall", "F1-score", "Support"]]
    for c in classes:
        r = report_dict[c]
        rows.append([c, f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                     f"{r['f1-score']:.3f}", str(int(r["support"]))])
    rows.append(["", "", "", "", ""])
    rows.append(["Accuracy", "", "", f"{report_dict['accuracy']:.3f}",
                 str(int(report_dict["macro avg"]["support"]))])
    for avg_key in ["macro avg", "weighted avg"]:
        r = report_dict[avg_key]
        rows.append([avg_key, f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                     f"{r['f1-score']:.3f}", str(int(r["support"]))])

    table = ax.table(cellText=rows, cellLoc="center", bbox=[0.05, 0.45, 0.9, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for j in range(5):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", weight="bold")

    pdf.savefig(fig)
    plt.close(fig)


def generate_report(cfg: Config, output_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    ds = load_datasets(cfg)
    _, _, test_loader = build_loaders(ds, cfg.batch_size)

    history = _load_history(cfg)
    stage1_model, stage2_model = _load_checkpoints(cfg, len(ds.classes), device)

    report_dict = classification_report_dict(stage2_model, test_loader, ds.classes, device)
    acc1 = accuracy(stage1_model, test_loader, device)
    acc2 = accuracy(stage2_model, test_loader, device)
    top2 = top2_accuracy(stage2_model, test_loader, device)
    inference = measure_inference_time(stage2_model, ds.test, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _title_page(pdf, cfg, ds, history, report_dict["accuracy"])
        _training_curves_page(pdf, history)
        _timing_page(pdf, history, inference)

        cm_fig = confusion_matrix_plot(stage2_model, test_loader, ds.classes, device)
        pdf.savefig(cm_fig)
        plt.close(cm_fig)

        _classification_report_page(pdf, report_dict, ds.classes)
        _comparison_page(pdf, acc1, acc2, top2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default=None, help="Overrides Config.architecture")
    parser.add_argument("--run-label", default="", help="Suffix (e.g. '_1') matching the training run")
    parser.add_argument("--output", default=None, help="Output PDF path")
    args = parser.parse_args()

    cfg = Config()
    if args.architecture:
        cfg.architecture = args.architecture
    cfg.run_label = args.run_label

    output_path = Path(args.output) if args.output else (
        cfg.checkpoint_dir.parent / "reports" / f"{cfg.run_id()}_report.pdf"
    )

    path = generate_report(cfg, output_path)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
