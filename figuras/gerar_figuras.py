"""Gera todas as figuras usadas no artigo/documentacao, em files/docs/mine/figuras/.

Todos os titulos e legendas ficam centralizados no dicionario TEXTOS abaixo -
para mudar qualquer texto exibido nas figuras, edite ali e rode de novo:
    ../venv/Scripts/python.exe figuras/gerar_figuras.py

As figuras de dados (matrizes de confusao, curvas por epoca, comparacoes) sao
recalculadas a partir dos checkpoints/history.json reais em src/checkpoints/
(reprodutivel, seed fixa). A figura fluxo_cnn.png e um diagrama conceitual
(nao vem de dados) desenhado com formas do matplotlib.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.dataset import load_datasets, build_loaders
from src.models import DefectClassifier
from src.train import set_seed
from src.evaluate import _predict, classification_report_dict

# As figuras continuam sendo gravadas junto do documento, em files/, que nao
# e versionada. Este script, sim, faz parte do repositorio.
OUT = REPO_ROOT / "files" / "docs" / "mine" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = REPO_ROOT / "src" / "checkpoints"


# ---------------------------------------------------------------------------
# Separador decimal: virgula, como manda a norma em portugues (ABNT/INMETRO).
# VirgulaFormatter herda do formatador padrao do matplotlib, entao os rotulos
# dos eixos ficam identicos aos de antes, so com "." trocado por ",".
# ---------------------------------------------------------------------------
class VirgulaFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def virgula_no_eixo(ax, x=False, y=True):
    """Aplica o separador decimal brasileiro aos eixos indicados."""
    if y:
        ax.yaxis.set_major_formatter(VirgulaFormatter())
    if x:
        ax.xaxis.set_major_formatter(VirgulaFormatter())


def num(valor, casas=3):
    """Numero formatado com virgula, para rotulos de barra e anotacoes."""
    return f"{valor:.{casas}f}".replace(".", ",")

ARCHS = {"resnet18": "ResNet-18", "mobilenet_v3_small": "MobileNetV3-Small"}
COLORS = {"resnet18": "#4C72B0", "mobilenet_v3_small": "#DD8452"}

# ---------------------------------------------------------------------------
# Todos os textos exibidos nas figuras - edite aqui para mudar titulos/legendas
# ---------------------------------------------------------------------------
TEXTOS = {
    "matriz_confusao_titulo": "Matriz de Confusão - {nome}",
    "matriz_confusao_xlabel": "Classe Predita",
    "matriz_confusao_ylabel": "Classe Verdadeira",

    "acc_epoca_titulo": "Accuracy por época ({nome})",
    "acc_epoca_ylabel": "Accuracy",
    "acc_epoca_xlabel": "Época",
    "acc_epoca_legenda_val": "Validação",
    "acc_epoca_legenda_train": "Treino",
    "acc_epoca_finetune_label": "início do fine-tuning",

    "loss_epoca_titulo": "Loss por época ({nome})",
    "loss_epoca_ylabel": "Loss",
    "loss_epoca_xlabel": "Época",
    "loss_epoca_legenda_val": "Validação",
    "loss_epoca_legenda_train": "Treino",
    "loss_epoca_finetune_label": "início do fine-tuning",

    "comparacao_metricas_titulo": "Comparação de métricas de classificação (conjunto de teste)",
    "comparacao_metricas_ylabel": "Valor",
    "comparacao_metricas_labels": ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1-score\n(macro)"],

    "comparacao_tempo_titulo": "Comparação de tempo total de treinamento\n(média de 3 execuções)",
    "comparacao_tempo_ylabel": "Tempo total de treinamento (minutos)",

    "comparacao_precisao_titulo": "Precision por classe no conjunto de teste",
    "comparacao_precisao_ylabel": "Precision",

    "comparacao_recall_titulo": "Recall por classe no conjunto de teste",
    "comparacao_recall_ylabel": "Recall",

    "fluxo_cnn_imagem": "Imagem",
    "fluxo_cnn_imagem_sub": "H × L × C",
    "fluxo_cnn_conv": "Convolução",
    "fluxo_cnn_conv_sub": "banco de filtros",
    "fluxo_cnn_norm": "Normalização",
    "fluxo_cnn_norm_sub": "batch norm",
    "fluxo_cnn_ativ": "Ativação",
    "fluxo_cnn_ativ_sub": "ReLU / h-swish",
    "fluxo_cnn_pool": "Pooling",
    "fluxo_cnn_pool_sub": "reduz H e L",
    "fluxo_cnn_fc": "Camada totalmente conectada",
    "fluxo_cnn_fc_sub": "vetor → 1 pontuação por classe",
    "fluxo_cnn_softmax": "Softmax → classe predita",
    "fluxo_cnn_softmax_sub": "probabilidades",
    "fluxo_cnn_backbone_nota": "backbone convolucional  ·  o conjunto se repete dezenas de vezes,\n"
                                "com mais canais e menor resolução a cada estágio",
}


def load_model_and_data(arch, device):
    cfg = Config()
    cfg.architecture = arch
    set_seed(cfg.seed)
    ds = load_datasets(cfg)
    _, _, test_loader = build_loaders(ds, cfg.batch_size)

    model = DefectClassifier(arch, num_classes=len(ds.classes)).to(device)
    model.unfreeze_finetune_block()
    model.load_state_dict(torch.load(CKPT / f"{arch}_stage2_finetuned.pth", map_location=device))
    return model, ds, test_loader


def fig_confusion_matrix(arch, nome, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel(TEXTOS["matriz_confusao_xlabel"])
    ax.set_ylabel(TEXTOS["matriz_confusao_ylabel"])
    ax.set_title(TEXTOS["matriz_confusao_titulo"].format(nome=nome))
    fig.tight_layout()
    fig.savefig(OUT / f"matriz_confusao_{arch}.png", dpi=200)
    plt.close(fig)


def fig_acc_loss_per_epoch(arch, nome, history):
    stage1, stage2 = history["stage1"], history["stage2"]
    n1 = len(stage1)
    epochs = list(range(1, n1 + len(stage2) + 1))
    color = COLORS[arch]

    for metric, key, title_key, ylabel_key, legend_val_key, legend_train_key, ft_key, filename in [
        ("acc", "acc", "acc_epoca_titulo", "acc_epoca_ylabel", "acc_epoca_legenda_val",
         "acc_epoca_legenda_train", "acc_epoca_finetune_label", f"accuracy_epoca_{arch}.png"),
        ("loss", "loss", "loss_epoca_titulo", "loss_epoca_ylabel", "loss_epoca_legenda_val",
         "loss_epoca_legenda_train", "loss_epoca_finetune_label", f"loss_epoca_{arch}.png"),
    ]:
        val_series = [m[f"val_{key}"] for m in stage1] + [m[f"val_{key}"] for m in stage2]
        train_series = [m[f"train_{key}"] for m in stage1] + [m[f"train_{key}"] for m in stage2]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(epochs, val_series, color=color, linestyle="-", linewidth=2, label=TEXTOS[legend_val_key])
        ax.plot(epochs, train_series, color=color, linestyle="--", linewidth=1.5, alpha=0.7,
                label=TEXTOS[legend_train_key])
        if n1 and len(stage2):
            ax.axvline(n1 + 0.5, color="gray", linestyle=":", linewidth=1.3)
            ax.annotate(TEXTOS[ft_key], xy=(n1 + 0.5, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(5, 6), textcoords="offset points", fontsize=10, color="#555555",
                        annotation_clip=False)
        ax.set_xlabel(TEXTOS["acc_epoca_xlabel"] if metric == "acc" else TEXTOS["loss_epoca_xlabel"])
        ax.set_ylabel(TEXTOS[ylabel_key])
        ax.set_title(TEXTOS[title_key].format(nome=nome))
        ax.set_xlim(1, len(epochs))
        fig.subplots_adjust(top=0.86)
        ax.legend(loc="lower right" if metric == "acc" else "upper right")
        virgula_no_eixo(ax)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=200)
        plt.close(fig)


def fig_comparacao_metricas(reports):
    labels = TEXTOS["comparacao_metricas_labels"]
    vals = {}
    for arch in ARCHS:
        r = reports[arch]
        vals[arch] = [r["accuracy"], r["macro avg"]["precision"], r["macro avg"]["recall"], r["macro avg"]["f1-score"]]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars_by_arch = {}
    for i, arch in enumerate(ARCHS):
        offset = (i - 0.5) * width
        xs = [xi + offset for xi in x]
        bars_by_arch[arch] = ax.bar(xs, vals[arch], width, label=ARCHS[arch], color=COLORS[arch])
    ax.set_ylim(0, 1.1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(TEXTOS["comparacao_metricas_ylabel"])
    ax.set_title(TEXTOS["comparacao_metricas_titulo"])
    ax.legend(loc="lower right", framealpha=0.95)
    for arch in ARCHS:
        for bar in bars_by_arch[arch]:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, num(h), ha="center", fontsize=9)
    virgula_no_eixo(ax)
    fig.tight_layout()
    fig.savefig(OUT / "comparacao_metricas.png", dpi=200)
    plt.close(fig)


def fig_comparacao_tempo(grand_totals_seconds):
    archs = list(ARCHS)
    minutos = [grand_totals_seconds[a] / 60 for a in archs]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar([ARCHS[a] for a in archs], minutos, color=[COLORS[a] for a in archs])
    ax.set_ylabel(TEXTOS["comparacao_tempo_ylabel"])
    ax.set_title(TEXTOS["comparacao_tempo_titulo"])
    for bar, v in zip(bars, minutos):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{num(v, 1)} min", ha="center",
                fontsize=10, weight="bold")
    virgula_no_eixo(ax)
    fig.tight_layout()
    fig.savefig(OUT / "comparacao_tempo.png", dpi=200)
    plt.close(fig)


def fig_comparacao_por_classe(reports, classes, metric, title_key, ylabel_key, filename):
    x = range(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, arch in enumerate(ARCHS):
        offset = (i - 0.5) * width
        xs = [xi + offset for xi in x]
        vals = [reports[arch][c][metric] for c in classes]
        bars = ax.bar(xs, vals, width, label=ARCHS[arch], color=COLORS[arch])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, num(v), ha="center", fontsize=10)
    virgula_no_eixo(ax)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes)
    ax.set_ylabel(TEXTOS[ylabel_key])
    ax.set_title(TEXTOS[title_key])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_fluxo_cnn():
    fig, ax = plt.subplots(figsize=(13, 4.6))

    def box(x, y, w, h, title, sub, facecolor, edgecolor):
        fb = plt.matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.6)
        ax.add_patch(fb)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=9, color="#444444")

    def arrow(x0, x1, y0, y1=None):
        y1 = y0 if y1 is None else y1
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.6))

    y0, h = 0.85, 1.0
    box(0.2, y0, 1.5, h, TEXTOS["fluxo_cnn_imagem"], TEXTOS["fluxo_cnn_imagem_sub"], "#f2f2f2", "#888888")
    arrow(1.7, 1.9, y0 + h / 2)

    backbone = [
        (1.9, TEXTOS["fluxo_cnn_conv"], TEXTOS["fluxo_cnn_conv_sub"]),
        (3.9, TEXTOS["fluxo_cnn_norm"], TEXTOS["fluxo_cnn_norm_sub"]),
        (5.9, TEXTOS["fluxo_cnn_ativ"], TEXTOS["fluxo_cnn_ativ_sub"]),
        (7.9, TEXTOS["fluxo_cnn_pool"], TEXTOS["fluxo_cnn_pool_sub"]),
    ]
    for x, title, sub in backbone:
        box(x, y0, 1.7, h, title, sub, "#eaf0fb", "#4C72B0")
    for i in range(len(backbone) - 1):
        x0 = backbone[i][0] + 1.7
        x1 = backbone[i + 1][0]
        arrow(x0, x1, y0 + h / 2)

    dash = plt.matplotlib.patches.FancyBboxPatch(
        (1.75, y0 - 0.55), 8.0, h + 0.8, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="none", edgecolor="#999999", linewidth=1.2, linestyle="--")
    ax.add_patch(dash)
    ax.text(1.9, y0 - 0.42, TEXTOS["fluxo_cnn_backbone_nota"], fontsize=9, style="italic", color="#555555")

    fc_x, fc_w = 10.0, 3.6
    fc_center = fc_x + fc_w / 2
    arrow(9.6, fc_x, y0 + h / 2)
    box(fc_x, y0, fc_w, h, TEXTOS["fluxo_cnn_fc"], TEXTOS["fluxo_cnn_fc_sub"], "#fdeee3", "#DD8452")

    softmax_y = y0 - 1.5
    arrow(fc_center, fc_center, y0 - 0.05, softmax_y + h + 0.05)
    box(fc_x, softmax_y, fc_w, h, TEXTOS["fluxo_cnn_softmax"], TEXTOS["fluxo_cnn_softmax_sub"], "#eaf5ea", "#4C9A4C")

    ax.set_xlim(0, fc_x + fc_w + 0.2)
    ax.set_ylim(softmax_y - 0.15, y0 + h + 0.3)
    ax.axis("off")

    fig.savefig(OUT / "fluxo_cnn.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cpu")
    histories, reports, grand_totals = {}, {}, {}

    for arch, nome in ARCHS.items():
        history = json.loads((CKPT / f"{arch}_history.json").read_text())
        histories[arch] = history

        # Media das 3 execucoes rotuladas (_1/_2/_3), mesmos numeros do artigo -
        # mais robusto que usar a execucao unica sem rotulo (que pode estar com
        # um esquema de timing mais antigo).
        run_totals = []
        for label in ("_1", "_2", "_3"):
            h = json.loads((CKPT / f"{arch}{label}_history.json").read_text())
            t = h["timing"]
            run_totals.append(t.get("grand_total_seconds", t.get("total_training_seconds")))
        grand_totals[arch] = sum(run_totals) / len(run_totals)

        model, ds, test_loader = load_model_and_data(arch, device)
        y_true, y_pred = _predict(model, test_loader, device)
        fig_confusion_matrix(arch, nome, y_true, y_pred, ds.classes)
        fig_acc_loss_per_epoch(arch, nome, history)

        reports[arch] = classification_report_dict(model, test_loader, ds.classes, device)
        classes = ds.classes

    fig_comparacao_metricas(reports)
    fig_comparacao_tempo(grand_totals)
    fig_comparacao_por_classe(reports, classes, "precision", "comparacao_precisao_titulo",
                               "comparacao_precisao_ylabel", "comparacao_precisao_classe.png")
    fig_comparacao_por_classe(reports, classes, "recall", "comparacao_recall_titulo",
                               "comparacao_recall_ylabel", "comparacao_recall_classe.png")
    fig_fluxo_cnn()

    print(f"Todas as figuras foram geradas em: {OUT}")


if __name__ == "__main__":
    main()
