# Classificador de defeitos superficiais em placas metálicas (NEU-DET)

Pipeline de transfer learning para classificar 6 tipos de defeito superficial em
chapas de aço a partir de imagens, comparando diferentes arquiteturas de CNN
(ResNet-18, ResNet-34, EfficientNet-B0, MobileNetV3-Small) em acurácia, tempo de
treinamento e latência de inferência.

Dataset: **NEU-DET** (Northeastern University Surface Defect Database) — 6 classes
(`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`),
imagens reais em escala de cinza, 200×200 pixels. Disponível publicamente (buscar por
"NEU-DET" ou "NEU surface defect database"); não está incluído neste repositório.

## Setup

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

Coloque o dataset em `files/data/` (pasta git-ignorada) nesta estrutura:

```text
files/data/
├── train/images/<classe>/       (240 imagens por classe)
└── validation/images/<classe>/  (60 imagens por classe)
```

## Uso

Treinar uma arquitetura (duas etapas: linear probe + fine-tuning):

```powershell
venv\Scripts\python -m src.train --architecture resnet18
```

Gerar o relatório em PDF (métricas, curvas de treino, timing, matriz de confusão):

```powershell
venv\Scripts\python -m src.report --architecture resnet18
```

Arquiteturas disponíveis: `resnet18`, `resnet34`, `efficientnet_b0`, `mobilenet_v3_small`
(registradas em `src/models.py`).

Para repetir uma execução sem sobrescrever a anterior, use `--run-label`:

```powershell
venv\Scripts\python -m src.train --architecture resnet18 --run-label _1
venv\Scripts\python -m src.report --architecture resnet18 --run-label _1
```

Também é possível conduzir o treino/avaliação de forma interativa pelo notebook
`src/classify_defects.ipynb`.

## Estrutura do pipeline (`src/`)

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Configuração central (arquitetura, hiperparâmetros, seed, caminhos) |
| `dataset.py` | Carrega o dataset, faz a divisão validação/teste (estratificada, seed fixa) e aplica as transformações/augmentation |
| `models.py` | Registro de arquiteturas + wrapper que congela o backbone e adapta a camada final |
| `train.py` | Loop de treino em duas etapas (linear probe → fine-tuning), com timing por época |
| `evaluate.py` | Métricas de avaliação: classification report, matriz de confusão, top-2 accuracy, latência de inferência |
| `report.py` | Gera o relatório em PDF a partir dos checkpoints/histórico salvos |

Saídas geradas (git-ignoradas, recriáveis a qualquer momento):

- `src/checkpoints/` — pesos treinados (`*_stage1_best.pth`, `*_stage2_finetuned.pth`) e histórico por época (`*_history.json`)
- `src/reports/` — relatórios em PDF por execução

## Metodologia (resumo)

1. **Transfer learning em duas etapas**: backbone pré-treinado no ImageNet, com a
   camada final substituída para 6 classes.
   - *Etapa 1 (linear probe)*: backbone congelado, só a camada final treina.
   - *Etapa 2 (fine-tuning)*: último bloco convolucional destravado, otimizador novo,
     learning rate 10x menor.
2. **Divisão dos dados**: o dataset original só tem treino/validação — o conjunto de
   validação é dividido 50/50 (estratificado) em validação/teste, para que a avaliação
   final use dados nunca vistos durante o treino.
3. **Reprodutibilidade**: uma seed fixa controla toda a aleatoriedade (divisão dos
   dados, inicialização de pesos, data augmentation). Verificado empiricamente —
   execuções independentes e completas produzem métricas de classificação idênticas.
4. **Métricas de comparação**: precisão, recall, F1-score (por classe e macro), tempo
   total de treinamento e latência de inferência (ms/imagem).

## Outros conteúdos do repositório

`files/docs/mine/` (git-ignorado) contém um artigo em formato ABNT comparando
ResNet-18 e MobileNetV3-Small, com scripts próprios para regenerar o `.docx`
(`gerar_artigo.py`) e as figuras (`gerar_figuras.py`) a partir dos dados reais do
projeto.
