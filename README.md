## **Lightweight Efficient Training & Model Optimization Toolkit**

---

| Goal                     | What We Implement                                   |
| ------------------------ | --------------------------------------------------- |
| Efficient Training    | Semi-Supervised + Fine-Tuning modes                 |
| Optimization Research | Distillation, learning rate scheduling              |
| Model Compression     | Pruning + Quantization (optional integration)       |
| Low-Label Learning    | Pseudo-labeling pipeline for real-world constraints |

🔹 Built for CIFAR-10 (auto-downloads).
🔹 Cleanly modular, good for experimentation & publication.

---

## Features

✅ 3 Training Modes

| Mode       | Description                                    | Command           |
| ---------- | ---------------------------------------------- | ----------------- |
| `baseline` | Train model fully supervised                   | `--mode baseline` |
| `finetune` | Train only classifier head (transfer learning) | `--mode finetune` |
| `pseudo`   | Semi-supervised with pseudo-labels             | `--mode pseudo`   |

- Efficient Backbones: `ResNet18`, `MobileNetV2`
- Auto dataset: CIFAR-10
- Checkpointing + Metrics + Learning curves
- Plots & confusion matrix
- Optional Distillation for performance vs. efficiency

---

## Project Structure

```
efficient-ml/
├─ data/                       # CIFAR-10 via torchvision (auto-download)
├─ src/
│  ├─ train.py                 # main training script (modes: baseline, finetune, pseudo)
│  ├─ eval.py                  # evaluation & produce confusion matrix
│  ├─ models.py                # model definitions (resnet18, mobilenet)
│  ├─ dataset.py               # labeled/unlabeled loaders, low-label split util
│  ├─ utils.py                 # training utilities, metrics, checkpointing
│  ├─ distill.py               # optional knowledge distillation recipe
│  └─ plots.py                 # plot training curves & comparisons
├─ notebooks/
│  └─ demo.ipynb               # quick-run demo notebook (one experiment)
├─ experiments/                # yaml configs or experiment notes
├─ requirements.txt
├─ README.md
└─ run_demo.sh
```

---

## Installation

```bash
git clone https://github.com/yourname/efficient-ml.git
cd efficient-ml
pip install -r requirements.txt
```

---

## Quickstart

Run default baseline training:

```bash
python src/train.py --mode baseline --epochs 10
```

Fine-tuning MobileNet:

```bash
python src/train.py --mode finetune --model mobilenet --epochs 5
```

Semi-supervised (Pseudo-labeling):

```bash
python src/train.py --mode pseudo --low-label 0.1
```

Evaluate & generate confusion matrix:

```bash
python src/eval.py --checkpoint outputs/model_best.pth
```

Plot training curves:

```bash
python src/plots.py
```

---
