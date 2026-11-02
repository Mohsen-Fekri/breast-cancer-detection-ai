# Breast Ultrasound Classifier with Grad-CAM

[![Live demo](https://img.shields.io/badge/Live%20demo-try%20it%20in%20your%20browser-0F766E)](https://justin-fekri.github.io/medical-image-cancer-detection-prototype/)
![CI](https://github.com/Justin-Fekri/medical-image-cancer-detection-prototype/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C)
![Research only](https://img.shields.io/badge/status-research%20only-important)
![License](https://img.shields.io/badge/license-MIT-green)

An EfficientNet-B0 classifier that scores breast ultrasound images as **malignant** or **normal/benign**, with **Grad-CAM** heatmaps that show which regions drove each prediction. It is trained and evaluated on **BreastMNIST**, a public breast ultrasound benchmark from MedMNIST v2.

**[Try the live demo](https://justin-fekri.github.io/medical-image-cancer-detection-prototype/)**: pick a sample scan or upload your own. The model runs in your browser with ONNX Runtime Web, so nothing is uploaded to a server.

> Educational and research-only prototype. **It is not a diagnostic device** and must not be used to screen, diagnose, treat, or rule out cancer. See [DISCLAIMER.md](DISCLAIMER.md).

![The workbench flagging a malignant test-set scan for review, with the sigmoid probability, threshold, and Grad-CAM overlay.](docs/workbench.png)

## Results

Evaluated once on the official **156-image BreastMNIST test split**, which the model never saw during training.

| Metric (threshold 0.50) | Result |
| --- | ---: |
| ROC-AUC | **0.916** |
| Accuracy | 0.872 |
| Sensitivity (malignant recall) | 0.857 |
| Specificity | 0.877 |
| Precision | 0.720 |
| F1 | 0.783 |

Full run metadata is in [`evaluation/breastmnist_results.json`](evaluation/breastmnist_results.json).

**How it was trained**

- Data: 546 training images (147 malignant, 399 normal or benign), 224×224, official MedMNIST split
- Model: ImageNet-initialized EfficientNet-B0 with a single-logit head; the last 3 feature blocks are fine-tuned, the rest stay frozen
- Class imbalance: positive-class weight in the BCE loss (about 2.7×)
- Augmentation: horizontal flips and rotations within ±10°
- 10 epochs, Adam (lr 1e-3), batch size 16, seed 2026, CPU only

## What the app does

- Standardizes any upload to **224×224 RGB** with ImageNet normalization
- Returns a **sigmoid probability** and compares it with an **adjustable review threshold**, so a reviewer can trade sensitivity for specificity without re-running the model
- Draws a **Grad-CAM** overlay and heatmap from the last convolutional block
- Ships six held-out test scans in `samples/` so the demo works with one click

![Grad-CAM overlay and class-activation heatmap for a malignant test scan.](docs/gradcam-result.png)

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 43127
```

Open `http://127.0.0.1:43127`. A CPU is enough.

The browser demo in `docs/` is the same model exported with `python -m scripts.export_onnx`. The export computes Grad-CAM in the forward pass: after global average pooling and one linear layer, the Grad-CAM weights are the head weights, so the browser needs no gradients. Serve it locally with `python -m http.server --directory docs`.

Or with Docker:

```bash
docker build -t breast-us-classifier .
docker run -p 8501:8501 breast-us-classifier
```

## Reproduce the training run

```bash
# 1. Download breastmnist_224.npz (31 MB) from https://zenodo.org/records/10519652
# 2. Export it to negative/ and positive/ folders
python scripts/prepare_breastmnist.py breastmnist_224.npz --output data/breastmnist

# 3. Train on the train split, evaluate on the held-out test split
python train.py \
  --data-dir data/breastmnist/train \
  --eval-dir data/breastmnist/test \
  --augment --unfreeze-last 3 \
  --epochs 10 --batch-size 16 --lr 1e-3 --seed 2026 \
  --output checkpoints/prototype_efficientnet_b0.pt
```

`train.py` also accepts any folder with `negative/` and `positive/` subfolders. Without `--eval-dir` it makes a stratified 80/20 split. Pass `--unfreeze-backbone` to fine-tune every layer.

## Tests

```bash
python -m pytest -q
```

The tests cover preprocessing, the binary head, threshold logic, Grad-CAM shapes, the train/validation split, and the metric calculations.

## Project layout

```
app.py                         Streamlit workbench
train.py                       Training and evaluation entry point
scripts/prepare_breastmnist.py Export the MedMNIST .npz to image folders
scripts/export_onnx.py         Export the checkpoint for the browser demo
docs/                          Browser demo (GitHub Pages) and screenshots
meddetect/                     Model, preprocessing, Grad-CAM, inference
samples/                       Six held-out BreastMNIST test scans
checkpoints/                   Trained EfficientNet-B0 weights
evaluation/                    Test-set metrics for the committed checkpoint
tests/                         Unit tests
```

## Limitations

- BreastMNIST has only 780 images from a single hospital (the BUSI dataset). Results on other scanners, sites, or populations are unknown.
- The splits are image-level. BUSI does not publish patient identifiers, so images from one patient could appear in more than one split.
- Probabilities are not calibrated, so a score of 0.99 is not a 99% chance of malignancy.
- At the default threshold the model misses about 1 in 7 malignant test images. A screening tool would need a much lower miss rate and external validation.
- Grad-CAM shows which pixels influenced the score. It does not prove the model learned clinically meaningful features.

## Data and citation

BreastMNIST is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you use it, cite:

- Yang, J., Shi, R., Wei, D. et al. MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data* 10, 41 (2023).
- Al-Dhabyani, W., Gomaa, M., Khaled, H., Fahmy, A. Dataset of breast ultrasound images. *Data in Brief* 28, 104863 (2020).

## License

MIT for the code. The license does not grant permission to use this software as a medical device.
