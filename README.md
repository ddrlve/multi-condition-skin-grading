# Multi-Condition Facial Skin Grading Prediction
## Deep Learning Final Project

One face photo in, severity scores (0 to 5) for six skin conditions out,
plus a heatmap showing which facial area drove each score.

## Why
A face rarely has just one skin problem. Single-label classifiers miss this,
and dermatologist visits are expensive. MOSAIC is an early screening aid,
not a diagnostic tool.

## How it works
ImageNet-pretrained backbone (EfficientNet-B0 / ResNet-50 / ConvNeXt-Tiny)
→ shared features → one output head per condition → Grad-CAM per head.

## Data
| Dataset | Role | Dataset |
|---|---|---|
| Facial Skin Analysis (Kaggle) | Training labels | [Kaggle](https://www.kaggle.com/datasets/killa92/facial-skin-analysis-and-type-classification) |
| ACNE04-v2 | Cross-dataset acne evaluation | [Kaggle](https://www.kaggle.com/datasets/karmagames/acne04-v2) & [Kaggle](https://github.com/AIpourlapeau/acne04v2) |
| SCIN | Skin tone audit | [GitHub](https://github.com/google-research-datasets/scin) |
| Skin Defects | Acne, Redness and Bags Under the Eyes (Optional | [Kaggle](https://www.kaggle.com/datasets/trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes ) |

## Run
pip install -r requirements.txt
streamlit run app.py

## Limitations
Few high-severity samples and limited dark skin representation.
Not a substitute for a dermatologist.
