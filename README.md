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
| Dataset | Role |
|---|---|
| Facial Skin Analysis (Kaggle) | Training labels |
| ACNE04-v2 | Cross-dataset acne evaluation |
| SCIN | Skin tone audit |

## Run
pip install -r requirements.txt
streamlit run app.py

## Limitations
Few high-severity samples and limited dark skin representation.
Not a substitute for a dermatologist.
