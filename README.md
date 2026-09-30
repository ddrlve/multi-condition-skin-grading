# Multi-Condition Facial Skin Grading Prediction

One face photo in, ordinal severity scores (0 to 5) for six facial skin conditions out, with a Grad-CAM heatmap per condition showing which facial region drove each score.

> **Disclaimer.** MOSAIC is an early screening aid built for academic research. It is **not** a medical device and does not diagnose disease. Consult a dermatologist for any clinical decision.

---

## Why

A face rarely has only one skin problem. Single-label classifiers miss co-occurring conditions and their different severity levels. MOSAIC estimates several conditions jointly from one image with a shared backbone and independent heads, and exposes its reasoning through per-head Grad-CAM.

Full requirements: [`PRD.md`](PRD.md). Execution plan: [`agent.md`](agent.md). Rules for the AI coding agent: [`CLAUDE.md`](CLAUDE.md).

## Architecture

```
               ┌──────────────────────────────┐
 face image ──▶│  Preprocess (resize, norm)   │
 (224x224)     └──────────────┬───────────────┘
                              ▼
               ┌──────────────────────────────┐
               │ Pretrained backbone (timm)   │  EfficientNet-B0 | ResNet-50 | ConvNeXt-Tiny
               │ global pooled features  f    │
               └──────────────┬───────────────┘
            ┌─────────┬───────┴──┬─────────┬─────────┬─────────┐
            ▼         ▼          ▼         ▼         ▼         ▼
         head_1    head_2     head_3    head_4    head_5    head_6     Dropout → Linear
         (acne)     ...                                                 mse: 1 output
                                                                        wce: 6 logits
                                                                        corn: 5 logits
            ▼         ▼          ▼         ▼         ▼         ▼
       severity scores 0..5 per condition  +  Grad-CAM per head
```

- **Backbone:** ImageNet-pretrained, loaded through `timm`, classifier removed (`num_classes=0`).
- **Heads:** one small head per condition, independent parameters.
- **Loss:** masked sum of per-head losses (missing labels ignored). Modes: `mse` regression, `wce` class-weighted cross entropy, optional `corn` ordinal.
- **Baseline:** single-output ResNet-18 trained on acne only with unweighted CE.

## Data

| Dataset                                      | Role                           | Source                                                                                            |
| -------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------- |
| Facial Skin Analysis and Type Classification | Train / val / test             | https://www.kaggle.com/datasets/killa92/facial-skin-analysis-and-type-classification              |
| ACNE04-v2                                    | External eval (acne head only) | https://www.kaggle.com/datasets/karmagames/acne04-v2 , https://github.com/AIpourlapeau/acne04v2   |
| Skin Defects (acne, redness, eye bags)       | Sanity check                   | https://www.kaggle.com/datasets/trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes |
| SCIN                                         | Skin-tone audit (eval only)    | https://github.com/google-research-datasets/scin                                                  |

Datasets are **not** redistributed in this repo. Limits of each dataset and what they can and cannot prove are listed in `PRD.md` Section 5.

## Repository Layout

```
multi-condition-skin-grading/
├── CLAUDE.md                  # rules for the AI coding agent
├── agent.md                   # phased execution plan / backlog
├── PRD.md                     # product requirements
├── README.md
├── requirements.txt
├── pyproject.toml             # ruff + pytest config
├── configs/
│   ├── base.yaml              # shared defaults
│   ├── data.yaml              # paths, selected conditions, split params
│   ├── baseline_resnet18.yaml
│   └── multihead/             # {effnet_b0,resnet50,convnext_tiny}_{mse,wce,corn}.yaml
├── data/                      # gitignored
│   ├── raw/{facial_skin,acne04v2,skin_defects,scin}/
│   ├── interim/               # audit outputs, hashes
│   └── splits/                # train.csv, val.csv, test.csv, split_manifest.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_label_analysis.ipynb
│   └── 03_results_review.ipynb
├── src/mosaic/
│   ├── config.py              # YAML loading + dataclass validation
│   ├── data/
│   │   ├── audit.py           # distribution, missingness, pHash duplicates
│   │   ├── labels.py          # condition selection, class weights
│   │   ├── splits.py          # grouped multilabel stratified split
│   │   ├── datasets.py        # FacialSkinDataset, Acne04Dataset, ScinDataset
│   │   └── transforms.py      # albumentations pipelines
│   ├── models/
│   │   ├── backbones.py       # timm factory + Grad-CAM target layers
│   │   ├── heads.py           # regression / classification / CORN heads
│   │   ├── multihead.py       # MultiHeadSeverityNet
│   │   └── baseline.py        # SingleOutputCNN
│   ├── losses/
│   │   ├── ordinal.py         # masked MSE, weighted CE, CORN
│   │   └── multitask.py       # head weighting
│   ├── training/
│   │   ├── trainer.py         # train/val loop, AMP, grad clipping
│   │   └── callbacks.py       # early stopping, checkpointing, LR logging
│   ├── evaluation/
│   │   ├── metrics.py         # QWK, MAE, off-by-one, macro-F1, Spearman
│   │   ├── bootstrap.py       # CIs and paired tests
│   │   └── plots.py           # all report figures
│   ├── explain/gradcam.py
│   ├── bias/scin_audit.py
│   └── utils/{seed.py,device.py,logging.py,io.py}
├── scripts/
│   ├── download_data.sh
│   ├── audit_data.py
│   ├── make_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── gradcam.py
│   ├── external_eval.py
│   ├── bias_audit.py
│   └── make_figures.py
├── app/streamlit_app.py
├── tests/
└── outputs/                   # gitignored: runs/, checkpoints/, figures/, tables/
```

## Setup

Requires Python 3.10 or 3.11.

```bash
git clone https://github.com/<org>/mosaic-skin-severity.git
cd mosaic-skin-severity
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` (pin exact versions after first successful install with `pip freeze`):

```
torch>=2.2
torchvision>=0.17
timm>=0.9
albumentations>=1.4
opencv-python-headless
numpy
pandas
scikit-learn
scipy
iterative-stratification
imagehash
Pillow
pyyaml
tensorboard
wandb            # optional
grad-cam         # pytorch-grad-cam
matplotlib
seaborn
streamlit
tqdm
pytest
ruff
```

Install PyTorch for your hardware first if needed (CUDA build, or default build for Apple Silicon MPS / CPU), following pytorch.org.

### Download Data

Kaggle datasets need an API token at `~/.kaggle/kaggle.json`.

```bash
bash scripts/download_data.sh
```

What the script does:

```bash
kaggle datasets download -d killa92/facial-skin-analysis-and-type-classification -p data/raw/facial_skin --unzip
kaggle datasets download -d karmagames/acne04-v2 -p data/raw/acne04v2 --unzip
kaggle datasets download -d trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes -p data/raw/skin_defects --unzip
# SCIN: follow the download instructions in the SCIN GitHub README
# (hosted on Google Cloud Storage), save to data/raw/scin/
```

## Usage

All commands read a YAML config. Any key can be overridden with `--set key=value`.

```bash
# 1. Audit and split (run once, commit data/splits/split_manifest.json)
python scripts/audit_data.py --config configs/data.yaml
python scripts/make_splits.py --config configs/data.yaml

# 2. Sanity check: 1 batch forward/backward, overfit-a-batch test
python scripts/train.py --config configs/multihead/effnet_b0_wce.yaml --dry-run

# 3. Train
python scripts/train.py --config configs/baseline_resnet18.yaml --seed 42
python scripts/train.py --config configs/multihead/effnet_b0_wce.yaml --seed 42

# 4. Evaluate on test (final phase only)
python scripts/evaluate.py --run outputs/runs/<run_id> --split test --bootstrap 1000

# 5. Explain
python scripts/gradcam.py --run outputs/runs/<run_id> --n-images 24

# 6. External and bias evaluation
python scripts/external_eval.py --run outputs/runs/<run_id> --dataset acne04v2
python scripts/bias_audit.py --run outputs/runs/<run_id>

# 7. Inference
python scripts/infer.py --run outputs/runs/<run_id> --image path/to/face.jpg

# 8. Figures for the report
python scripts/make_figures.py --runs outputs/runs/ --out outputs/figures/

# 9. App
streamlit run app/streamlit_app.py
```

Monitor training:

```bash
tensorboard --logdir outputs/runs
```

## Results

Filled in after Phase 7. Test metrics are reported as mean ± std over 3 seeds, with 95% bootstrap CIs for the best model.

| Model                      | Loss      | Mean QWK | Mean MAE | Acne QWK |
| -------------------------- | --------- | -------- | -------- | -------- |
| ResNet-18 single-output    | CE        | –        | –        | –        |
| EfficientNet-B0 multi-head | MSE / WCE | –        | –        | –        |
| ResNet-50 multi-head       | MSE / WCE | –        | –        | –        |
| ConvNeXt-Tiny multi-head   | MSE / WCE | –        | –        | –        |

## Limitations

- Few samples at high severity levels for several conditions.
- ACNE04-v2 validates only the acne head, on a different grading scale.
- SCIN has no severity labels; the skin-tone audit uses proxy metrics.
- Dark skin tones are underrepresented in the training data.

## Team

Dian Rakhmawati Lestari

## License

Code: MIT (or as required by the course). Datasets remain under their original licenses and are used for non-commercial research only.
