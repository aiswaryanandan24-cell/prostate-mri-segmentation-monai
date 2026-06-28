# Prostate Gland Segmentation on MRI - Fine-tuning a Pre-trained MONAI Model

## Overview

This project refines a pre-trained deep learning model for **prostate gland segmentation on MRI** using the [MONAI](https://monai.io/) framework done for AI med research lab.

**Task A** demonstrates the *batch-effect* problem: a MONAI UNet pre-trained on the Prostate158 dataset completely fails on a new dataset (PI-CAI / D2), achieving an average Dice score of just **0.0014** - because different hospitals use different MRI scanners, protocols, and patient populations.

**Task B** fine-tunes the same model on D2 with ROI cropping, improving performance to a test Dice of **0.8213** - a ~587× improvement.

---

## Dataset (D2)

80 patients from the public [PI-CAI](https://pi-cai.grand-challenge.org/) dataset.

Each patient folder contains:

```
patient_001/
├── patient_001_t2w.nii.gz    ← T2-weighted MRI (input)
└── patient_001_gland.nii.gz  ← Prostate gland mask (target)
```

---

## Pre-trained Model

| Property | Detail |
|----------|--------|
| Source | [MONAI Model Zoo](https://monai.io/model-zoo.html) - `prostate_mri_anatomy` |
| Architecture | 3D UNet with residual blocks |
| Trained on | Prostate158 dataset |
| Output classes | Background, central gland, peripheral zone |

Downloaded automatically on first run.

---

## Setup

```bash
pip install torch torchvision monai nibabel matplotlib
```

Edit `DATA_DIR` in `prostate_segmentation.py`:

```python
DATA_DIR = r"path/to/PI-CAI/file_data"   # ← change this
```

---

## Usage

```bash
python prostate_segmentation.py
```

Runs the full pipeline:

1. Downloads the pre-trained model (once)
2. **Task A** - evaluates pre-trained model on D2; prints per-patient Dice scores
3. **Task B** - fine-tunes for 10 epochs with ROI cropping; saves best checkpoint
4. Evaluates on held-out test set; saves all visualisation figures

---

## Data Split

| Split | Patients |
|-------|----------|
| Train | 48 |
| Val | 16 |
| Test | 16 |

---

## Key Design Choices

**ROI Cropping** - A bounding box is computed around the prostate gland mask and expanded with a small margin. The crop is resized to 96×96×96. This removes irrelevant background structures, reduces noise, and lets the model focus entirely on the prostate region.

**Loss** - `DiceLoss(to_onehot_y=True, softmax=True)` for 3-class segmentation.

**Optimiser** - Adam, lr = 1e-3, all layers fine-tuned.

**Model selection** - Best checkpoint saved based on validation Dice; reloaded for test evaluation.

---

## Results

| Stage | Avg. Dice |
|-------|-----------|
| Pre-trained (no fine-tuning) | 0.0014 |
| Fine-tuned - validation | 0.7901 |
| Fine-tuned - test (unseen) | **0.8213** |

Test Dice (0.8213) > Val Dice (0.7901) - strong generalisation to unseen patients, no overfitting detected.

**Per-patient range (test set):**
- Best patient: Dice = 0.9018
- Worst patient: Dice = 0.5333

---

## Output Files

| File | Description |
|------|-------------|
| `best_crop_model.pt` | Fine-tuned model weights |
| `results_comparison_bar.png` | Bar chart - pre-trained vs. fine-tuned |
| `results_multislice.png` | Multi-slice prediction grid |
| `best_patient_TPFPFN.png` | TP / FP / FN overlay - best test patient |
| `worst_patient_TPFPFN.png` | TP / FP / FN overlay - worst test patient |
| `random_5_patients.png` | 4-column grid for 5 random test patients |

**Overlay colour code:** Green = True Positive · Red = False Positive · Blue = False Negative

---

## References

- Saha, A. et al. *PI-CAI: Prostate Imaging – Cancer AI* (2022). Grand Challenge dataset.
- MONAI Consortium. *MONAI: Medical Open Network for AI* (2020). https://monai.io
- MONAI Model Zoo - `prostate_mri_anatomy`. https://monai.io/model-zoo.html

---

Aiswarya Perumbilly
Research Assistant
AI-MED LAB
Indiana University,Indianapolis
