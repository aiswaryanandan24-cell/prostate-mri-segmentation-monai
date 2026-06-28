"""
Prostate Gland Segmentation on MRI — Fine-tuning a Pre-trained MONAI Model
============================================================================

    a) Evaluate the pre-trained MONAI 'prostate_mri_anatomy' model on dataset D2
       to demonstrate the batch-effect problem.
    b) Fine-tune the pre-trained model on D2 to improve segmentation performance.

Dataset (D2): PI-CAI public subset — 80 patients, each with:
    <patient_id>_t2w.nii.gz   — T2-weighted MRI
    <patient_id>_gland.nii.gz — Prostate gland segmentation mask

Pre-trained model: MONAI Model Zoo — 'prostate_mri_anatomy'
    https://monai.io/model-zoo.html
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import random

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from monai.bundle import ConfigParser, download
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Paths  (edit DATA_DIR to point at your local dataset)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = r"path/to/PI-CAI/file_data"          # ← change this
MODEL_BUNDLE_DIR = os.path.join(DATA_DIR, "model_bundle", "prostate_mri_anatomy")
WEIGHTS_PATH    = os.path.join(MODEL_BUNDLE_DIR, "models", "model.pt")
BEST_MODEL_PATH = "best_crop_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Download pre-trained model (run once)
# ─────────────────────────────────────────────────────────────────────────────
def download_model():
    """Download the MONAI 'prostate_mri_anatomy' bundle if not already present."""
    if os.path.exists(MODEL_BUNDLE_DIR):
        print("Model bundle already exists — skipping download.")
        return
    download(
        name="prostate_mri_anatomy",
        bundle_dir=os.path.join(DATA_DIR, "model_bundle"),
    )
    print("Model downloaded")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_patient_list(data_dir: str) -> list[str]:
    """Return sorted list of numeric patient IDs (filters out non-patient folders)."""
    return sorted(p for p in os.listdir(data_dir) if p.isdigit())


def load_model_from_bundle(weights_path: str = WEIGHTS_PATH) -> torch.nn.Module:
    """Load the pre-trained network from the MONAI bundle config + weights."""
    parser = ConfigParser()
    parser.read_config(os.path.join(MODEL_BUNDLE_DIR, "configs", "inference.json"))
    network = parser.get_parsed_content("network")
    network.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return network.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Evaluation metric
# ─────────────────────────────────────────────────────────────────────────────
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Binary Dice score between prediction and ground-truth arrays.
    Any non-zero value is treated as foreground (gland).
    """
    pred_bin = (pred > 0).astype(float)
    gt_bin   = (gt   > 0).astype(float)
    intersection = (pred_bin * gt_bin).sum()
    return float((2.0 * intersection) / (pred_bin.sum() + gt_bin.sum() + 1e-6))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Task A — Evaluate pre-trained model on D2 (batch-effect demonstration)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_pretrained(network: torch.nn.Module,
                        patients: list[str],
                        n_patients: int = 5) -> float:
    """
    Run sliding-window inference on the first `n_patients` using the
    pre-trained model (no fine-tuning). Returns the average Dice score.

    The very low score demonstrates the batch-effect: the model was trained
    on a different scanner/protocol and does not generalise to D2.
    """
    network.eval()
    scores = []

    for p in patients[:n_patients]:
        t2_img    = nib.load(os.path.join(DATA_DIR, p, f"{p}_t2w.nii.gz")).get_fdata()
        gland_img = nib.load(os.path.join(DATA_DIR, p, f"{p}_gland.nii.gz")).get_fdata()

        # Normalise
        t2_norm = (t2_img - t2_img.min()) / (t2_img.max() - t2_img.min() + 1e-8)

        # Resize depth from native 32 → 96 to match model ROI
        t2_tensor  = torch.tensor(t2_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t2_resized = F.interpolate(t2_tensor, size=(256, 256, 96),
                                   mode="trilinear", align_corners=False).to(DEVICE)

        with torch.no_grad():
            output = sliding_window_inference(
                t2_resized, roi_size=(96, 96, 96),
                sw_batch_size=1, overlap=0.5, predictor=network,
            )

        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Resize prediction back to original depth for fair Dice comparison
        pred_tensor  = torch.tensor(pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pred_resized = F.interpolate(pred_tensor, size=(256, 256, 32),
                                     mode="nearest").squeeze().numpy()

        d = dice_score(pred_resized, gland_img)
        scores.append(d)
        print(f"  Patient {p}: Dice = {d:.4f}")

    avg = float(np.mean(scores))
    print(f"\nAverage Dice (pre-trained, no fine-tuning): {avg:.4f}")
    print("→ Near-zero Dice confirms batch-effect: model does not generalise to D2.")
    return avg


def visualise_pretrained(network: torch.nn.Module, patient_id: str):
    """Overlay pre-trained prediction on T2 scan for one patient."""
    network.eval()
    t2_img    = nib.load(os.path.join(DATA_DIR, patient_id, f"{patient_id}_t2w.nii.gz")).get_fdata()
    gland_img = nib.load(os.path.join(DATA_DIR, patient_id, f"{patient_id}_gland.nii.gz")).get_fdata()

    t2_norm    = (t2_img - t2_img.min()) / (t2_img.max() - t2_img.min() + 1e-8)
    t2_tensor  = torch.tensor(t2_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t2_resized = F.interpolate(t2_tensor, size=(256, 256, 96),
                               mode="trilinear", align_corners=False).to(DEVICE)

    with torch.no_grad():
        output = sliding_window_inference(
            t2_resized, roi_size=(96, 96, 96),
            sw_batch_size=1, overlap=0.5, predictor=network,
        )

    pred        = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    pred_tensor = torch.tensor(pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pred_orig   = F.interpolate(pred_tensor, size=(256, 256, 32),
                                mode="nearest").squeeze().numpy()

    mid = t2_img.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(t2_img[:, :, mid], cmap="gray")
    axes[0].set_title("T2 MRI scan")
    axes[0].axis("off")

    axes[1].imshow(gland_img[:, :, mid], cmap="gray")
    axes[1].set_title("Ground truth gland")
    axes[1].axis("off")

    axes[2].imshow(t2_img[:, :, mid], cmap="gray")
    axes[2].imshow(pred_orig[:, :, mid], cmap="Reds", alpha=0.5)
    axes[2].imshow(gland_img[:, :, mid], cmap="Greens", alpha=0.4)
    axes[2].set_title("Red = predicted | Green = ground truth")
    axes[2].axis("off")

    d = dice_score(pred_orig, gland_img)
    plt.suptitle(f"Patient {patient_id} — Pre-trained model (Dice = {d:.4f})")
    plt.tight_layout()
    plt.savefig(f"pretrained_{patient_id}.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Task B — Fine-tune on D2
# ─────────────────────────────────────────────────────────────────────────────

# ── 6a. ROI-cropping utilities ──────────────────────────────────────────────

def get_crop_coords(gland: np.ndarray,
                    margin: int = 15,
                    threshold: float = 0.5):
    """
    Compute the 3-D bounding box around non-zero gland voxels + margin.
    Returns (x1, x2, y1, y2, z1, z2) or None if the gland is empty.
    """
    coords = np.argwhere(gland >= threshold)
    if len(coords) == 0:
        return None

    x1, y1, z1 = coords.min(axis=0)
    x2, y2, z2 = coords.max(axis=0)

    x1 = max(0, x1 - margin);  x2 = min(gland.shape[0], x2 + margin)
    y1 = max(0, y1 - margin);  y2 = min(gland.shape[1], y2 + margin)
    z1 = max(0, z1 - margin);  z2 = min(gland.shape[2], z2 + margin)

    return x1, x2, y1, y2, z1, z2


def load_and_crop(patient_id: str,
                  data_dir: str = DATA_DIR,
                  threshold: float = 0.5,
                  margin: int = 15,
                  target_size: tuple = (96, 96, 96)):
    """
    Load T2 + gland for one patient, crop tightly around the prostate ROI,
    normalise, and resize to `target_size`. Returns (t2_tensor, gland_tensor)
    with shape (1, D, H, W), or (None, None) if no gland is found.
    """
    t2    = nib.load(os.path.join(data_dir, patient_id, f"{patient_id}_t2w.nii.gz")).get_fdata()
    gland = nib.load(os.path.join(data_dir, patient_id, f"{patient_id}_gland.nii.gz")).get_fdata()

    gland_clean = (gland >= threshold).astype(np.float32)
    coords = get_crop_coords(gland, margin=margin, threshold=threshold)
    if coords is None:
        print(f"  Warning: no gland found for patient {patient_id} — skipping.")
        return None, None

    x1, x2, y1, y2, z1, z2 = coords
    t2_crop    = t2[x1:x2, y1:y2, z1:z2]
    gland_crop = gland_clean[x1:x2, y1:y2, z1:z2]

    # Min-max normalise T2
    t2_crop = (t2_crop - t2_crop.min()) / (t2_crop.max() - t2_crop.min() + 1e-8)

    # Convert to tensors and resize
    t2_t = torch.tensor(t2_crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    g_t  = torch.tensor(gland_crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    t2_t = F.interpolate(t2_t, size=target_size, mode="trilinear", align_corners=False)
    g_t  = F.interpolate(g_t,  size=target_size, mode="nearest")

    return t2_t.squeeze(0), g_t.squeeze(0)


# ── 6b. Training loop ────────────────────────────────────────────────────────

def fine_tune(train_patients: list[str],
              val_patients: list[str],
              n_epochs: int = 10,
              lr: float = 1e-3,
              save_path: str = BEST_MODEL_PATH) -> torch.nn.Module:
    """
    Fine-tune the pre-trained model on D2 training patients.
    Uses ROI cropping + DiceLoss (softmax, one-hot) for 3-class segmentation.
    Saves the best-validation-Dice checkpoint to `save_path`.

    Returns the best model ready for evaluation.
    """
    network = load_model_from_bundle()
    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_fn   = DiceLoss(to_onehot_y=True, softmax=True)

    best_val_dice = 0.0

    for epoch in range(n_epochs):
        # ── Training ──
        network.train()
        train_loss = 0.0
        n_train    = 0

        for p in train_patients:
            t2_t, g_t = load_and_crop(p)
            if t2_t is None:
                continue

            t2_t = t2_t.unsqueeze(0).to(DEVICE)
            g_t  = g_t.unsqueeze(0).long().to(DEVICE)

            optimizer.zero_grad()
            loss = loss_fn(network(t2_t), g_t)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train    += 1

        avg_train = train_loss / max(n_train, 1)

        # ── Validation ──
        network.eval()
        val_dices = []

        for p in val_patients:
            t2_t, g_t = load_and_crop(p)
            if t2_t is None:
                continue
            with torch.no_grad():
                out  = network(t2_t.unsqueeze(0).to(DEVICE))
            pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
            gt   = g_t.squeeze().numpy()
            val_dices.append(dice_score(pred, gt))

        avg_val = float(np.mean(val_dices)) if val_dices else 0.0

        marker = ""
        if avg_val > best_val_dice:
            best_val_dice = avg_val
            torch.save(network.state_dict(), save_path)
            marker = " *** BEST SAVED"

        print(f"Epoch {epoch+1:02d}/{n_epochs} — "
              f"train loss: {avg_train:.4f} | val Dice: {avg_val:.4f}{marker}")

    print(f"\nBest validation Dice: {best_val_dice:.4f}")
    print(f"Weights saved to: {save_path}")

    # Reload best weights
    best_model = load_model_from_bundle(save_path)
    best_model.eval()
    return best_model


# ── 6c. Test evaluation ──────────────────────────────────────────────────────

def evaluate_finetuned(model: torch.nn.Module,
                       test_patients: list[str]) -> tuple[list, float]:
    """
    Evaluate the fine-tuned model on held-out test patients.
    Returns (per-patient dice scores list, average dice).
    """
    model.eval()
    dices = []

    for p in test_patients:
        t2_t, g_t = load_and_crop(p)
        if t2_t is None:
            continue
        with torch.no_grad():
            out  = model(t2_t.unsqueeze(0).to(DEVICE))
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
        gt   = g_t.squeeze().numpy()
        d    = dice_score(pred, gt)
        dices.append(d)
        print(f"  Patient {p}: Dice = {d:.4f}")

    avg = float(np.mean(dices))
    print(f"\nAverage Dice (fine-tuned, test set): {avg:.4f}")
    return dices, avg


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_bar(pretrained_dice: float,
                     val_dice: float,
                     test_dice: float):
    """Bar chart comparing pre-trained vs. fine-tuned Dice scores."""
    labels = ["Pre-trained\n(no fine-tuning)", "Fine-tuned\n(validation)", "Fine-tuned\n(test)"]
    values = [pretrained_dice, val_dice, test_dice]
    colours = ["#e74c3c", "#e67e22", "#27ae60"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colours, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average Dice Score", fontsize=13)
    ax.set_title("Effect of Fine-tuning — Prostate Gland Segmentation", fontsize=13)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results_comparison_bar.png")


def plot_multi_slice(model: torch.nn.Module,
                     patient_id: str,
                     slices: list[int] = None,
                     save_name: str = "multi_slice.png"):
    """
    3-row grid: T2 scan | Ground truth (green) | Prediction (red)
    across multiple axial slices for one patient.
    """
    t2_t, g_t = load_and_crop(patient_id)
    if t2_t is None:
        return

    model.eval()
    with torch.no_grad():
        out  = model(t2_t.unsqueeze(0).to(DEVICE))
    pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
    gt   = g_t.squeeze().numpy()

    depth = t2_t.shape[-1]
    if slices is None:
        step   = depth // 6
        slices = list(range(step, depth - step, step))[:5]

    fig, axes = plt.subplots(3, len(slices), figsize=(4 * len(slices), 12))

    for i, s in enumerate(slices):
        t2_sl = t2_t[0, :, :, s].numpy()

        axes[0, i].imshow(t2_sl, cmap="gray")
        axes[0, i].set_title(f"Slice {s}", fontsize=11)
        axes[0, i].axis("off")

        axes[1, i].imshow(t2_sl, cmap="gray")
        axes[1, i].imshow(gt[:, :, s], cmap="Greens", alpha=0.5)
        axes[1, i].axis("off")

        axes[2, i].imshow(t2_sl, cmap="gray")
        axes[2, i].imshow((pred[:, :, s] > 0), cmap="Reds", alpha=0.5)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("T2 MRI scan", fontsize=12)
    axes[1, 0].set_ylabel("Ground truth (green)", fontsize=12)
    axes[2, 0].set_ylabel("Prediction (red)", fontsize=12)

    d = dice_score(pred, gt)
    fig.suptitle(f"Patient {patient_id} — Dice: {d:.4f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_name}")


def plot_tp_fp_fn(model: torch.nn.Module,
                  patient_id: str,
                  save_name: str = "tp_fp_fn.png"):
    """
    4-panel figure for the slice with the most gland content:
    MRI | Ground truth | Prediction | TP (green) / FP (red) / FN (blue)
    """
    t2_t, g_t = load_and_crop(patient_id)
    if t2_t is None:
        return

    model.eval()
    with torch.no_grad():
        out  = model(t2_t.unsqueeze(0).to(DEVICE))
    pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
    gt   = g_t.squeeze().numpy()

    # Find slice with most gland
    best_s = int(np.argmax([gt[:, :, s].sum() for s in range(gt.shape[2])]))

    pred_bin = (pred[:, :, best_s] > 0).astype(np.uint8)
    gt_bin   = (gt[:, :, best_s] > 0.5).astype(np.uint8)

    TP = (pred_bin == 1) & (gt_bin == 1)
    FP = (pred_bin == 1) & (gt_bin == 0)
    FN = (pred_bin == 0) & (gt_bin == 1)

    t2_sl   = t2_t[0, :, :, best_s].numpy()
    overlay = np.stack([t2_sl] * 3, axis=-1).copy()
    overlay[TP] = [0.0, 0.8, 0.0]
    overlay[FP] = [0.9, 0.1, 0.1]
    overlay[FN] = [0.1, 0.1, 0.9]

    d = dice_score(pred, gt)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("white")

    axes[0].imshow(t2_sl, cmap="gray"); axes[0].set_title("MRI Scan"); axes[0].axis("off")

    gt_rgb = np.zeros((*t2_sl.shape, 3))
    gt_rgb[gt_bin == 1] = [0.0, 0.5, 0.0]
    axes[1].imshow(t2_sl, cmap="gray"); axes[1].imshow(gt_rgb, alpha=0.7)
    axes[1].set_title("Ground Truth"); axes[1].axis("off")

    pred_rgb = np.zeros((*t2_sl.shape, 3))
    pred_rgb[pred_bin == 1] = [0.1, 0.3, 0.8]
    axes[2].imshow(t2_sl, cmap="gray"); axes[2].imshow(pred_rgb, alpha=0.7)
    axes[2].set_title("Model Prediction"); axes[2].axis("off")

    axes[3].imshow(t2_sl, cmap="gray"); axes[3].imshow(overlay, alpha=0.7)
    axes[3].set_title("Green=TP | Red=FP | Blue=FN"); axes[3].axis("off")

    fig.suptitle(f"Patient {patient_id} | Slice {best_s} | Dice: {d:.4f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_name}")


def plot_random_patients(model: torch.nn.Module,
                         patient_list: list[str],
                         n: int = 5,
                         save_name: str = "random_patients.png"):
    """
    4-column grid (MRI, GT, Pred, TP/FP/FN) for `n` randomly chosen patients.
    Dice score colour-coded green/orange/red by threshold.
    """
    sample = random.sample(patient_list, min(n, len(patient_list)))
    model.eval()

    fig, axes = plt.subplots(len(sample), 4, figsize=(20, 5 * len(sample)))
    fig.patch.set_facecolor("white")

    for row, p in enumerate(sample):
        t2_t, g_t = load_and_crop(p)
        if t2_t is None:
            continue

        with torch.no_grad():
            out  = model(t2_t.unsqueeze(0).to(DEVICE))
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
        gt   = g_t.squeeze().numpy()

        best_s   = int(np.argmax([gt[:, :, s].sum() for s in range(gt.shape[2])]))
        pred_bin = (pred[:, :, best_s] > 0).astype(np.uint8)
        gt_bin   = (gt[:, :, best_s] > 0.5).astype(np.uint8)

        TP = (pred_bin == 1) & (gt_bin == 1)
        FP = (pred_bin == 1) & (gt_bin == 0)
        FN = (pred_bin == 0) & (gt_bin == 1)

        t2_sl   = t2_t[0, :, :, best_s].numpy()
        overlay = np.stack([t2_sl] * 3, axis=-1).copy()
        overlay[TP] = [0.0, 0.8, 0.0]
        overlay[FP] = [0.9, 0.1, 0.1]
        overlay[FN] = [0.1, 0.1, 0.9]

        d = dice_score(pred, gt)

        titles = ["MRI Scan", "Ground Truth", "Prediction", "Green=TP|Red=FP|Blue=FN"]
        for col, (title, img, cmap) in enumerate(zip(
            titles,
            [t2_sl, None, None, None],
            ["gray", None, None, None],
        )):
            ax = axes[row, col]
            ax.axis("off")
            ax.set_title(title if row == 0 else "", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"Patient {p}", fontsize=10)

        axes[row, 0].imshow(t2_sl, cmap="gray")
        axes[row, 0].set_ylabel(f"Patient {p}", fontsize=10)

        gt_rgb = np.zeros((*t2_sl.shape, 3)); gt_rgb[gt_bin == 1] = [0.0, 0.5, 0.0]
        axes[row, 1].imshow(t2_sl, cmap="gray"); axes[row, 1].imshow(gt_rgb, alpha=0.7)

        pred_rgb = np.zeros((*t2_sl.shape, 3)); pred_rgb[pred_bin == 1] = [0.1, 0.3, 0.8]
        axes[row, 2].imshow(t2_sl, cmap="gray"); axes[row, 2].imshow(pred_rgb, alpha=0.7)

        axes[row, 3].imshow(t2_sl, cmap="gray"); axes[row, 3].imshow(overlay, alpha=0.7)

        colour = "green" if d > 0.7 else ("orange" if d > 0.4 else "red")
        axes[row, 3].text(1.02, 0.5, f"Dice:\n{d:.4f}",
                          transform=axes[row, 3].transAxes,
                          fontsize=12, fontweight="bold",
                          va="center", ha="left", color=colour)

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Setup ──
    download_model()

    patients = get_patient_list(DATA_DIR)
    print(f"Total patients found: {len(patients)}")

    # 60 / 20 / 20 split
    train_patients = patients[:48]
    val_patients   = patients[48:64]
    test_patients  = patients[64:]
    print(f"Split — Train: {len(train_patients)} | Val: {len(val_patients)} | Test: {len(test_patients)}")

    # ─────────────────────────────────────
    # Task A: Pre-trained model evaluation
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK A — Pre-trained model on D2 (batch-effect demo)")
    print("=" * 60)

    pretrained_net = load_model_from_bundle()
    pretrained_net.eval()

    pretrained_avg_dice = evaluate_pretrained(pretrained_net, patients, n_patients=5)
    visualise_pretrained(pretrained_net, patients[0])

    # ─────────────────────────────────────
    # Task B: Fine-tuning
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK B — Fine-tuning on D2")
    print("=" * 60)

    best_model = fine_tune(
        train_patients=train_patients,
        val_patients=val_patients,
        n_epochs=10,
        lr=1e-3,
        save_path=BEST_MODEL_PATH,
    )

    # ─────────────────────────────────────
    # Test evaluation & visualisation
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating fine-tuned model on held-out test set")
    print("=" * 60)

    test_dices, avg_test_dice = evaluate_finetuned(best_model, test_patients)

    # Summary bar chart
    # Note: best_val_dice is printed during fine_tune; replace 0.7901 with your run's value
    BEST_VAL_DICE = 0.7901   # ← update after your training run
    plot_results_bar(
        pretrained_dice=pretrained_avg_dice,
        val_dice=BEST_VAL_DICE,
        test_dice=avg_test_dice,
    )

    # Per-patient analysis
    best_idx  = int(np.argmax(test_dices))
    worst_idx = int(np.argmin(test_dices))
    print(f"\nBest  test patient: {test_patients[best_idx]}  — Dice: {test_dices[best_idx]:.4f}")
    print(f"Worst test patient: {test_patients[worst_idx]} — Dice: {test_dices[worst_idx]:.4f}")

    # Multi-slice visualisation for first test patient
    plot_multi_slice(best_model, test_patients[0],
                     save_name="results_multislice.png")

    # TP/FP/FN overlays for best and worst patients
    plot_tp_fp_fn(best_model, test_patients[best_idx],
                  save_name="best_patient_TPFPFN.png")
    plot_tp_fp_fn(best_model, test_patients[worst_idx],
                  save_name="worst_patient_TPFPFN.png")

    # Grid of 5 random patients
    plot_random_patients(best_model, test_patients,
                         n=5, save_name="random_5_patients.png")

    print("\n✓ All done! Check the saved .png files for results.")


if __name__ == "__main__":
    main()
