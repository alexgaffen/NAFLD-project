"""
Tiny U-Net PSR fibrosis inference module.

Self-contained inference code extracted from
psr_pseudolabel_tiny_unet_v14OnlyTrain.py. Importable on CPU-only
machines. Provides:

    - TinyUNet model
    - Pseudo-label / multichannel input helpers required by the model
    - load_model(checkpoint_path, device=None) -> TinyUNet
    - run_inference(model, img_rgb, device=None, threshold=0.5) -> dict
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from skimage.filters import frangi
from skimage.measure import label, regionprops


FRANGI_WEIGHT = 0.01


# ── Basic helpers ────────────────────────────────────────────────────────

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin)


def remove_small_components(mask_bool: np.ndarray, min_size: int = 40) -> np.ndarray:
    lbl = label(mask_bool)
    out = np.zeros_like(mask_bool, dtype=bool)
    for r in regionprops(lbl):
        if r.area >= min_size:
            out[lbl == r.label] = True
    return out


def filter_fibrous_components(mask_bool: np.ndarray,
                              min_area: int = 20,
                              min_eccentricity: float = 0.85,
                              min_aspect_ratio: float = 2.0,
                              keep_large: bool = False) -> np.ndarray:
    lbl = label(mask_bool)
    out = np.zeros_like(mask_bool, dtype=bool)
    for r in regionprops(lbl):
        if r.area < min_area:
            continue

        major = max(float(r.major_axis_length), 1.0)
        minor = max(float(r.minor_axis_length), 1.0)
        aspect_ratio = major / minor
        ecc = float(r.eccentricity)

        keep = (ecc >= min_eccentricity) or (aspect_ratio >= min_aspect_ratio)
        if keep_large and r.area >= 400:
            keep = True

        if keep:
            out[lbl == r.label] = True
    return out


def get_tissue_mask(
    img_array,
    gray_thresh: int = 238,
    min_sum: int = 25,
    white_thr: float = 0.92,
    sat_thr: float = 0.05,
    min_obj_size: int = 64,
    use_morphology: bool = True,
):
    """
    Robust tissue mask for regular RGB images and SVS patches.
    Returns boolean mask of shape [H, W].
    """
    img = img_array.astype(np.float32)

    if img.max() <= 1.5:
        rgb01 = img
        img255 = img * 255.0
    else:
        rgb01 = img / 255.0
        img255 = img

    gray01 = rgb01.mean(axis=2)
    gray255 = img255.mean(axis=2)

    maxc = rgb01.max(axis=2)
    minc = rgb01.min(axis=2)
    sat = (maxc - minc) / (maxc + 1e-8)

    bg_white = (gray01 > white_thr) & (sat < sat_thr)
    bg_simple = (img255.sum(axis=-1) <= min_sum) | (gray255 >= gray_thresh)

    tissue_mask = ~(bg_white | bg_simple)
    tissue_mask = tissue_mask.astype(np.uint8)

    if use_morphology:
        try:
            kernel = np.ones((3, 3), np.uint8)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                tissue_mask,
                connectivity=8
            )

            clean = np.zeros_like(tissue_mask, dtype=np.uint8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_obj_size:
                    clean[labels == i] = 1

            tissue_mask = clean

        except ImportError:
            pass

    return tissue_mask.astype(bool)


def extract_psr_color_candidates(img_rgb: np.ndarray,
                                 tissue_gray_thresh: int = 240,
                                 min_saturation: int = 60,
                                 min_value: int = 40,
                                 min_a_channel: int = 140,
                                 red_hue_ranges=((0, 20), (160, 179)),
                                 min_size: int = 30) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    tissue_mask = gray < tissue_gray_thresh

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    red_hue_mask = np.zeros_like(gray, dtype=bool)
    for hmin, hmax in red_hue_ranges:
        red_hue_mask |= ((h >= hmin) & (h <= hmax))

    red_hsv = red_hue_mask & (s >= min_saturation) & (v >= min_value)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1]
    red_lab = a >= min_a_channel

    color_mask = tissue_mask & red_hsv & red_lab
    color_mask = remove_small_components(color_mask, min_size=min_size)

    return {
        'tissue_mask': tissue_mask,
        'red_hsv': red_hsv,
        'red_lab': red_lab,
        'color_mask': color_mask,
        'lab_a_norm': normalize01(a.astype(np.float32))
    }


def compute_frangi_response(img_rgb: np.ndarray,
                            color_mask: Optional[np.ndarray] = None,
                            use_lab_a: bool = True,
                            sigmas=(1, 2, 3),
                            beta=0.5,
                            gamma=15) -> np.ndarray:
    if use_lab_a:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        work = lab[:, :, 1].astype(np.float32)
    else:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        work = hsv[:, :, 1].astype(np.float32)

    work = normalize01(work)
    response = frangi(work, sigmas=sigmas, black_ridges=False, beta=beta, gamma=gamma)
    response = normalize01(response)

    if color_mask is not None:
        response = response * color_mask.astype(np.float32)

    return response.astype(np.float32)


def build_pseudo_labels(img_rgb: np.ndarray,
                        min_saturation: int = 65,
                        min_value: int = 45,
                        min_a_channel: int = 145,
                        frangi_high: float = 0.05,
                        frangi_low: float = 0.005,
                        min_size: int = 8,
                        min_area: int = 8,
                        min_eccentricity: float = 0.45,
                        min_aspect_ratio: float = 1.4,
                        weak_positive_from_color: bool = True,
                        weak_lab_a_thr: float = 0.58,
                        neg_lab_a_thr: float = 0.50,
                        strong_pos_weight: float = 1.0,
                        weak_pos_weight: float = 0.25,
                        neg_weight: float = 0.70) -> Dict[str, np.ndarray]:

    parts = extract_psr_color_candidates(
        img_rgb,
        min_saturation=min_saturation,
        min_value=min_value,
        min_a_channel=min_a_channel,
        min_size=min_size
    )

    tissue_mask = get_tissue_mask(img_rgb).astype(bool)

    color_mask_raw = parts["color_mask"].astype(bool)
    color_mask = color_mask_raw & tissue_mask

    lab_a_norm = parts["lab_a_norm"]

    frangi_resp = compute_frangi_response(
        img_rgb,
        color_mask=color_mask,
        use_lab_a=True
    )

    frangi_resp = frangi_resp * tissue_mask.astype(np.float32)

    line_mask = (frangi_resp >= frangi_high) & tissue_mask

    fibrous_mask = filter_fibrous_components(
        line_mask,
        min_area=min_area,
        min_eccentricity=min_eccentricity,
        min_aspect_ratio=min_aspect_ratio,
        keep_large=False
    )

    strong_pos = color_mask & fibrous_mask & tissue_mask
    strong_pos = remove_small_components(strong_pos, min_size=max(6, min_size))
    strong_pos = strong_pos & tissue_mask

    if weak_positive_from_color:
        weak_pos = (
            color_mask
            & (~strong_pos)
            & (lab_a_norm >= weak_lab_a_thr)
            & tissue_mask
        )
    else:
        weak_pos = np.zeros_like(strong_pos, dtype=bool)

    pseudo_neg = (
        tissue_mask
        & (~color_mask)
        & (~strong_pos)
        & (~weak_pos)
        & (frangi_resp < frangi_low)
        & (lab_a_norm < neg_lab_a_thr)
    )

    ignore = ~tissue_mask

    pseudo_label = np.full(img_rgb.shape[:2], 255, dtype=np.uint8)
    pseudo_label[pseudo_neg] = 0
    pseudo_label[weak_pos] = 1
    pseudo_label[strong_pos] = 1

    supervision_weight = np.zeros(img_rgb.shape[:2], dtype=np.float32)
    supervision_weight[pseudo_neg] = float(neg_weight)
    supervision_weight[weak_pos] = float(weak_pos_weight)
    supervision_weight[strong_pos] = float(strong_pos_weight)

    return {
        "tissue_mask": tissue_mask,
        "color_mask_raw": color_mask_raw,
        "color_mask": color_mask,
        "lab_a_norm": lab_a_norm,
        "frangi_response": frangi_resp,
        "line_mask": line_mask,
        "fibrous_mask": fibrous_mask,
        "strong_pos": strong_pos,
        "weak_pos": weak_pos,
        "pseudo_pos": strong_pos | weak_pos,
        "pseudo_neg": pseudo_neg,
        "ignore": ignore,
        "pseudo_label": pseudo_label,
        "supervision_weight": supervision_weight,
    }


def build_multichannel_input(
    img_rgb: np.ndarray,
    lab_a_norm: np.ndarray,
    frangi_response: np.ndarray,
    frangi_weight: float = 0.01
) -> np.ndarray:
    """
    Build 5-channel model input: RGB (3) + Lab-a norm (1) + Frangi response (1).
    Returns array of shape [5, H, W], dtype float32.
    """
    img_rgb = img_rgb.astype(np.float32) / 255.0

    lab_a_norm = lab_a_norm.astype(np.float32)
    if lab_a_norm.max() > 1.0 or lab_a_norm.min() < 0.0:
        lab_a_norm = (lab_a_norm - lab_a_norm.min()) / (
            lab_a_norm.max() - lab_a_norm.min() + 1e-8
        )

    frangi_response = frangi_response.astype(np.float32)
    frangi_response = normalize01(frangi_response)
    frangi_response = frangi_response * frangi_weight

    x = np.concatenate(
        [
            img_rgb,
            lab_a_norm[..., None],
            frangi_response[..., None]
        ],
        axis=-1
    )

    x = np.transpose(x, (2, 0, 1)).astype(np.float32)
    return x


# ── Tiny U-Net architecture ──────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, base_ch=16):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 2, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


# ── Tiled inference ──────────────────────────────────────────────────────

def _grid_coords(width: int, height: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    xs = list(range(0, max(1, width - patch_size + 1), stride))
    ys = list(range(0, max(1, height - patch_size + 1), stride))
    if len(xs) == 0:
        xs = [0]
    if len(ys) == 0:
        ys = [0]
    if xs[-1] != max(0, width - patch_size):
        xs.append(max(0, width - patch_size))
    if ys[-1] != max(0, height - patch_size):
        ys.append(max(0, height - patch_size))
    return [(x, y) for y in ys for x in xs]


def predict_image_probability(model: nn.Module,
                              img_rgb: np.ndarray,
                              tile_size: int = 256,
                              overlap: int = 32,
                              device=None,
                              progress_callback=None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    h, w = img_rgb.shape[:2]
    stride = tile_size - overlap

    prob_sum = np.zeros((h, w), dtype=np.float32)
    count_sum = np.zeros((h, w), dtype=np.float32)

    coords = _grid_coords(w, h, patch_size=tile_size, stride=stride)

    with torch.no_grad():
        total = len(coords)
        for idx, (x, y) in enumerate(coords, start=1):
            patch = img_rgb[y:y + tile_size, x:x + tile_size]
            ph, pw = patch.shape[:2]

            if ph != tile_size or pw != tile_size:
                canvas = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                canvas[:ph, :pw] = patch
                patch = canvas

            pseudo_info = build_pseudo_labels(patch)
            xin = build_multichannel_input(
                patch,
                lab_a_norm=pseudo_info['lab_a_norm'],
                frangi_response=pseudo_info['frangi_response']
            )
            xt = torch.from_numpy(xin).unsqueeze(0).float().to(device)
            probs = torch.sigmoid(model(xt))[0, 0].cpu().numpy()
            probs = probs[:ph, :pw]

            prob_sum[y:y + ph, x:x + pw] += probs
            count_sum[y:y + ph, x:x + pw] += 1.0

            if progress_callback is not None:
                progress_callback(idx, total, x=int(x), y=int(y))

    return prob_sum / np.maximum(count_sum, 1e-8)


# ── Public API ───────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device=None) -> TinyUNet:
    """
    Load a trained TinyUNet checkpoint.
    The checkpoint is expected to be a dict with a "model" key containing
    the state dict (matches the format saved by the training pipeline).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(in_channels=5, out_channels=1, base_ch=16)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    return model


def run_inference(model, img_rgb: np.ndarray, device=None, threshold: float = 0.5, progress_callback=None):
    """
    Run U-Net inference on an RGB image and return fibrosis prediction.

    Returns dict with keys:
        prob_map           [H, W] float32 probability map
        fibrosis_mask      [H, W] bool mask of pixels above threshold within tissue
        fibrosis_fraction  float, fibrosis_mask.sum() / tissue_mask.sum()
        tissue_mask        [H, W] bool tissue mask
    """
    prob_map = predict_image_probability(
        model, img_rgb, tile_size=256, overlap=32, device=device, progress_callback=progress_callback
    )
    tissue_mask = get_tissue_mask(img_rgb)
    fibrosis_mask = (prob_map >= threshold) & tissue_mask
    tissue_pixel_count = int(tissue_mask.sum())
    if tissue_pixel_count > 0:
        fibrosis_fraction = float(fibrosis_mask.sum()) / float(tissue_pixel_count)
    else:
        fibrosis_fraction = 0.0
    return {
        "prob_map": prob_map,
        "fibrosis_mask": fibrosis_mask,
        "fibrosis_fraction": fibrosis_fraction,
        "tissue_mask": tissue_mask,
    }
