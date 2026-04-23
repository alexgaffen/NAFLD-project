
# https://stackoverflow.com/questions/68500861/visual-studio-code-does-not-render-ipywidgets-correctly

# python -m pip install -U --force pip ->uninstalls pip and force reinstalls
# The path can also be read from a config file, etc.

# Home and lab path happen to be the same
OPENSLIDE_PATH = r'C:\\Program Files\\openslide-bin-4.0.0.11-windows-x64\\bin'


import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        from openslide import OpenSlide
else:
    from openslide import OpenSlide

# setup
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans
from skfuzzy.cluster import cmeans_predict
from glob import glob
# from openslide import OpenSlide
from pprint import pprint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img,img_to_array
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import tensorflow as tf

#from tqdm import tqdm
import gc
import cv2
from PIL import Image
import ipywidgets as widgets
from IPython.display import display
# %matplotlib inline


# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
psr_results = []

# Cache for deconvolution data so rethresholding is instant.
# Keyed by cache_key (string) -> { red_stain, tissue_mask, img_shape }
_deconv_cache = {}

# Point to the 'models' subfolder
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

data_path = './data'
'''
test_path = os.path.join(folder_path, 'OneDrive_1_5-8-2024.zip')
extracted_dir_path = '/content/data/'
!unzip "{test_path}" -d "{extracted_dir_path}"
'''
vgg_model=VGG16(weights='imagenet',include_top=False)

with open(os.path.join(model_save_path,'pca_model.pkl'), 'rb') as file:
    pca_model = pickle.load(file)

with open(os.path.join(model_save_path, 'final_merged_params.pkl'), 'rb') as file:
    model_params = pickle.load(file)

centroids = model_params['centroids']
best_m = model_params.get('best_m', 1.15)  # K-Means-initialized FCM with strict fuzziness
cluster_map = model_params.get('cluster_map', None)  # e.g. {0:0, 1:1, 2:2, 3:3, 4:3}
#8f7972 -> 143,121,114
# 0.560,0.474,0.447

stain_matrix = np.array([[0.148, 0.722, 0.618],
                        [0.462, 0.602, 0.651],
                        [0.187, 0.523, 0.831]])

# stain_matrix = np.array([[0.39, 0.39, 0.39],  
#                          [0.560, 0.474, 0.447],
#                          [0.29, 0.33, 0.29]])

# stain_matrix = np.array([[0.3, 0.3, 0.3],  
#                          [0.560, 0.474, 0.447],
#                          [0.3, 0.3, 0.3]])

#  [[1. 0. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]]

cluster_lookup_table = {
    0: 'Category A: None',
    1: 'Category C: Bridging',
    2: 'Category D: Cirrosis',
    3: 'Category B: Perisinusoidal/Portal',
}

membership_column_names = ['None', 'Bridging', 'Cirrosis', 'Perisinusoidal']


#Helper functions
def extract_features(pil_img):
    img = pil_img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = vgg_model.predict(x)
    predictions = predictions.flatten()
    return predictions

def _find_threshold_by_descent(red_stain, tissue_mask, num_steps=100, jump_factor=2.0):
    """
    Descending threshold sweep with jump detection.

    Treats the red-stain channel as a topographical map. Starts at the
    ceiling (max red value) and steps the threshold downward. At each
    step, measures the extent (% tissue pixels selected). When the
    rate of change in extent jumps significantly (> jump_factor times
    the running average rate), we've crossed from true collagen into
    brown parenchyma — so we snap back to the threshold just before
    the jump.

    Uses a pre-sorted array + searchsorted so each step is O(log n)
    instead of O(n), giving O(n log n) total (dominated by the sort).

    Parameters
    ----------
    red_stain   : 2D array of red-stain deconvolution values
    tissue_mask : boolean 2D array (True = tissue pixel)
    num_steps   : how many threshold levels to test between max and min
    jump_factor : a jump is detected when the rate exceeds this many
                  times the running average rate

    Returns
    -------
    float : the chosen threshold
    """
    tissue_vals = red_stain[tissue_mask]
    if tissue_vals.size < 200:
        return 2.4  # fallback

    ceiling = float(np.percentile(tissue_vals, 99.5))  # ignore extreme outliers
    floor   = float(np.percentile(tissue_vals, 5))      # don't go below 5th pctl
    if ceiling - floor < 0.05:
        return 2.4

    # Sort once — extent lookups become O(log n) via searchsorted
    sorted_vals = np.sort(tissue_vals)
    n = sorted_vals.size
    step_size = (ceiling - floor) / num_steps

    prev_extent = 0.0
    rates = []          # extent increase per step
    thresholds = []     # threshold at each step

    # Sweep from ceiling downward
    for i in range(num_steps + 1):
        t = ceiling - i * step_size
        # O(log n) instead of O(n)
        idx = np.searchsorted(sorted_vals, t, side='right')
        extent = (n - idx) / n * 100.0
        rate = extent - prev_extent

        thresholds.append(t)
        rates.append(rate)
        prev_extent = extent

    # Find the jump: where rate suddenly spikes relative to running avg
    # Skip the first few steps (often noisy at the very top)
    min_steps = 5
    best_thresh = floor  # default to floor if no jump found

    for i in range(min_steps, len(rates)):
        # Running average of rates so far (excluding current)
        avg_rate = np.mean(rates[min_steps:i]) if i > min_steps else rates[i]
        if avg_rate < 1e-6:
            continue

        if rates[i] > jump_factor * avg_rate and rates[i] > 0.5:
            # Jump detected — use the threshold from one step before
            best_thresh = thresholds[max(i - 1, 0)]
            print(f"[Threshold] Jump at step {i}: rate={rates[i]:.2f}% vs avg={avg_rate:.2f}% -> thresh={best_thresh:.3f}")
            break
    else:
        # No clear jump found — use the point where rate is highest
        # (the biggest single-step increase = boundary between red & brown)
        peak_idx = np.argmax(rates[min_steps:]) + min_steps
        best_thresh = thresholds[max(peak_idx - 1, 0)]
        print(f"[Threshold] No jump, using peak-rate at step {peak_idx}: thresh={best_thresh:.3f}")

    print(f"[Threshold] Sweep: ceiling={ceiling:.2f}  floor={floor:.2f}  chosen={best_thresh:.3f}")
    return best_thresh


def fibrosis_filter(window, stain_matrix=None):
    # Always use the adaptive descending-threshold method.
    return _fibrosis_filter_original(window, stain_matrix)


def _fibrosis_filter_simple(window, stain_matrix=None):
    """Simple fixed-threshold method (kept for reference / fallback)."""
    _temp_stain_matrix = np.array([[0.39, 0.39, 0.39],
                                   [0.560, 0.474, 0.447],
                                   [0.29, 0.33, 0.29]])

    img_array = np.array(window)
    img_float = img_array.astype(np.float32) / 255.0
    stains = np.dot(-np.log(img_float + np.finfo(float).eps), _temp_stain_matrix.T)
    red_stain = stains[:, :, 0]

    mask = (red_stain > 0.9) & (img_array.sum(axis=-1) > 50)

    binary_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    binary_mask[mask] = 255

    total_selected_pixels = int(np.sum(binary_mask == 255))
    tissue_mask = img_array.sum(axis=-1) > 50
    tissue_pixel_count = int(np.sum(tissue_mask))
    if tissue_pixel_count > 0:
        selected_ratio = (total_selected_pixels / tissue_pixel_count) * 100
    else:
        selected_ratio = 0.0

    result_image = np.zeros_like(img_array)
    result_image[binary_mask == 255] = [255, 255, 255]
    result_image_pil = Image.fromarray(result_image)

    thresh = 0.9
    return result_image_pil, total_selected_pixels, selected_ratio, tissue_pixel_count, thresh


def _fibrosis_filter_original(window, stain_matrix=None):
    """ORIGINAL METHOD — commented out temporarily, kept for easy restore."""
    if stain_matrix is None:
        stain_matrix = globals()['stain_matrix']

    img_array = np.array(window)

    # ── Brightness normalization ───────────────────────────────
    tissue_px = img_array[(img_array.sum(axis=-1) > 50) & (np.mean(img_array, axis=2) < 240)]
    if tissue_px.size > 0:
        current_median = float(np.median(tissue_px))
        if current_median > 10:
            target_median = 160.0
            scale = target_median / current_median
            img_norm = np.clip(img_array.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            print(f"[Brightness] median {current_median:.0f} -> scaled by {scale:.2f}")
        else:
            img_norm = img_array
    else:
        img_norm = img_array

    img_float = img_norm.astype(np.float32) / 255.0
    stains = np.dot(-np.log(img_float + np.finfo(float).eps), stain_matrix.T)
    red_stain = stains[:, :, 0]

    # Tissue mask MUST be derived from the ORIGINAL image so that
    # white/near-white background is reliably excluded regardless of
    # the brightness-normalisation scale factor.  When a slide is
    # already bright (median > 160) the normalisation scales pixels
    # DOWN, which used to push 255-white background below the 220
    # cut-off and incorrectly count it as tissue.
    tissue_mask = (img_array.sum(axis=-1) > 50) & (np.mean(img_array, axis=2) < 220)

    thresh = _find_threshold_by_descent(red_stain, tissue_mask)

    mask = (red_stain > thresh) & tissue_mask
    mask = mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    total_selected_pixels = np.sum(mask == 255)
    tissue_pixel_count = int(np.sum(tissue_mask))
    if tissue_pixel_count > 0:
        selected_ratio = (total_selected_pixels / tissue_pixel_count) * 100
    else:
        selected_ratio = 0.0

    result_image = np.zeros_like(img_array)
    result_image[mask == 255] = [255, 255, 255]

    result_image_pil = Image.fromarray(result_image)

    return result_image_pil, total_selected_pixels, selected_ratio, tissue_pixel_count, thresh


def cache_deconv_data(cache_key, red_stain, tissue_mask, img_shape, auto_threshold=None):
    """Store deconvolution intermediates so rethreshold is instant."""
    _deconv_cache.clear()  # only keep one cached image at a time
    _deconv_cache[cache_key] = {
        'red_stain': red_stain,
        'tissue_mask': tissue_mask,
        'img_shape': img_shape,
        'auto_threshold': auto_threshold,
    }


def rethreshold(cache_key, new_threshold):
    """Apply a new global threshold to cached deconvolution data.
    Resets any per-pixel area deltas. Returns (mask_pil, total_pixels, ratio, tissue_count) or None."""
    entry = _deconv_cache.get(cache_key)
    if entry is None:
        return None

    red_stain = entry['red_stain']
    tissue_mask = entry['tissue_mask']
    h, w = red_stain.shape

    raw = (red_stain > new_threshold) & tissue_mask
    entry['raw_mask'] = raw.copy()
    entry['has_local_edits'] = False
    entry['current_threshold'] = new_threshold
    entry['threshold_delta'] = np.zeros((h, w), dtype=np.float32)
    entry['undo_stack'] = []
    entry['_last_edit_region'] = None

    mask = raw.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    total_selected = int(np.sum(mask == 255))
    tissue_count = int(np.sum(tissue_mask))
    ratio = (total_selected / tissue_count * 100) if tissue_count > 0 else 0.0

    result_image = np.zeros((h, w, 3), dtype=np.uint8)
    result_image[mask == 255] = [255, 255, 255]
    mask_pil = Image.fromarray(result_image)

    return mask_pil, total_selected, ratio, tissue_count


def _regions_similar(r1, r2, tol=5):
    """Check if two pixel regions are close enough to be the same magnifier area."""
    return (abs(r1[0] - r2[0]) <= tol and abs(r1[1] - r2[1]) <= tol and
            abs(r1[2] - r2[2]) <= tol and abs(r1[3] - r2[3]) <= tol)


def _recompute_mask(entry):
    """Recompute fibrosis mask from current threshold_delta. Returns (mask_pil, total, ratio, tissue_count)."""
    red_stain = entry['red_stain']
    tissue_mask = entry['tissue_mask']
    h, w = red_stain.shape
    base_thresh = entry.get('current_threshold', entry.get('auto_threshold', 2.4))
    td = entry.get('threshold_delta')
    if td is not None and np.any(td != 0):
        effective = base_thresh + td
    else:
        effective = base_thresh
    raw = (red_stain > effective) & tissue_mask
    entry['raw_mask'] = raw
    mask = raw.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    total_selected = int(np.sum(mask == 255))
    tissue_count = int(np.sum(tissue_mask))
    ratio = (total_selected / tissue_count * 100) if tissue_count > 0 else 0.0
    result_image = np.zeros((h, w, 3), dtype=np.uint8)
    result_image[mask == 255] = [255, 255, 255]
    mask_pil = Image.fromarray(result_image)
    return mask_pil, total_selected, ratio, tissue_count


def rethreshold_area(cache_key, delta, x1, y1, x2, y2):
    """Apply a relative threshold delta to a specific normalised region (0-1 coords).
    Each pixel accumulates its own offset; clamped at +/-1.5.
    Returns (mask_pil, total_pixels, ratio, tissue_count) or None."""
    entry = _deconv_cache.get(cache_key)
    if entry is None:
        return None

    red_stain = entry['red_stain']
    tissue_mask = entry['tissue_mask']
    h, w = red_stain.shape

    if 'threshold_delta' not in entry:
        entry['threshold_delta'] = np.zeros((h, w), dtype=np.float32)
    if 'undo_stack' not in entry:
        entry['undo_stack'] = []

    base_thresh = entry.get('current_threshold', entry.get('auto_threshold', 2.4))

    px1 = max(0, int(x1 * w))
    py1 = max(0, int(y1 * h))
    px2 = min(w, int(x2 * w))
    py2 = min(h, int(y2 * h))

    if px2 <= px1 or py2 <= py1:
        return None

    # Push undo snapshot when editing a new region
    current_region = (px1, py1, px2, py2)
    last_region = entry.get('_last_edit_region')
    if last_region is None or not _regions_similar(last_region, current_region):
        entry['undo_stack'].append(entry['threshold_delta'].copy())
        entry['_last_edit_region'] = current_region

    MAX_DELTA = 1.5
    region = entry['threshold_delta'][py1:py2, px1:px2]
    region += delta
    np.clip(region, -MAX_DELTA, MAX_DELTA, out=region)

    entry['has_local_edits'] = True

    return _recompute_mask(entry)


def reset_area(cache_key, x1, y1, x2, y2):
    """Reset threshold delta to zero in a specific normalised region.
    Pushes current state to undo stack."""
    entry = _deconv_cache.get(cache_key)
    if entry is None:
        return None

    h, w = entry['red_stain'].shape

    if 'threshold_delta' not in entry:
        entry['threshold_delta'] = np.zeros((h, w), dtype=np.float32)
    if 'undo_stack' not in entry:
        entry['undo_stack'] = []

    px1 = max(0, int(x1 * w))
    py1 = max(0, int(y1 * h))
    px2 = min(w, int(x2 * w))
    py2 = min(h, int(y2 * h))

    if px2 <= px1 or py2 <= py1:
        return None

    entry['undo_stack'].append(entry['threshold_delta'].copy())
    entry['_last_edit_region'] = None
    entry['threshold_delta'][py1:py2, px1:px2] = 0.0
    entry['has_local_edits'] = bool(np.any(entry['threshold_delta'] != 0))

    return _recompute_mask(entry)


def undo_area(cache_key):
    """Undo the last area modification by restoring from the undo stack."""
    entry = _deconv_cache.get(cache_key)
    if entry is None:
        return None

    undo_stack = entry.get('undo_stack', [])
    if not undo_stack:
        return None

    entry['threshold_delta'] = undo_stack.pop()
    entry['_last_edit_region'] = None
    entry['has_local_edits'] = bool(np.any(entry['threshold_delta'] != 0))

    return _recompute_mask(entry)


def get_delta_map(cache_key):
    """Return a small base64 PNG showing modified regions in green, untouched in white."""
    entry = _deconv_cache.get(cache_key)
    if entry is None:
        return None
    td = entry.get('threshold_delta')
    if td is None or not np.any(td != 0):
        return None
    h, w = td.shape
    modified = (td != 0).astype(np.uint8)
    max_side = 100
    scale = max_side / max(h, w)
    map_h = max(1, int(h * scale))
    map_w = max(1, int(w * scale))
    modified_small = cv2.resize(modified, (map_w, map_h), interpolation=cv2.INTER_NEAREST)
    img = np.full((map_h, map_w, 4), [255, 255, 255, 255], dtype=np.uint8)
    img[modified_small > 0] = [78, 205, 196, 255]
    pil_img = Image.fromarray(img, 'RGBA')
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def get_excluded_mask(cache_key, max_side=2048):
    """Return a base64 RGBA PNG that paints excluded (non-tissue) pixels
    in green and leaves tissue pixels transparent.

    Excluded pixels are exactly the ones removed from the extent
    denominator (too dark or too light to be tissue), so the overlay
    matches the percentage calculation pixel-for-pixel.
    """
    entry = _deconv_cache.get(cache_key)
    if entry is None:
        return None
    tissue_mask = entry['tissue_mask']
    h, w = tissue_mask.shape

    # Downsample for fast transmission while staying nearest-neighbour
    # so the boundary stays crisp.
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        target_h = max(1, int(h * scale))
        target_w = max(1, int(w * scale))
        excluded = (~tissue_mask).astype(np.uint8)
        excluded = cv2.resize(excluded, (target_w, target_h),
                              interpolation=cv2.INTER_NEAREST)
    else:
        target_h, target_w = h, w
        excluded = (~tissue_mask).astype(np.uint8)

    img = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    # Bright green, semi-transparent so the original image shows through.
    img[excluded > 0] = [76, 217, 100, 140]
    pil_img = Image.fromarray(img, 'RGBA')
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _normalised_region_to_pixels(entry, x1, y1, x2, y2):
    """Convert (x1,y1,x2,y2) normalised 0-1 coords to integer pixel
    bounds inside the cached image. Returns None when invalid."""
    h, w = entry['red_stain'].shape
    px1 = max(0, int(x1 * w))
    py1 = max(0, int(y1 * h))
    px2 = min(w, int(x2 * w))
    py2 = min(h, int(y2 * h))
    if px2 <= px1 or py2 <= py1:
        return None
    return px1, py1, px2, py2


def analyze_area(cache_key, x1, y1, x2, y2):
    """Compute fibrosis extent for a specific normalised region using the
    same cached red_stain / tissue_mask / threshold as the global mask.

    Honours any threshold deltas applied by the user. Returns
    {ratio, fibrosis_pixels, tissue_pixels} or None if no cache.
    """
    entry = _deconv_cache.get(cache_key)
    if entry is None:
        return None
    bounds = _normalised_region_to_pixels(entry, x1, y1, x2, y2)
    if bounds is None:
        return None
    px1, py1, px2, py2 = bounds

    red_stain = entry['red_stain'][py1:py2, px1:px2]
    tissue_mask = entry['tissue_mask'][py1:py2, px1:px2]
    base_thresh = entry.get('current_threshold', entry.get('auto_threshold', 2.4))
    td = entry.get('threshold_delta')
    if td is not None and np.any(td != 0):
        effective = base_thresh + td[py1:py2, px1:px2]
    else:
        effective = base_thresh

    raw = (red_stain > effective) & tissue_mask
    mask = raw.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    fibrosis_pixels = int(np.sum(mask == 255))
    tissue_pixels = int(np.sum(tissue_mask))
    ratio = (fibrosis_pixels / tissue_pixels * 100) if tissue_pixels > 0 else 0.0
    return {
        'fibrosis_ratio': float(ratio),
        'fibrosis_pixels': fibrosis_pixels,
        'tissue_pixels': tissue_pixels,
    }


def classify_area(cache_key, x1, y1, x2, y2):
    """Run VGG16 + PCA + FCM on a single cropped region of the current
    fibrosis mask. Fast (single-tile path). Returns membership scores
    or None when the cache is missing / region is empty / region is
    almost entirely background.
    """
    entry = _deconv_cache.get(cache_key)
    if entry is None:
        return None
    bounds = _normalised_region_to_pixels(entry, x1, y1, x2, y2)
    if bounds is None:
        return None
    px1, py1, px2, py2 = bounds

    # Same exclusion criterion as the global classifier
    tissue_frac = float(np.mean(entry['tissue_mask'][py1:py2, px1:px2]))
    if tissue_frac < 0.10:
        return {'status': 'background', 'tissue_frac': tissue_frac}

    mask_pil, _, _, _ = _recompute_mask(entry)
    mask_np = np.array(mask_pil)
    tile = mask_np[py1:py2, px1:px2]
    if tile.shape[0] < 32 or tile.shape[1] < 32:
        return {'status': 'too_small'}

    tile_pil = Image.fromarray(tile)
    try:
        features = extract_features(tile_pil)
        reduc_features = pca_model.transform([features])
        u, _, _, _, _, _ = cmeans_predict(reduc_features.T, centroids,
                                          m=best_m, error=0.1, maxiter=100)
        if cluster_map is not None:
            n_groups = max(cluster_map.values()) + 1
            u_merged = np.zeros((n_groups, u.shape[1]))
            for src, dst in cluster_map.items():
                u_merged[dst] += u[src]
            u = u_merged
        label_idx = int(np.argmax(u, axis=0))
        label_text = cluster_lookup_table.get(label_idx, f"Unknown Cluster {label_idx}")
        scores = {membership_column_names[i]: float(u[i][0]) for i in range(4)}
        return {
            'status': 'success',
            'cluster_label': label_text,
            'membership_scores': scores,
            'tissue_frac': tissue_frac,
        }
    finally:
        del tile_pil


def classify_from_mask(cache_key, patch_size=512, top_n=5, progress_callback=None):
    """
    Run VGG16 + PCA + FCM classification on the current refined B&W mask.

    For large (patchable) images the mask is tiled into patch_size squares
    and each tissue tile is classified independently.  The final score is
    the average of the *top_n* worst (most-fibrotic) tiles — i.e. the tiles
    whose membership leans most toward higher-stage disease.

    For small images the whole mask is classified in a single pass.

    The tissue_mask from the original analysis (left image) is used to
    determine which tiles to skip — carrying the exclusion principle from
    the original PSR staining over to the mask classification.

    progress_callback(current_tile, total_tiles, tissue_count,
                      grid_rows, grid_cols, tile_row, tile_col, is_tissue)
    is called after every tile so the frontend can show live scanning.

    Returns a dict with cluster_label, membership_scores, patch_count,
    worst_patches (list of {row,col,py,px} dicts for the top-N tiles),
    or None when no cached data is available.
    """
    entry = _deconv_cache.get(cache_key)
    if entry is None:
        return None

    # Rebuild the current mask from cache (honours all threshold edits)
    mask_pil, _, _, _ = _recompute_mask(entry)
    mask_np = np.array(mask_pil)
    h, w = mask_np.shape[:2]

    # Inclusion criterion = same one used by analyze_single_file_patched
    # and by the extent denominator: the cached tissue_mask (left panel).
    # No more relying on a separate cached patched grid — we recompute
    # tile inclusion on the fly from the source of truth.
    cached_tissue_mask = entry['tissue_mask']
    TISSUE_FRAC_THRESHOLD = 0.25

    # Determine if we should tile (same heuristic as patched analysis)
    is_large = max(h, w) > 2048

    if not is_large:
        # ── Single-image classification: feed the B&W mask to VGG16 ──
        features = extract_features(mask_pil)
        reduc_features = pca_model.transform([features])
        u, _, _, _, _, _ = cmeans_predict(reduc_features.T, centroids, m=best_m, error=0.1, maxiter=100)
        if cluster_map is not None:
            n_groups = max(cluster_map.values()) + 1
            u_merged = np.zeros((n_groups, u.shape[1]))
            for src, dst in cluster_map.items():
                u_merged[dst] += u[src]
            u = u_merged
        label_idx = int(np.argmax(u, axis=0))
        label_text = cluster_lookup_table.get(label_idx, f"Unknown Cluster {label_idx}")
        scores = {membership_column_names[i]: float(u[i][0]) for i in range(4)}
        return {
            "status": "success",
            "cluster_label": label_text,
            "membership_scores": scores,
            "patch_count": 1,
            **scores,
        }

    # ── Patch-based classification for large images ──
    rows_range = list(range(0, h, patch_size))
    cols_range = list(range(0, w, patch_size))
    num_rows = len(rows_range)
    num_cols = len(cols_range)
    total_tiles = num_rows * num_cols
    current_tile = 0

    patch_memberships = []
    patch_coords = []  # (row_idx, col_idx, py, px) for each classified tile

    for ri, py in enumerate(rows_range):
        for ci, px in enumerate(cols_range):
            current_tile += 1
            tile = mask_np[py:py + patch_size, px:px + patch_size]
            th, tw = tile.shape[:2]
            if th < 32 or tw < 32:
                continue

            # ── Exclusion: same criterion as analyze_single_file_patched.
            # Compute on-the-fly from the cached tissue_mask (left panel)
            # rather than relying on a separately cached tile grid.
            tile_frac = float(np.mean(
                cached_tissue_mask[py:py + th, px:px + tw]
            ))
            if tile_frac < TISSUE_FRAC_THRESHOLD:
                if progress_callback:
                    progress_callback(current_tile, total_tiles, len(patch_memberships),
                                      grid_rows=num_rows, grid_cols=num_cols,
                                      tile_row=ri, tile_col=ci, is_tissue=False)
                continue

            tile_pil = Image.fromarray(tile)
            try:
                features = extract_features(tile_pil)
                reduc_features = pca_model.transform([features])
                u, _, _, _, _, _ = cmeans_predict(reduc_features.T, centroids, m=best_m, error=0.1, maxiter=100)
                if cluster_map is not None:
                    n_groups = max(cluster_map.values()) + 1
                    u_merged = np.zeros((n_groups, u.shape[1]))
                    for src, dst in cluster_map.items():
                        u_merged[dst] += u[src]
                    u = u_merged
                patch_memberships.append([float(u[i][0]) for i in range(4)])
                patch_coords.append({'row': ri, 'col': ci, 'py': py, 'px': px})
            except Exception as e:
                print(f"[classify_from_mask] Patch ({py},{px}) failed: {e}")
            finally:
                del tile_pil

            if len(patch_memberships) % 10 == 0:
                gc.collect()

            if progress_callback:
                progress_callback(current_tile, total_tiles, len(patch_memberships),
                                  grid_rows=num_rows, grid_cols=num_cols,
                                  tile_row=ri, tile_col=ci, is_tissue=True)

    if not patch_memberships:
        return {"status": "error", "message": "No classifiable tiles found in mask"}

    # Rank tiles by severity: higher-stage membership (Bridging + Cirrosis)
    # indices: 1=Bridging, 2=Cirrosis
    severity = [m[1] + m[2] for m in patch_memberships]
    ranked_indices = sorted(range(len(severity)), key=lambda i: severity[i], reverse=True)
    top_indices = ranked_indices[:min(top_n, len(ranked_indices))]
    top_memberships = [patch_memberships[i] for i in top_indices]
    avg_memberships = np.mean(top_memberships, axis=0)

    label_idx = int(np.argmax(avg_memberships))
    label_text = cluster_lookup_table.get(label_idx, f"Unknown Cluster {label_idx}")
    scores = {membership_column_names[i]: float(avg_memberships[i]) for i in range(4)}

    # Collect worst patch coordinates for frontend overlay
    worst_patches = [patch_coords[i] for i in top_indices]

    print(f"[classify_from_mask] {len(patch_memberships)} tiles classified, "
          f"top-{min(top_n, len(patch_memberships))} worst averaged → {label_text}")

    return {
        "status": "success",
        "cluster_label": label_text,
        "membership_scores": scores,
        "patch_count": len(patch_memberships),
        "top_n_used": min(top_n, len(patch_memberships)),
        "worst_patches": worst_patches,
        "grid_rows": num_rows,
        "grid_cols": num_cols,
        "img_h": h,
        "img_w": w,
        "patch_size": patch_size,
        **scores,
    }


def predict_cluster_pil(pil_image, already_filtered=False, original_pil=None):
    """
    Classify an image using VGG16 + PCA + Fuzzy C-Means.

    VGG16 needs the original RGB tissue image for meaningful features.
    The B&W mask from fibrosis_filter is only used for the fibrosis percentage.

    Parameters
    ----------
    pil_image   : PIL image (filtered mask if already_filtered=True, else raw RGB)
    already_filtered : if True, skip fibrosis_filter (percentage will be None)
    original_pil : the original RGB image for VGG16 feature extraction.
                   If None, falls back to pil_image (legacy behaviour).
    """
    image = np.array(pil_image)

    if already_filtered:
        percentage = None
    else:
        _, _, percentage, _, _ = fibrosis_filter(image)

    # VGG16 must see the original RGB tissue, NOT the B&W mask
    feature_source = original_pil if original_pil is not None else pil_image
    features = extract_features(feature_source)
    reduc_features = pca_model.transform([features])
    u, _, _, _, _, _ = cmeans_predict(reduc_features.T, centroids, m=best_m, error=0.1, maxiter=100)
    # Merge memberships according to cluster_map (e.g. fold cluster 4 into 3)
    if cluster_map is not None:
        n_groups = max(cluster_map.values()) + 1
        u_merged = np.zeros((n_groups, u.shape[1]))
        for src, dst in cluster_map.items():
            u_merged[dst] += u[src]
        u = u_merged
    cluster_label = np.argmax(u, axis=0)
    return u, cluster_label, percentage


# Single Image
def update_filter_slide_legacy(change):
    slide_path = os.path.join(image_folder, image_dropdown.value)
    slide = OpenSlide(slide_path)

    level = level_slider.value
    slide_path = os.path.join(image_folder, image_dropdown.value)
    slide = OpenSlide(slide_path)

    level = level_slider.value
    x = x_slider.value
    y = y_slider.value
    window_width = window_width_slider.value
    window_height = window_height_slider.value

    region = slide.read_region((x, y), level, (window_width, window_height))
    region = region.convert("RGB")  # Convert to RGB

    result_image_pil, total_selected_pixels, selected_ratio, _, _ = fibrosis_filter(region)
    u, cluster_label, _ = predict_cluster_pil(result_image_pil, already_filtered=True, original_pil=region)

    with output:
        output.clear_output(wait=True)
        plt.figure(figsize=(30, 20))
        plt.subplot(1, 2, 1)
        plt.imshow(region)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(result_image_pil)
        plt.title('Filtered Image')
        plt.axis('off')
        plt.show()

        print("Total Selected Pixels:", total_selected_pixels)
        print("Selected Ratio:", selected_ratio, "%")
        print("Cluster Label:", cluster_lookup_table[cluster_label.item()])

def update_filter_slide(image_folder,file,level_slider,x_slider,y_slider,window_width_slider,window_height_slider):
    slide_path = os.path.join(image_folder, file)
    slide = OpenSlide(slide_path)

    level = level_slider.value
    x = x_slider.value
    y = y_slider.value
    window_width = window_width_slider.value
    window_height = window_height_slider.value

    region = slide.read_region((x, y), level, (window_width, window_height))
    region = region.convert("RGB")  # Convert to RGB

    result_image_pil, total_selected_pixels, selected_ratio, _, _ = fibrosis_filter(region)
    u, cluster_label, _ = predict_cluster_pil(result_image_pil, already_filtered=True, original_pil=region)

    output = widgets.Output()

    with output:
        # output.clear_output(wait=True)
        # plt.figure(figsize=(30, 20))
        # plt.subplot(1, 2, 1)
        # plt.imshow(region)
        # plt.title('Original Image')
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.imshow(result_image_pil)
        # plt.title('Filtered Image')
        # plt.axis('off')
        # plt.show()

        print("\n \n Total Selected Pixels:", total_selected_pixels)
        print("Selected Ratio:", selected_ratio, "%")
        print(f"Cluster Label: {cluster_lookup_table[cluster_label.item()]} \n \n")
    return [total_selected_pixels,selected_ratio,cluster_lookup_table[cluster_label.item()]]



# For non svs files
def update_filter_image(change):
    selection_work_around = 'KM-UUO1-3- %6.610.jpg'
    # image_dropdown.value
    original_image_path = os.path.join(image_folder, selection_work_around)
    print(f'\n {original_image_path} \n')
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    output = widgets.Output()
    original_pil = Image.fromarray(original_image)
    result_image_pil, total_selected_pixels, selected_ratio, _, _ = fibrosis_filter(original_image)
    u, cluster_label, _ = predict_cluster_pil(result_image_pil, already_filtered=True, original_pil=original_pil)
    with output:
        output.clear_output(wait=True)
        plt.figure(figsize=(30, 20))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(result_image_pil, cmap='gray')
        plt.title('Filtered Image')
        plt.axis('off')
        plt.show()

        # print("Total Selected Pixels:", total_selected_pixels)
        # print("Selected Ratio:", selected_ratio, "%")
        # print("Cluster Label:", cluster_lookup_table[cluster_label.item()])

        # psr_results.append([selected_ratio,image_dropdown.value])
        print("\n \n Total Selected Pixels:", total_selected_pixels)
        print("Selected Ratio:", selected_ratio, "%")
        print(f"Cluster Label: {cluster_lookup_table[cluster_label.item()]} \n \n")


def predict_cluster(image_path, stain_matrix=None):
    if stain_matrix is None:
        stain_matrix = globals()['stain_matrix']

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img, pxl, percentage, _, _ = fibrosis_filter(image, stain_matrix)
    # Pass the original RGB image to VGG16, not the B&W mask
    original_pil = Image.fromarray(image)
    features = extract_features(original_pil)
    reduc_features = pca_model.transform([features])
    u, _, _, _, _, _ = cmeans_predict(reduc_features.T, centroids, m=best_m, error=0.1, maxiter=100)
    if cluster_map is not None:
        n_groups = max(cluster_map.values()) + 1
        u_merged = np.zeros((n_groups, u.shape[1]))
        for src, dst in cluster_map.items():
            u_merged[dst] += u[src]
        u = u_merged
    cluster_label = np.argmax(u, axis=0)
    return u, cluster_label, percentage

def process_images_in_folder(folder_path, predict_cluster_func):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', 'tif')):
            image_path = os.path.join(folder_path, filename)
            u, cluster_label, percentage = predict_cluster_func(image_path)
            # results.append([filename, percentage, cluster_lookup_table[cluster_label.item()]])
            results.append([filename, percentage, cluster_lookup_table[cluster_label.item()],u[0][0],u[1][0],u[2][0],u[3][0]])
    # results.append(["104.5.tif",0.0,"Category A: None",0.8119407227,0.05121940722792983,0.098033421630126128,0.03098885441976844])
    # results.append(["15A.tif",3.3791232638888893,"Category C: Bridging",0.058330845853,0.00048083308458533,0.84677525217225086,0.0921591135627087])
    # results.append(["16E.tif",1.472829861111111,"Category C: Bridging",0.03421226000923161,0.10285556764417932,0.7324887669357235,0.1304434054108655])
    # results.append(["61.7.tif",0.0,"Category A: None",0.86083308458,0.0418876622792983,4.121940722792983e-31,0.0998885441976844])

    df = pd.DataFrame(results, columns=['image_name', 'percentage', 'cluster_label','None','Bridging','Cirrosis','Perisinusoidal'])
    # df = pd.DataFrame(results, columns=['image_name', 'percentage', 'cluster_label')
    return df

def process_images_in_folder_svs(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('svs'):

            # image_folder = './data/kidney'
            # image_files = [f for f in os.listdir(image_folder) if f.endswith('.svs')]
            slide = OpenSlide(os.path.join(folder_path, filename))

            level_0_width, level_0_height = slide.level_dimensions[0]
            level_slider = widgets.IntSlider(value=0, min=0, max=len(slide.level_dimensions)-1, description='Level:')
            x_slider = widgets.IntSlider(value=18777, min=0, max=level_0_width, description='X:')
            y_slider = widgets.IntSlider(value=10453, min=0, max=level_0_height, description='Y:')
            window_width_slider = widgets.IntSlider(value=1920, min=1, max=level_0_width, description='Window Width:')
            window_height_slider = widgets.IntSlider(value=1200, min=1, max=level_0_height, description='Window Height:')
            # image_dropdown.observe(update_filter_slide, 'value')
            level_slider.observe(update_filter_slide, 'value')
            x_slider.observe(update_filter_slide, 'value')
            y_slider.observe(update_filter_slide, 'value')
            window_width_slider.observe(update_filter_slide, 'value')
            window_height_slider.observe(update_filter_slide, 'value')

            u, cluster_label, percentage = update_filter_slide(image_folder=folder_path, file=filename,level_slider=level_slider, x_slider=x_slider,y_slider=y_slider,
                                                               window_width_slider=window_width_slider,window_height_slider=window_height_slider)
            results.append([filename, percentage, cluster_label])

    df = pd.DataFrame(results, columns=['image_name', 'percentage', 'cluster_label'])
    return df

def process_all_images(folder_path):
    # Process all image types

    df_non_svs = process_images_in_folder(folder_path, predict_cluster)

    df_svs = process_images_in_folder_svs(folder_path)
    
    return pd.concat([df_non_svs, df_svs], ignore_index=True)

import base64
from io import BytesIO


def pil_to_b64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def open_image_for_processing(file_path, max_side=1024):
    if file_path.lower().endswith('.svs'):
        slide = OpenSlide(file_path)
        try:
            img_pil = slide.get_thumbnail((max_side, max_side)).convert("RGB")
        finally:
            slide.close()
    else:
        img_pil = Image.open(file_path).convert("RGB")
        img_pil.thumbnail((max_side, max_side), Image.LANCZOS)
    return img_pil


# ── Patch helpers ──────────────────────────────────────────────────

def _is_tissue_patch(patch_np):
    """Return True unless the patch is almost entirely white (≥95% blank)."""
    gray = np.mean(patch_np.astype(np.float32), axis=2)
    return (np.sum(gray > 220) / gray.size) < 0.95


def _select_slide_level(slide, max_dim=8192):
    """Pick the highest-resolution level whose longest side fits in *max_dim*."""
    for level, (w, h) in enumerate(slide.level_dimensions):
        if max(w, h) <= max_dim:
            return level
    return len(slide.level_dimensions) - 1


def analyze_single_file_patched(file_path, patch_size=512, max_processing_dim=8192, progress_callback=None):
    """
    Patch-based analysis for SVS (and large TIF) files.

    1. Opens the image at a resolution level ≤ max_processing_dim.
    2. Tiles it into patch_size × patch_size squares.
    3. Skips background patches (blank white areas).
    4. Runs fibrosis_filter on every tissue patch, stitches the mask.
    5. Aggregates fibrosis ratio from *all* tissue patches.
    6. Classification still uses a thumbnail so the trained PCA/FCM
       model (which was trained on whole-tissue views) gets compatible input.

    progress_callback(current_patch, total_candidates, tissue_count) is called
    after every tile so the frontend can show live updates.
    """
    try:
        import traceback as _tb
        is_svs = file_path.lower().endswith('.svs')

        # ── Load the image at a workable resolution ──
        if is_svs:
            slide = OpenSlide(file_path)
            level = _select_slide_level(slide, max_processing_dim)
            level_w, level_h = slide.level_dimensions[level]
            full_pil = slide.read_region((0, 0), level, (level_w, level_h)).convert("RGB")
            full_image = np.array(full_pil)
            del full_pil
            slide.close()
            try:
                os.remove(file_path)
                print(f"[GC] Deleted uploaded file: {file_path}")
            except Exception:
                pass
            print(f"[Patch] SVS opened at level {level} → {level_w}×{level_h}")
        else:
            try:
                # Try OpenSlide first (works for TIFs with pyramids)
                slide = OpenSlide(file_path)
                level = _select_slide_level(slide, max_processing_dim)
                level_w, level_h = slide.level_dimensions[level]
                full_pil = slide.read_region((0, 0), level, (level_w, level_h)).convert("RGB")
                full_image = np.array(full_pil)
                del full_pil
                slide.close()
                try:
                    os.remove(file_path)
                    print(f"[GC] Deleted uploaded file: {file_path}")
                except Exception:
                    pass
                print(f"[Patch] Large TIFF opened via OpenSlide at level {level}")
            except Exception:
                # Fallback to Pillow if OpenSlide fails (e.g. standard TIFF without pyramid)
                print("[Patch] Fallback to Pillow for image load")
                img = Image.open(file_path).convert("RGB")
                if max(img.size) > max_processing_dim:
                    img.thumbnail((max_processing_dim, max_processing_dim), Image.LANCZOS)
                full_image = np.array(img)
                del img
                try:
                    os.remove(file_path)
                    print(f"[GC] Deleted uploaded file: {file_path}")
                except Exception:
                    pass

        h, w = full_image.shape[:2]

        # Send an early preview from the SAME image being tiled so the
        # frontend overlay aligns perfectly with the displayed image.
        if progress_callback:
            preview_pil = Image.fromarray(full_image)
            preview_pil.thumbnail((2048, 2048), Image.LANCZOS)
            progress_callback(0, 1, 0,
                              analysis_preview=f"data:image/jpeg;base64,{pil_to_b64(preview_pil)}")
            del preview_pil

        # ── Run fibrosis_filter on the FULL image (no seams) ──
        full_mask_pil, total_fibrosis_pixels, overall_ratio, total_tissue_pixels, auto_thresh = fibrosis_filter(full_image)
        full_mask = np.array(full_mask_pil)
        del full_mask_pil

        # Cache deconvolution data for interactive rethresholding
        _sm = globals()['stain_matrix']
        _tpx = full_image[(full_image.sum(axis=-1) > 50) & (np.mean(full_image, axis=2) < 240)]
        if _tpx.size > 0:
            _cm = float(np.median(_tpx))
            if _cm > 10:
                _img_n = np.clip(full_image.astype(np.float32) * (160.0 / _cm), 0, 255).astype(np.uint8)
            else:
                _img_n = full_image
        else:
            _img_n = full_image
        _img_f = _img_n.astype(np.float32) / 255.0
        _stains = np.dot(-np.log(_img_f + np.finfo(float).eps), _sm.T)
        cache_deconv_data(
            os.path.basename(file_path),
            _stains[:, :, 0],
            (full_image.sum(axis=-1) > 50) & (np.mean(full_image, axis=2) < 220),
            full_image.shape,
            auto_threshold=auto_thresh,
        )
        del _tpx, _img_n, _img_f, _stains

        print(f"[Patch] Full-image fibrosis filter: {overall_ratio:.2f}%  tissue_px={total_tissue_pixels}")

        # Per-tile work was removed — VGG16/FCM only happens during the
        # Diagnose step (classify_from_mask). The left-panel scanning grid
        # served as a visual progress indicator for that dead pass and is
        # therefore no longer streamed.  patch_count is reported as the
        # number of tiles that meet the >=25% tissue inclusion criterion.
        patch_size_count = 512
        rows = list(range(0, h, patch_size_count))
        cols = list(range(0, w, patch_size_count))
        cached_entry = _deconv_cache.get(os.path.basename(file_path))
        cached_tissue_mask = cached_entry['tissue_mask'] if cached_entry else None
        patch_count = 0
        if cached_tissue_mask is not None:
            for py in rows:
                for px in cols:
                    ph = min(patch_size_count, h - py)
                    pw_actual = min(patch_size_count, w - px)
                    if ph < 32 or pw_actual < 32:
                        continue
                    tile_frac = float(np.mean(
                        cached_tissue_mask[py:py + ph, px:px + pw_actual]
                    ))
                    if tile_frac >= 0.25:
                        patch_count += 1

        print(f"[Patch] {patch_count} tissue tiles flagged — fibrosis {overall_ratio:.2f}%")

        # ── Resize for frontend display, then free large arrays ──
        display_orig = Image.fromarray(full_image)
        display_orig.thumbnail((8192, 8192), Image.LANCZOS)

        display_mask = Image.fromarray(full_mask)
        display_mask.thumbnail((8192, 8192), Image.LANCZOS)

        del full_image, full_mask
        gc.collect()
        print("[GC] Released full_image and full_mask")

        return {
            "status": "success",
            "original_image": f"data:image/jpeg;base64,{pil_to_b64(display_orig)}",
            "filtered_image": f"data:image/jpeg;base64,{pil_to_b64(display_mask)}",
            "fibrosis_ratio": float(overall_ratio),
            "threshold": float(auto_thresh),
            "patch_count": patch_count,
        }
    except Exception as e:
        print(f"Error in patched analysis: {e}")
        import traceback; traceback.print_exc()
        return {"status": "error", "message": str(e)}


def preview_single_file(file_path, preview_size=1536):
    """
    Fast preview path: generates original + fibrosis mask on downsampled image.
    """
    try:
        img_pil = open_image_for_processing(file_path, max_side=preview_size)
        img_np = np.array(img_pil)
        filtered_pil, total_pixels, ratio, _, _ = fibrosis_filter(img_np)

        return {
            "status": "success",
            "is_preview": True,
            "original_image": f"data:image/jpeg;base64,{pil_to_b64(img_pil)}",
            "filtered_image": f"data:image/jpeg;base64,{pil_to_b64(filtered_pil)}",
            "fibrosis_ratio": float(ratio),
        }
    except Exception as e:
        print(f"Error in preview: {e}")
        return {"status": "error", "message": str(e)}

def analyze_single_file(file_path):
    """
    Called by main.py. Routes to patch-based analysis for SVS / large TIF,
    or the fast thumbnail path for small images (JPG, PNG, small TIF).
    """
    is_svs = file_path.lower().endswith('.svs')
    is_large_tif = (
        file_path.lower().endswith(('.tif', '.tiff'))
        and os.path.getsize(file_path) > 50 * 1024 * 1024  # > 50 MB
    )

    if is_svs or is_large_tif:
        print(f"[analyze] Using PATCH-BASED pipeline for {os.path.basename(file_path)}")
        return analyze_single_file_patched(file_path)

    # ── Fast thumbnail path for small images ──
    print(f"[analyze] Using THUMBNAIL pipeline for {os.path.basename(file_path)}")
    try:
        # Guard against oversized flat images that would exhaust RAM
        MAX_FLAT_BYTES = 50 * 1024 * 1024  # 50 MB
        if os.path.getsize(file_path) > MAX_FLAT_BYTES:
            return {
                "status": "error",
                "message": "File too large for standard image analysis (max 50 MB). Use SVS or TIF format for whole-slide images."
            }

        # 1. Open the image
        img_pil = open_image_for_processing(file_path, max_side=2048)
        try:
            os.remove(file_path)
            print(f"[GC] Deleted uploaded file: {file_path}")
        except Exception:
            pass

        # 2. Run your existing Fibrosis Filter
        # Note: We convert PIL -> Numpy for your filter
        img_np = np.array(img_pil)
        filtered_pil, total_pixels, ratio, _, auto_thresh = fibrosis_filter(img_np)

        # Cache deconvolution data for rethresholding
        cache_key = os.path.basename(file_path)
        # Recompute the intermediates for caching (lightweight)
        _sm = globals()['stain_matrix']
        tissue_px = img_np[(img_np.sum(axis=-1) > 50) & (np.mean(img_np, axis=2) < 240)]
        if tissue_px.size > 0:
            cm = float(np.median(tissue_px))
            if cm > 10:
                img_n = np.clip(img_np.astype(np.float32) * (160.0 / cm), 0, 255).astype(np.uint8)
            else:
                img_n = img_np
        else:
            img_n = img_np
        img_f = img_n.astype(np.float32) / 255.0
        stains = np.dot(-np.log(img_f + np.finfo(float).eps), _sm.T)
        cache_deconv_data(cache_key, stains[:, :, 0],
                         (img_np.sum(axis=-1) > 50) & (np.mean(img_np, axis=2) < 220),
                         img_np.shape, auto_threshold=auto_thresh)

        # 3. Run your existing Cluster Prediction
        # Pass original RGB to VGG16 for meaningful features
        u, cluster_label, _ = predict_cluster_pil(filtered_pil, already_filtered=True, original_pil=img_pil)
        
        # Handle the label lookup safely
        label_idx = cluster_label.item()
        label_text = cluster_lookup_table.get(label_idx, f"Unknown Cluster {label_idx}")
        membership_scores = {
            membership_column_names[0]: float(u[0][0]),
            membership_column_names[1]: float(u[1][0]),
            membership_column_names[2]: float(u[2][0]),
            membership_column_names[3]: float(u[3][0]),
        }

        return {
            "status": "success",
            "original_image": f"data:image/jpeg;base64,{pil_to_b64(img_pil)}",
            "filtered_image": f"data:image/jpeg;base64,{pil_to_b64(filtered_pil)}",
            "fibrosis_ratio": float(ratio),
            "cluster_label": label_text,
            "membership_scores": membership_scores,
            "threshold": float(auto_thresh),
            "None": membership_scores["None"],
            "Perisinusoidal": membership_scores["Perisinusoidal"],
            "Bridging": membership_scores["Bridging"],
            "Cirrosis": membership_scores["Cirrosis"],
        }
    except Exception as e:
        print(f"Error in analysis: {e}")
        return {"status": "error", "message": str(e)}

# ------------
image_folder = './data/kidney'
# image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
# image_dropdown = widgets.Dropdown(options=image_files, description='Select Image:')

# output = widgets.Output()
# image_dropdown.observe(update_filter_image, 'value')
# update_filter_image(None)
# display(widgets.VBox([image_dropdown, output]))

# --------
# C:\Projects\nafld_back\data\Alpha-SMA-IHC Quantification-%Area
# './data/PSR'
# './data/kidney'

# ------------------------------

# image_folder = '/content/data/'
# image_files = [f for f in os.listdir(image_folder) if f.endswith('.svs')]
# image_dropdown = widgets.Dropdown(options=image_files, description='Select Image:')
# slide = OpenSlide(os.path.join(image_folder, image_files[0]))
# level_0_width, level_0_height = slide.level_dimensions[0]


# level_slider = widgets.IntSlider(value=0, min=0, max=len(slide.level_dimensions)-1, description='Level:')
# x_slider = widgets.IntSlider(value=18777, min=0, max=level_0_width, description='X:')
# y_slider = widgets.IntSlider(value=10453, min=0, max=level_0_height, description='Y:')
# window_width_slider = widgets.IntSlider(value=1920, min=1, max=level_0_width, description='Window Width:')
# window_height_slider = widgets.IntSlider(value=1200, min=1, max=level_0_height, description='Window Height:')

# output = widgets.Output()

# image_dropdown.observe(update_filter_slide, 'value')
# level_slider.observe(update_filter_slide, 'value')
# x_slider.observe(update_filter_slide, 'value')
# y_slider.observe(update_filter_slide, 'value')
# window_width_slider.observe(update_filter_slide, 'value')
# window_height_slider.observe(update_filter_slide, 'value')


# update_filter_slide(None)

# display(widgets.VBox([image_dropdown, level_slider, x_slider, y_slider, window_width_slider, window_height_slider, output]))
# -------------------------------------------------

# display(widgets.VBox([image_dropdown, level_slider, x_slider, y_slider, window_width_slider, window_height_slider, output]))


# #save as csv
# data_path = './data/kidney'
# df = process_images_in_folder(data_path, predict_cluster)
# df.head()
# df.to_csv('PSR.csv', index=False)
# # C:\Projects\nafld_back\data\kidney\KF-Sham1-1- %0.687.jpg
# print(predict_cluster('./data/kidney/KF-Sham1-1- %0.687.jpg'))


#save as csv
# data_path = './data/PSR'
# df = process_images_in_folder_svs(data_path)
# df.head()
# df.to_csv('PSR.csv', index=False)
# # C:\Projects\nafld_back\data\kidney\KF-Sham1-1- %0.687.jpg
# print(predict_cluster('./data/kidney/KF-Sham1-1- %0.687.jpg'))
# print(psr_results)

# %([^\.]+)\. REJEX for for finding percentage of file