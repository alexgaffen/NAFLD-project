
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

with open(os.path.join(model_save_path, 'fixed_model_params.pkl'), 'rb') as file:
    model_params = pickle.load(file)

centroids = model_params['centroids']
best_m = model_params.get('best_m', 1.15)  # K-Means-initialized FCM with strict fuzziness
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
    1: 'Category B: Perisinusoidal/Portal',
    2: 'Category C: Bridging',
    3: 'Category D: Cirrosis',
}

membership_column_names = ['None', 'Perisinusoidal', 'Bridging', 'Cirrosis']


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
    if stain_matrix is None:
        stain_matrix = globals()['stain_matrix']

    img_array = np.array(window)

    # ── Brightness normalization ───────────────────────────────
    # Stretch tissue pixel intensities to a consistent range so
    # that brighter/darker exposures produce the same OD values.
    # Target: tissue median ≈ 160  (typical well-lit PSR slide)
    tissue_px = img_array[(img_array.sum(axis=-1) > 50) & (np.mean(img_array, axis=2) < 240)]
    if tissue_px.size > 0:
        current_median = float(np.median(tissue_px))
        if current_median > 10:  # avoid division by near-zero
            target_median = 160.0
            scale = target_median / current_median
            img_norm = np.clip(img_array.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            print(f"[Brightness] median {current_median:.0f} -> scaled by {scale:.2f}")
        else:
            img_norm = img_array
    else:
        img_norm = img_array

    img_float = img_norm.astype(np.float32) / 255.0
    #perform color deconv
    stains = np.dot(-np.log(img_float + np.finfo(float).eps), stain_matrix.T)
    #Select the stain for red regions
    red_stain = stains[:, :, 0]

    # Tissue mask: exclude white background and very dark pixels (on normalized image)
    tissue_mask = (img_norm.sum(axis=-1) > 50) & (np.mean(img_norm, axis=2) < 220)

    # Descending threshold sweep — finds where red collagen ends
    # and brown parenchyma begins
    thresh = _find_threshold_by_descent(red_stain, tissue_mask)

    mask = (red_stain > thresh) & tissue_mask
    mask = mask.astype(np.uint8) * 255

    # ── Morphological opening ──────────────────────────────────
    # Remove isolated small clusters of pixels (noise / faint staining
    # of portal structures).  A 3×3 elliptical kernel removes objects
    # smaller than ~3 px across, matching QuPath's behaviour of only
    # counting contiguous collagen regions.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    total_selected_pixels = np.sum(mask == 255)
    # Use tissue pixels (not total pixels) as denominator so white
    # background doesn't dilute the extent calculation.
    tissue_pixel_count = int(np.sum(tissue_mask))
    if tissue_pixel_count > 0:
        selected_ratio = (total_selected_pixels / tissue_pixel_count) * 100
    else:
        selected_ratio = 0.0

    result_image = np.zeros_like(img_array)
    result_image[mask == 255] = [255, 255, 255]

    result_image_pil = Image.fromarray(result_image)

    return result_image_pil, total_selected_pixels, selected_ratio, tissue_pixel_count

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
        _, _, percentage, _ = fibrosis_filter(image)

    # VGG16 must see the original RGB tissue, NOT the B&W mask
    feature_source = original_pil if original_pil is not None else pil_image
    features = extract_features(feature_source)
    reduc_features = pca_model.transform([features])
    u, _, _, _, _, _ = cmeans_predict(reduc_features.T, centroids, m=best_m, error=0.1, maxiter=100)
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

    result_image_pil, total_selected_pixels, selected_ratio, _ = fibrosis_filter(region)
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

    result_image_pil, total_selected_pixels, selected_ratio, _ = fibrosis_filter(region)
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
    result_image_pil, total_selected_pixels, selected_ratio, _ = fibrosis_filter(original_image)
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
    img, pxl, percentage, _ = fibrosis_filter(image, stain_matrix)
    # Pass the original RGB image to VGG16, not the B&W mask
    original_pil = Image.fromarray(image)
    features = extract_features(original_pil)
    reduc_features = pca_model.transform([features])
    u, _, _, _, _, _ = cmeans_predict(reduc_features.T, centroids, m=best_m, error=0.1, maxiter=100)
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

    df = pd.DataFrame(results, columns=['image_name', 'percentage', 'cluster_label','None','Perisinusoidal','Bridging','Cirrosis'])
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


def _select_slide_level(slide, max_dim=4096):
    """Pick the highest-resolution level whose longest side fits in *max_dim*."""
    for level, (w, h) in enumerate(slide.level_dimensions):
        if max(w, h) <= max_dim:
            return level
    return len(slide.level_dimensions) - 1


def analyze_single_file_patched(file_path, patch_size=512, max_processing_dim=4096, progress_callback=None):
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
        full_mask_pil, total_fibrosis_pixels, overall_ratio, total_tissue_pixels = fibrosis_filter(full_image)
        full_mask = np.array(full_mask_pil)
        del full_mask_pil
        print(f"[Patch] Full-image fibrosis filter: {overall_ratio:.2f}%  tissue_px={total_tissue_pixels}")

        patch_count = 0
        patch_memberships = []      # list of membership arrays (4,)

        # Count total candidate tiles for progress reporting
        rows = list(range(0, h, patch_size))
        cols = list(range(0, w, patch_size))
        num_rows = len(rows)
        num_cols = len(cols)
        total_tiles = num_rows * num_cols
        current_tile = 0

        # ── Tile for VGG16 classification — every tile gets classified ──
        for ri, py in enumerate(rows):
            for ci, px in enumerate(cols):
                current_tile += 1
                patch = full_image[py:py + patch_size, px:px + patch_size]
                ph, pw_actual = patch.shape[:2]

                if ph < 32 or pw_actual < 32:
                    continue

                # Extract this tile's mask for VGG16
                tile_mask = full_mask[py:py + ph, px:px + pw_actual]
                tile_mask_pil = Image.fromarray(tile_mask)
                patch_pil = Image.fromarray(patch)
                try:
                    patch_u, patch_cl, _ = predict_cluster_pil(
                        tile_mask_pil, already_filtered=True, original_pil=patch_pil
                    )
                    patch_memberships.append([float(patch_u[i][0]) for i in range(4)])
                except Exception as e:
                    print(f"[Patch] Classification failed on patch ({py},{px}): {e}")
                finally:
                    del tile_mask_pil, patch_pil

                patch_count += 1

                # Free unreferenced tensors periodically
                if patch_count % 10 == 0:
                    gc.collect()

                if progress_callback:
                    progress_callback(current_tile, total_tiles, patch_count,
                                      grid_rows=num_rows, grid_cols=num_cols,
                                      tile_row=ri, tile_col=ci, is_tissue=True)

        print(f"[Patch] {patch_count} tissue patches classified — fibrosis {overall_ratio:.2f}%")

        # ── Average fuzzy memberships across all patches ──
        if patch_memberships:
            avg_memberships = np.mean(patch_memberships, axis=0)
        else:
            avg_memberships = np.zeros(4)

        # ── Resize for frontend display, then free large arrays ──
        display_orig = Image.fromarray(full_image)
        display_orig.thumbnail((2048, 2048), Image.LANCZOS)

        display_mask = Image.fromarray(full_mask)
        display_mask.thumbnail((2048, 2048), Image.LANCZOS)

        del full_image, full_mask
        gc.collect()
        print("[GC] Released full_image and full_mask")

        # Derive label from highest average membership
        final_label = int(np.argmax(avg_memberships))
        label_text = cluster_lookup_table.get(final_label, f"Unknown Cluster {final_label}")
        membership_scores = {
            membership_column_names[i]: float(avg_memberships[i]) for i in range(4)
        }

        return {
            "status": "success",
            "original_image": f"data:image/jpeg;base64,{pil_to_b64(display_orig)}",
            "filtered_image": f"data:image/jpeg;base64,{pil_to_b64(display_mask)}",
            "fibrosis_ratio": float(overall_ratio),
            "cluster_label": label_text,
            "membership_scores": membership_scores,
            "patch_count": patch_count,
            "None": membership_scores["None"],
            "Perisinusoidal": membership_scores["Perisinusoidal"],
            "Bridging": membership_scores["Bridging"],
            "Cirrosis": membership_scores["Cirrosis"],
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
        filtered_pil, total_pixels, ratio, _ = fibrosis_filter(img_np)

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
        # Your fibrosis_filter returns: (result_pil, total_pixels, ratio, tissue_pixel_count)
        filtered_pil, total_pixels, ratio, _ = fibrosis_filter(img_np)

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