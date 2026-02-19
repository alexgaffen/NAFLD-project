
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

with open(os.path.join(model_save_path, 'model_params.pkl'), 'rb') as file:
    model_params = pickle.load(file)

centroids = model_params['centroids']

# u = model_params['u']
# u0 = model_params['u0']
# d = model_params['d']
# jm = model_params['jm']
# p = model_params['p']
# fpc = model_params['fpc']
best_m = model_params.get('best_m', 1.5) # Default to 1.5 if missing
#8f7972 -> 143,121,114
# 0.560,0.474,0.447

# stain_matrix = np.array([[0.148, 0.722, 0.618],
#                          [0.462, 0.602, 0.651],
#                          [0.187, 0.523, 0.831]])

stain_matrix = np.array([[0.39, 0.39, 0.39],  
                         [0.560, 0.474, 0.447],
                         [0.29, 0.33, 0.29]])

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

def fibrosis_filter(window, stain_matrix=None):
    if stain_matrix is None:
        stain_matrix = globals()['stain_matrix']

    img_array = np.array(window)
    img_float = img_array.astype(np.float32) / 255.0
    #perform color deconv
    stains = np.dot(-np.log(img_float + np.finfo(float).eps), stain_matrix.T)
    #Select the stain for red regions
    red_stain = stains[:, :, 0]

    print(f"[DEBUG stain_matrix row0]: {stain_matrix[0]}")
    print(f"[DEBUG red_stain]: min={red_stain.min():.3f}  mean={red_stain.mean():.3f}  max={red_stain.max():.3f}")
    print(f"[DEBUG threshold hits]: >1.5 → {(red_stain > 1.5).mean()*100:.1f}%  |  >1.8 → {(red_stain > 1.8).mean()*100:.1f}%  |  >2.0 → {(red_stain > 2.0).mean()*100:.1f}%")

    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    mask = (red_stain > 1.5) & (img_array.sum(axis=-1) > 50)
    mask = mask.astype(np.uint8) * 255
    total_selected_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    selected_ratio = (total_selected_pixels / total_pixels) * 100

    result_image = np.zeros_like(img_array)
    result_image[mask == 255] = [255, 255, 255]

    result_image_pil = Image.fromarray(result_image)

    return result_image_pil, total_selected_pixels, selected_ratio

def predict_cluster_pil(pil_image, already_filtered=False):
    image = np.array(pil_image)

    if already_filtered:
        img = pil_image
        percentage = None
    else:
        img, _, percentage = fibrosis_filter(image)

    features = extract_features(img)
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

    result_image_pil, total_selected_pixels, selected_ratio = fibrosis_filter(region)
    u, cluster_label, _ = predict_cluster_pil(result_image_pil, already_filtered=True)

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

    result_image_pil, total_selected_pixels, selected_ratio = fibrosis_filter(region)
    u, cluster_label, _ = predict_cluster_pil(result_image_pil, already_filtered=True)

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
    result_image_pil, total_selected_pixels, selected_ratio = fibrosis_filter(original_image)
    u, cluster_label, _ = predict_cluster_pil(result_image_pil, already_filtered=True)
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
    img, pxl, percentage = fibrosis_filter(image, stain_matrix)
    features = extract_features(img)
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
    img.save(buffered, format="JPEG")
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


def preview_single_file(file_path, preview_size=768):
    """
    Fast preview path: generates original + fibrosis mask on downsampled image.
    """
    try:
        img_pil = open_image_for_processing(file_path, max_side=preview_size)
        img_np = np.array(img_pil)
        filtered_pil, total_pixels, ratio = fibrosis_filter(img_np)

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
    Called by main.py. Opens the image, runs the AI, returns base64 images to frontend.
    """
    try:
        # 1. Open the image
        img_pil = open_image_for_processing(file_path, max_side=1024)

        # 2. Run your existing Fibrosis Filter
        # Note: We convert PIL -> Numpy for your filter
        img_np = np.array(img_pil)
        # Your fibrosis_filter returns: (result_pil, total_pixels, ratio)
        filtered_pil, total_pixels, ratio = fibrosis_filter(img_np)

        # 3. Run your existing Cluster Prediction
        # We use the filtered image for prediction
        u, cluster_label, _ = predict_cluster_pil(filtered_pil, already_filtered=True)
        
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