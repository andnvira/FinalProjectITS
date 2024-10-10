
import streamlit as st
import os
from PIL import Image
import numpy as np
import cv2
import io
from io import BytesIO
import time
import scipy.ndimage as ndi
import math
import torch
import random
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
import pickle
from sklearn.pipeline import Pipeline
import os
import tensorflow as tf
import pandas as pd

os.environ['PYOPENGL_PLATFORM'] = 'egl'

#---------------------------------------------Preprocessing---------------------------------------
def load_image(img):
    im = Image.open(img)
    return im

def normalize_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_img = gray / 255.0
    return normalized_img

def color_norm(img):
    pixels = np.array(img)
    ar_mean = np.mean(pixels, axis=(0,1)) 
    Ri = pixels[:,:,0]
    Gi = pixels[:,:,1]
    Bi = pixels[:,:,2]
    (H,W) = pixels.shape[:2]

    for x in range(H):
        for y in range(W):
            if ar_mean[0] != 0:
                val = np.min(Ri[x, y] / float(ar_mean[0]) * 124.21850142079624)
                Ri[x, y] = 255 if val > 255 else val
            if ar_mean[1] != 0:
                val = np.min(Gi[x, y] / float(ar_mean[1]) * 61.74248535662327)
                Gi[x, y] = 255 if val > 255 else val
            if ar_mean[2] != 0:
                val = np.min(Bi[x, y] / float(ar_mean[2]) * 15.596426572394947)
                Bi[x, y] = 255 if val > 255 else val
    
    merged = np.dstack((Ri, Gi, Bi))
    return merged

def contrast_enhance(img):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(15, 15))
    Red = img[...,0]
    Green = img[...,1]
    Blue = img[...,2]
    Green_fix = clahe.apply(Green)
    new_img = np.stack([Red, Green_fix, Blue], axis=2)
    return new_img

def img_preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = color_norm(img)
    img = contrast_enhance(img)
    img = cv2.medianBlur(img, 3)
    return img

#------------------------------Feature Extraction-----------------------------------
def extract_glcm_features(image):
    try:
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        levels = 256
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image_uint = (image * 255).astype(np.uint8)

        glcm = greycomatrix(image_uint, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

        features = {
            'contrast': greycoprops(glcm, 'contrast').mean(),
            'dissimilarity': greycoprops(glcm, 'dissimilarity').mean(),
            'homogeneity': greycoprops(glcm, 'homogeneity').mean(),
            'energy': greycoprops(glcm, 'energy').mean(),
            'correlation': greycoprops(glcm, 'correlation').mean(),
            'ASM': greycoprops(glcm, 'ASM').mean(),
            'variance': np.var(image.flatten())
        }
        return features
    except Exception as e:
        print(f"Error extracting GLCM features: {e}")
        st.write(f"Error extracting GLCM features: {e}")
        return None



#------------------------------Load CNN Model (VGG16)-----------------------------------
@st.cache_resource
def load_vgg16_model():
    model_path = r"C:\Users\lenovo\Downloads\TA\VGG16_model.h5"
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model

#------------------------------Classification using CNN (VGG16)-----------------------------------
def classify_with_vgg16(image, model):
    try:
        target_size = (224, 224)
        
        # Preprocess the image
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)
        image_normalized = image_np.astype(np.float32) / 255.0
        image_resized = cv2.resize(image_normalized, target_size)
        image_expanded = np.expand_dims(image_resized, axis=0)
        
        # Perform prediction
        prediction = model.predict(image_expanded)
        
        # Mapping prediction to class labels
        class_labels = ['Normal', 'Mild', 'Moderate', 'Severe']
        predicted_class = class_labels[np.argmax(prediction)]
        
        return predicted_class, prediction
    except Exception as e:
        print(f"Error classifying image with VGG16: {e}")
        st.write(f"Error classifying image with VGG16: {e}")
        return None, None

#------------------------------------------------Load SVM Classification Model------------------------------------------
svm_model_path = r"C:\Users\lenovo\Downloads\TA\svm_model.pkl"
with open(svm_model_path, 'rb') as file:
    svm_model = pickle.load(file)

#------------------------------Classification using SVM-----------------------------------
def classify_with_svm(features, model):
    try:
        features_array = np.array(list(features.values())).reshape(1, -1)
        prediction = model.predict(features_array)
        return prediction
    except Exception as e:
        print(f"Error classifying with SVM: {e}")
        st.write(f"Error classifying with SVM: {e}")
        return None

#------------------------------------------------START GUI HERE------------------------------------------
#----------------------------------------------------COVER--------------------------------------------
def Cover():
    st.title("📙Tugas Akhir")
    st.header("👀 Klasifikasi Multikelas Glaukoma pada Citra Fundus Retina Menggunakan Convolutional Neural Networks (CNN) 👀")
    st.subheader("🌺 Andini Vira Salsabilla - 5023201065 🌺")
    st.markdown("### Dosen Pembimbing 1 : Dr. Tri Arief Sardjono, S.T., M.T\n"
                "### Dosen Pembimbing 2 : Nada Fitrieyatul Hikmah, S.T., M.T")
    
     # Display a predefined image
    image_path = r"C:\Users\lenovo\Downloads\TA\TA DOKUM\CITRA FUNDUS.jpg"  
    st.image(image_path, use_column_width=True)

    st.sidebar.info("Please note the information below:\n"
                    "- Select **🪄PREPROCESSING** page to enhancing image quality\n"
                    "- Select **🔎PREDICTION** page to predict glaucoma using **:green[preprocessed image]** \n"
                    "- Select **📋ABOUT** page to find more information")

def Preprocessing():
    if "im" not in st.session_state:
        st.session_state["im"] = None
    if "img_resize" not in st.session_state:
        st.session_state["img_resize"] = None
    if "img_color" not in st.session_state:
        st.session_state["img_color"] = None
    if "img_clahe" not in st.session_state:
        st.session_state["img_clahe"] = None
    if "img_medfilt" not in st.session_state:
        st.session_state["img_medfilt"] = None

    st.sidebar.markdown('Press **:red[RESET]** button to reset this page')
    reset = st.sidebar.button("Reset", use_container_width=True)
    if reset:
        st.session_state["im"] = None
        st.session_state["img_resize"] = None
        st.session_state["img_color"] = None
        st.session_state["img_clahe"] = None
        st.session_state["img_medfilt"] = None
        st.sidebar.success("This page has been reset.")

    st.markdown("# Image Preprocessing")
    img = st.sidebar.file_uploader(label="🖥️ Upload an image", type=['jpg', 'png'])

    if img is not None:
        st.session_state["im"] = load_image(img)
        width, height = st.session_state["im"].size
        st.sidebar.write("original image size : {} x {}".format(width, height))
        st.markdown("### Original Image")
        st.image(st.session_state["im"], width=250)

    col1, col2 = st.columns(2)

    st.sidebar.title("Preprocessing Steps")
    crop = st.sidebar.button("Start Preprocessing", use_container_width=True)
    if crop and st.session_state["im"] is not None:
        img_np = np.array(st.session_state["im"])
        st.session_state["img_resize"] = cv2.resize(img_np, (256, 256))
        st.session_state["img_color"] = color_norm(st.session_state["img_resize"])
        st.session_state["img_clahe"] = contrast_enhance(st.session_state["img_color"])
        st.session_state["img_medfilt"] = cv2.medianBlur(st.session_state["img_clahe"], 3)

    if st.session_state["img_resize"] is not None:
        w, h = st.session_state["img_resize"].shape[:2]
        st.sidebar.write("preprocessed image size : {} x {}".format(w, h))
        with col2:
            st.markdown("### Resized Image")
            st.image(st.session_state["img_resize"], use_column_width=True)

    if st.session_state["img_color"] is not None:
        with col1:
            st.markdown("### Color Normalized Image")
            st.image(st.session_state["img_color"], use_column_width=True)

    if st.session_state["img_clahe"] is not None:
        with col2:
            st.markdown("### Contrast Enhanced Image")
            st.image(st.session_state["img_clahe"], use_column_width=True)

    if st.session_state["img_medfilt"] is not None:
        with col1:
            st.markdown("### Median Filtered Image")
            st.image(st.session_state["img_medfilt"], use_column_width=True)

    st.sidebar.info("ℹ️ When pressing the button above, a series of preprocessing steps will be carried out\n"
                    "- 1️⃣Resize 📏\n"
                    "- 2️⃣Color Normalization 🎨\n"
                    "- 3️⃣Contrast Enhancement🔧\n"
                    "- 4️⃣Median Filter 🧹")

    if st.session_state["img_medfilt"] is not None:
        im_pil = Image.fromarray(st.session_state["img_medfilt"])
        im_bytes = BytesIO()
        im_pil.save(im_bytes, format='JPEG')
        file_name = img.name.split('.')[0] + '_preproc.jpg'
        st.sidebar.download_button(
            label="🗃️ Download Image",
            data=im_bytes.getvalue(),
            file_name=file_name,
            mime="image/jpeg",
            use_container_width=True)

#---------------------------------------------START PREDICTION HERE-----------------------------------
def Prediksi():
    if "prediction_label_vgg16" not in st.session_state:
        st.session_state["prediction_label_vgg16"] = None

    col1, col2 = st.columns(2)

    if "im" not in st.session_state or st.session_state["im"] is None:
        st.warning("Please upload an image in the 🪄PREPROCESSING page")
        return

    st.markdown("### Preprocessed Image")
    st.image(st.session_state["img_medfilt"], width=250)

    # Handle CNN (VGG16) prediction button
    cnn_predict = st.button("Classify using CNN", use_container_width=True)
    if cnn_predict:
        with st.spinner("Classifying using CNN..."):
            try:
                # Load VGG16 model
                vgg16_model_path = r"C:\Users\lenovo\Downloads\TA\VGG16_model.h5"
                vgg16_model = tf.keras.models.load_model(vgg16_model_path)

                # Classify using VGG16 model
                result, prediction = classify_with_vgg16(st.session_state["im"], vgg16_model)

                if result is not None and prediction is not None:
                    class_labels = ['Normal', 'Mild', 'Moderate', 'Severe']
                    prediction_df = pd.DataFrame(prediction, columns=class_labels)
                    st.bar_chart(prediction_df.T)
                    st.write(f"Predicted Class: {result}")
                    st.session_state["prediction_label_vgg16"] = result
                else:
                    st.session_state["prediction_label_vgg16"] = "Prediction failed (VGG16)"
                    st.write("VGG16 prediction failed. Please check the VGG16 classification steps.")

            except Exception as e:
                st.session_state["prediction_label_vgg16"] = "Prediction error (VGG16)"
                st.write(f"Error predicting with VGG16: {e}")

    # Display VGG16 prediction result
    if st.session_state["prediction_label_vgg16"] is not None:
        with col2:
            st.markdown(f"<h2>VGG16 Prediction Result:</h2><p style='font-size: 30px; color: green;"
                        f"font-weight: bold;'>{st.session_state['prediction_label_vgg16']}</p><br>", unsafe_allow_html=True)

    # Reset button in sidebar
    st.sidebar.markdown('---')
    reset = st.sidebar.button("Reset Prediction", key="reset_prediction", use_container_width=True)
    if reset:
        st.session_state["prediction_label_vgg16"] = None



#-------------------------------------------------START ABOUT HERE--------------------------------------
def About():
    st.title("Klasifikasi Multikelas Glaukoma pada Citra Fundus Retina Menggunakan Convolutional Neural Networks (CNN)")
    st.info("👩‍🎓Andini Vira Salsabilla - 5023201065\n"
            "- 👨‍🏫Dosen Pembimbing 1 : Dr. Tri Arief Sardjono, S.T., M.T\n"
            "- 👩‍🏫Dosen Pembimbing 2 : Nada Fitrieyatul Hikmah, S.T., M.T")

def main():
    step = st.sidebar.selectbox("Select Page: ", ['🏡HOME', '🪄PREPROCESSING', '🔎PREDICTION', '📋ABOUT'])
    if step == '🏡HOME':
        Cover()
    if step == '🪄PREPROCESSING':
        Preprocessing()
    if step == '🔎PREDICTION':
        Prediksi()    
    if step == '📋ABOUT':
        About()

if __name__ == "__main__":
    main()