import streamlit as st
import rasterio
import numpy as np
import scipy.io
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

# Set up page configurations
st.set_page_config(page_title="PUTRA ForHelps Pro", layout="wide")
st.title("🌲 PUTRA Forest Health & Fungal Diversity Predictor")

# --- PART A: Thesis Regression Constants (y = mx + c) ---
fungal_models = {
    "Shannon": {"m": 0.0632, "c": 0.584},
    "Richness": {"m": 0.0510, "c": 0.320},
    "Simpson": {"m": 0.0712, "c": 0.410},
    "Evenness": {"m": 0.0425, "c": 0.150}
}

# --- PART B: Load MATLAB Neural Network Weights ---
@st.cache_resource
def load_nn_weights():
    try:
        return scipy.io.loadmat('model_230_weights.mat')
    except Exception as e:
        return None

weights = load_nn_weights()

# --- PART C: File Upload & UI Handling ---
uploaded_file = st.file_uploader("Upload a UAV TIFF Image", type=["tif", "tiff"])

if uploaded_file:
    with rasterio.open(uploaded_file) as src:
        data = src.read()
        num_bands = data.shape[0]

        # Contrast-stretched display logic for rendering standard preview
        rgb_display = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
        display_bands = min(num_bands, 3)
        for i in range(display_bands):
            band = data[i, :, :]
            p2, p98 = np.percentile(band, (2, 98))
            if p98 - p2 > 0:
                stretched = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                rgb_display[:, :, i] = stretched.astype(np.uint8)
            else:
                rgb_display[:, :, i] = 0
        
        display_img = Image.fromarray(rgb_display)

    # Split-screen user interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Analysis Area")
        st.write("Click anywhere on the canopy map to analyze localized metrics:")
        coords = streamlit_image_coordinates(display_img, use_column_width=True)

    with col2:
        st.subheader("Analysis Results")
        if coords:
            # Scale UI click coordinates back to raw array data coordinates
            img_width, img_height = display_img.size
            data_h, data_w = data.shape[1], data.shape[2]
            
            x = int(coords["x"] * (data_w / img_width))
            y = int(coords["y"] * (data_h / img_height))
            x, y = min(max(x, 0), data_w - 1), min(max(y, 0), data_h - 1)
            
            st.write(f"**Selected Coordinates:** Pixel (X: {x}, Y: {y})")

            # --- SMART BAND DETECTION & INDEX CALCULATION ---
            if num_bands >= 5:
                # Full Multispectral Extraction (PlanetScope)
                b_blue, b_green, b_red, b_re, b_nir = [float(data[i, y, x]) for i in range(5)]
                
                ndvi  = (b_nir - b_red) / (b_nir + b_red + 1e-8)
                gndvi = (b_nir - b_green) / (b_nir + b_green + 1e-8)
                ndre  = (b_nir - b_re) / (b_nir + b_re + 1e-8)
                cire  = (b_nir / (b_re + 1e-8)) - 1.0
                savi  = ((b_nir - b_red) / (b_nir + b_red + 0.5 + 1e-8)) * 1.5
                msavi = 0.5 * (2 * b_nir + 1 - np.sqrt(max(0, (2 * b_nir + 1)**2 - 8 * (b_nir - b_red))))
                
                vari_denom = (b_green + b_red - b_blue + 1e-8)
                vari  = (b_green - b_red) / vari_denom if vari_denom != 0 else 0.0

                primary_index_val = ndre # Recommended by thesis for 5+ bands
                st.info(f"Multispectral Primary (NDRE): **{round(primary_index_val, 4)}**")
                has_full_bands = True

            else:
                # Fallback for RGB (3-band) or 4-band images
                b_blue = float(data[0, y, x]) if num_bands > 0 else 0.0
                b_green = float(data[1, y, x]) if num_bands > 1 else 0.0
                b_red = float(data[2, y, x]) if num_bands > 2 else 0.0
                
                vari_denom = (b_green + b_red - b_blue + 1e-8)
                vari  = (b_green - b_red) / vari_denom if vari_denom != 0 else 0.0
                
                primary_index_val = vari
                st.info(f"RGB Fallback (VARI): **{round(primary_index_val, 4)}**")
                has_full_bands = False

            # --- PART D: Fungal Diversity Dashboard (Restoring the big numbers) ---
            st.markdown("#### Fungal Community Profiling Estimates")
            metric_cols = st.columns(2)
            
            for i, (metric, params) in enumerate(fungal_models.items()):
                predicted_val = (primary_index_val - params["c"]) / params["m"]
                metric_cols[i % 2].metric(metric, round(predicted_val, 3))
            
            st.write("---")

            # --- PART E: Model 2.30 Forward Pass Neural Network Engine ---
            if has_full_bands:
                st.markdown("#### Structural Forest Health Zonal Classification")
                if weights is not None:
                    features = np.array([[ndvi, gndvi, ndre, cire, savi, msavi, vari]], dtype=np.float32)
                    W1, b1 = weights['W1'], weights['b1'].flatten()
                    W2, b2 = weights['W2'], weights['b2'].flatten()
                    mu, sigma = weights.get('mu'), weights.get('sigma')
                    
                    if mu is not None and mu.size > 0:
                        features = (features - mu.flatten()) / sigma.flatten()
                    
                    z1 = np.dot(features, W1.T) + b1
                    a1 = np.maximum(0, z1) 
                    z2 = np.dot(a1, W2.T) + b2
                    pred_class_idx = int(np.argmax(z2, axis=1)[0]) + 1 
                    
                    class_mapping = {
                        1: "Class 1 (Disturbed / TRA) — High Urban Fragmentation",
                        2: "Class 2 (Secondary / STF) — Moderate Structural Density",
                        3: "Class 3 (Pristine / TNP) — Primary Climax Canopy Framework"
                    }
                    pred_label = class_mapping.get(pred_class_idx, f"Unidentified Eco-Class ID ({pred_class_idx})")
                    
                    if pred_class_idx == 1:
                        st.error(f"🚨 **Ecosystem State:** {pred_label}")
                    elif pred_class_idx == 2:
                        st.warning(f"⚠️ **Ecosystem State:** {pred_label}")
                    elif pred_class_idx == 3:
                        st.success(f"🌲 **Ecosystem State:** {pred_label}")
                else:
                    st.error("Neural Network evaluation is offline. Missing 'model_230_weights.mat'.")
            else:
                st.warning("⚠️ High-resolution Multispectral imagery (5+ bands) required to unlock the Deep Learning Model. Showing baseline RGB metrics only.")
