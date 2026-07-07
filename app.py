import streamlit as st
import rasterio
import numpy as np
import scipy.io
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PUTRA ForHelps Pro", layout="wide", initial_sidebar_state="expanded")

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
        st.error(f"⚠️ Warning: Could not load 'model_230_weights.mat'. Neural network classification will be unavailable. Error: {e}")
        return None

weights = load_nn_weights()

# --- PART C: Sidebar UI - Ecosystem Context & File Upload ---
st.sidebar.title("🌲 PUTRA ForHelps Pro")
st.sidebar.markdown("### Step 1: Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload UAV Multispectral TIFF", type=["tif", "tiff"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Step 2: Define Ecosystem Context")
site_type = st.sidebar.radio(
    "Select the Site Management Category:",
    ("Urban / Fragmented", "Recreational / Moderate", "Primary Forest / Dense")
)

# Dynamic descriptions to guide the operator
if site_type == "Urban / Fragmented":
    st.sidebar.info("**Description:** Areas with fragmented canopies and exposed understory or soil (e.g., Taman Rimba Alam). \n\n**System Action:** Optimizes using Soil-Adjusted Indices (**SAVI / MSAVI**) to mitigate background noise from pavement and exposed earth.")
elif site_type == "Recreational / Moderate":
    st.sidebar.info("**Description:** Managed zones where vegetation is continuous but not excessively layered (e.g., Sungai Tekala). \n\n**System Action:** Optimizes using standard broadband indices (**NDVI / VARI**) for maximum stability and high-integrity correlation.")
else:
    st.sidebar.info("**Description:** Undisturbed ecosystems with high biomass and multi-storied canopy architecture (e.g., Taman Negara Pahang). \n\n**System Action:** Optimizes using Red-Edge bands (**NDRE / CIRE**) to penetrate dense canopies and prevent spectral saturation.")

st.sidebar.markdown("---")
st.sidebar.markdown("Universiti Putra Malaysia (UPM) | Ag/Bio Engineering")

# --- PART D: Main Processing UI - Spatial Query & Analytics ---
if uploaded_file:
    # 1. Read and process the GeoTIFF
    with rasterio.open(uploaded_file) as src:
        data = src.read()
        num_bands = data.shape[0]

        # True Color RGB preview logic based on PlanetScope sensor type
        rgb_display = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
        
        if num_bands == 8:
            vis_bands = [5, 3, 1]  # PS 8-Band: Red (5), Green (3), Blue (1)
        elif num_bands == 4:
            vis_bands = [2, 1, 0]  # PS 4-Band: Red (2), Green (1), Blue (0)
        else:
            vis_bands = [0, 1, 2][:num_bands] # Fallback for unknown sensor
            
        for idx, b_idx in enumerate(vis_bands):
            if b_idx < num_bands:
                band = data[b_idx, :, :]
                p2, p98 = np.percentile(band, (2, 98))
                if p98 - p2 > 0:
                    stretched = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                    rgb_display[:, :, idx] = stretched.astype(np.uint8)
        
        display_img = Image.fromarray(rgb_display)

    # 2. Split-screen user interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Step 3: Spatial Query")
        st.write("Click anywhere on the canopy map to extract localized biophysical metrics:")
        # Added key="map_click" to maintain coordinate state across Streamlit UI toggles
        coords = streamlit_image_coordinates(display_img, use_column_width=True, key="map_click")

    with col2:
        st.subheader("Analysis Results")
        if coords:
            # Scale UI click coordinates back to raw array data coordinates
            img_width, img_height = display_img.size
            data_h, data_w = data.shape[1], data.shape[2]
            
            x = int(coords["x"] * (data_w / img_width))
            y = int(coords["y"] * (data_h / img_height))
            x, y = min(max(x, 0), data_w - 1), min(max(y, 0), data_h - 1)
            
            st.markdown(f"**Targeted Pixel:** (X: {x}, Y: {y})")
            
            # --- Advanced Band Extraction for PlanetScope ---
            if num_bands == 8:
                b_blue  = np.clip((float(data[1, y, x]) / 10000.0), 0.0, 1.0)
                b_green = np.clip((float(data[3, y, x]) / 10000.0), 0.0, 1.0)
                b_red   = np.clip((float(data[5, y, x]) / 10000.0), 0.0, 1.0)
                b_re    = np.clip((float(data[6, y, x]) / 10000.0), 0.0, 1.0)
                b_nir   = np.clip((float(data[7, y, x]) / 10000.0), 0.0, 1.0)
            elif num_bands == 4:
                b_blue  = np.clip((float(data[0, y, x]) / 10000.0), 0.0, 1.0)
                b_green = np.clip((float(data[1, y, x]) / 10000.0), 0.0, 1.0)
                b_red   = np.clip((float(data[2, y, x]) / 10000.0), 0.0, 1.0)
                b_re    = 0.0  # Red-Edge unavailable in 4-band
                b_nir   = np.clip((float(data[3, y, x]) / 10000.0), 0.0, 1.0)
            else:
                b_blue  = np.clip((float(data[0, y, x]) / 10000.0), 0.0, 1.0) if num_bands > 0 else 0.0
                b_green = np.clip((float(data[1, y, x]) / 10000.0), 0.0, 1.0) if num_bands > 1 else 0.0
                b_red   = np.clip((float(data[2, y, x]) / 10000.0), 0.0, 1.0) if num_bands > 2 else 0.0
                b_re    = np.clip((float(data[3, y, x]) / 10000.0), 0.0, 1.0) if num_bands > 3 else 0.0
                b_nir   = np.clip((float(data[4, y, x]) / 10000.0), 0.0, 1.0) if num_bands > 4 else 0.0

            # Calculate all 7 indices (Required for Neural Network Input Matrix)
            ndvi  = (b_nir - b_red) / (b_nir + b_red + 1e-8)
            gndvi = (b_nir - b_green) / (b_nir + b_green + 1e-8)
            ndre  = (b_nir - b_re) / (b_nir + b_re + 1e-8)
            cire  = (b_nir / (b_re + 1e-8)) - 1.0
            savi  = ((b_nir - b_red) / (b_nir + b_red + 0.5 + 1e-8)) * 1.5
            msavi = 0.5 * (2 * b_nir + 1 - np.sqrt((2 * b_nir + 1)**2 - 8 * (b_nir - b_red)))
            
            vari_denom = (b_green + b_red - b_blue + 1e-8)
            vari  = (b_green - b_red) / vari_denom if vari_denom != 0 else 0.0

            # --- Step 4: User Toggles Between Suitable Indices ---
            st.markdown("#### Select Localized Index")
            
            if site_type == "Urban / Fragmented":
                index_choice = st.radio("Toggle Recommended Urban Index:", ["SAVI", "MSAVI"], horizontal=True)
                primary_vi_name = index_choice
                primary_vi_val = savi if index_choice == "SAVI" else msavi
                
            elif site_type == "Recreational / Moderate":
                index_choice = st.radio("Toggle Recommended Recreational Index:", ["NDVI", "VARI"], horizontal=True)
                primary_vi_name = index_choice
                primary_vi_val = ndvi if index_choice == "NDVI" else vari
                
            else:
                index_choice = st.radio("Toggle Recommended Forest Index:", ["NDRE", "CIRE"], horizontal=True)
                primary_vi_name = index_choice
                primary_vi_val = ndre if index_choice == "NDRE" else cire

            st.info(f"📊 **Active Predictor ({primary_vi_name}):** {round(primary_vi_val, 4)}")

            # --- Safety Threshold: Out-of-Bounds Shadow/Noise Rejection ---
            if primary_vi_val < 0.05:
                st.error("⚠️ **Anomalous Spectral Signature Detected.** The targeted coordinate indicates a non-vegetative surface, exposed infrastructure, or deep canopy shadow. Fungal diversity cannot be estimated.")
            else:
                # --- Step 5a: Fungal Diversity Output (Linear Engine) ---
                st.markdown("#### Fungal Community Profiling Estimates")
                col_a, col_b = st.columns(2)
                
                # Calculate using the dynamically selected index from the toggle
                shannon_val = (fungal_models["Shannon"]["m"] * primary_vi_val) + fungal_models["Shannon"]["c"]
                richness_val = (fungal_models["Richness"]["m"] * primary_vi_val) + fungal_models["Richness"]["c"]
                simpson_val = (fungal_models["Simpson"]["m"] * primary_vi_val) + fungal_models["Simpson"]["c"]
                evenness_val = (fungal_models["Evenness"]["m"] * primary_vi_val) + fungal_models["Evenness"]["c"]
                
                col_a.metric("Shannon (H')", round(shannon_val, 4))
                col_a.metric("Simpson (1-D)", round(simpson_val, 4))
                col_b.metric("Species Richness (S)", round(richness_val, 4))
                col_b.metric("Species Evenness (J')", round(evenness_val, 4))
            
            st.write("---")

            # --- Step 5b: Structural Forest Health Classification (Neural Net Engine) ---
            st.markdown("#### Structural Forest Health Zonal Classification")
            
            if weights is not None:
                # The neural network ALWAYS requires all 7 indices for its forward pass
                features = np.array([[ndvi, gndvi, ndre, cire, savi, msavi, vari]], dtype=np.float32)
                
                # Extract structural MAT variables
                W1 = weights['W1']          
                b1 = weights['b1'].flatten() 
                W2 = weights['W2']          
                b2 = weights['b2'].flatten() 
                
                mu = weights.get('mu')
                sigma = weights.get('sigma')
                
                # Standardize
                if mu is not None and mu.size > 0 and sigma is not None and sigma.size > 0:
                    features = (features - mu.flatten()) / sigma.flatten()
                
                # Hidden Layer (ReLU)
                z1 = np.dot(features, W1.T) + b1
                a1 = np.maximum(0, z1) 
                
                # Output Layer
                z2 = np.dot(a1, W2.T) + b2
                pred_class_idx = int(np.argmax(z2, axis=1)[0]) + 1 
                
                # Metadata Mapping
                class_mapping = {
                    1: "Class 1 (Disturbed / TRA) — High Urban Fragmentation & Canopy Stress",
                    2: "Class 2 (Secondary / STF) — Regenerative Regrowth & Moderate Structural Density",
                    3: "Class 3 (Pristine / TNP) — Dense Primary Climax Canopy Framework"
                }
                
                pred_label = class_mapping.get(pred_class_idx, f"Unidentified Eco-Class ID ({pred_class_idx})")
                
                # UI Output
                if pred_class_idx == 1:
                    st.error(f"🚨 **Ecosystem State:** {pred_label}")
                elif pred_class_idx == 2:
                    st.warning(f"⚠️ **Ecosystem State:** {pred_label}")
                elif pred_class_idx == 3:
                    st.success(f"🌲 **Ecosystem State:** {pred_label}")
            else:
                st.info("Neural Network evaluation is offline. Check backend weight files.")
else:
    # Default landing screen
    st.info("👈 Please define the ecosystem context and upload a multispectral TIFF file via the sidebar to begin.")
