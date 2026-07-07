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

    # 2. Split-screen user interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Step 3: Spatial Query")
        st.write("Click anywhere on the canopy map to extract localized biophysical metrics:")
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
