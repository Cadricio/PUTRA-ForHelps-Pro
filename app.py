import streamlit as st
import rasterio
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PUTRA ForHelps", layout="wide", initial_sidebar_state="expanded")

# --- PART A: Thesis Regression Constants (y = mx + c) ---
# Statistically corrected (VI on X-axis, Diversity on Y-axis)
fungal_models = {
    "Shannon": {
        "NDVI": {"m": 1.8543, "c": -0.6830},
        "SAVI": {"m": 1.2367, "c": -0.6834},
        "MSAVI": {"m": 3.1055, "c": -1.9543},
        "VARI": {"m": 0.9631, "c": 0.5954},
        "NDRE": {"m": 1.1079, "c": 0.1769},
        "CIRE": {"m": 0.0516, "c": 0.6926}
    },
    "Simpson": {
        "NDVI": {"m": 1.9919, "c": -0.8033},
        "SAVI": {"m": 1.3284, "c": -0.8036},
        "MSAVI": {"m": 3.3402, "c": -2.1728},
        "VARI": {"m": 1.0408, "c": 0.5681},
        "NDRE": {"m": 1.1765, "c": 0.1293},
        "CIRE": {"m": 0.0546, "c": 0.6777}
    },
    "Richness": {
        "NDVI": {"m": 43.9998, "c": -24.8319},
        "SAVI": {"m": 29.3411, "c": -24.8365},
        "MSAVI": {"m": 70.9345, "c": -52.4702},
        "VARI": {"m": 18.8463, "c": 6.7589},
        "NDRE": {"m": 35.0274, "c": -10.1097},
        "CIRE": {"m": 1.7435, "c": 5.7449}
    },
    "Evenness": {
        "NDVI": {"m": 0.5823, "c": 0.3211},
        "SAVI": {"m": 0.3884, "c": 0.3209},
        "MSAVI": {"m": 1.0414, "c": -0.1390},
        "VARI": {"m": 0.3988, "c": 0.6923},
        "NDRE": {"m": 0.1377, "c": 0.7278},
        "CIRE": {"m": 0.0037, "c": 0.8027}
    }
}

# --- PART B: Sidebar UI - Ecosystem Context & File Upload ---
st.sidebar.title("🌲 PUTRA ForHelps")
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

# --- PART C: Main Processing UI - Spatial Query & Analytics ---
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
        # key="map_click" maintains coordinate state across Streamlit UI toggles
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
            # Automatically scales 16-bit uint down to true 0.0-1.0 reflectance
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
                b_re    = 0.0  
                b_nir   = np.clip((float(data[3, y, x]) / 10000.0), 0.0, 1.0)
            else:
                b_blue  = np.clip((float(data[0, y, x]) / 10000.0), 0.0, 1.0) if num_bands > 0 else 0.0
                b_green = np.clip((float(data[1, y, x]) / 10000.0), 0.0, 1.0) if num_bands > 1 else 0.0
                b_red   = np.clip((float(data[2, y, x]) / 10000.0), 0.0, 1.0) if num_bands > 2 else 0.0
                b_re    = np.clip((float(data[3, y, x]) / 10000.0), 0.0, 1.0) if num_bands > 3 else 0.0
                b_nir   = np.clip((float(data[4, y, x]) / 10000.0), 0.0, 1.0) if num_bands > 4 else 0.0

            # --- Step 4: User Toggles Between Suitable Indices ---
            st.markdown("#### Select Localized Index")
            
            primary_vi_val = 0.0
            primary_vi_name = ""

            if site_type == "Urban / Fragmented":
                index_choice = st.radio("Toggle Recommended Urban Index:", ["SAVI", "MSAVI"], horizontal=True)
