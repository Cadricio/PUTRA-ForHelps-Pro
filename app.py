import streamlit as st
import rasterio
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PUTRA ForHelps", layout="wide", initial_sidebar_state="expanded")

# --- PART A: Thesis Regression Constants (y = mx + c) ---
# Values extracted directly from final Excel aggregated scatter plots
fungal_models = {
    "Shannon": {
        "NDVI": {"m": 4.97, "c": -0.901},
        "SAVI": {"m": 3.07, "c": -0.593},
        "MSAVI": {"m": 7.98, "c": -4.0},
        "VARI": {"m": 2.79, "c": 2.46},
        "NDRE": {"m": 2.74, "c": 1.56},
        "CIRE": {"m": 0.132, "c": 2.81}
    },
    "Simpson": {
        "NDVI": {"m": 0.273, "c": 0.715},
        "SAVI": {"m": 0.175, "c": 0.725},
        "MSAVI": {"m": 0.451, "c": 0.534},
        "VARI": {"m": 0.154, "c": 0.899},
        "NDRE": {"m": 0.143, "c": 0.855},
        "CIRE": {"m": 0.0067, "c": 0.921}
    },
    "Richness": {
        "NDVI": {"m": 348.0, "c": -238.0},
        "SAVI": {"m": 176.0, "c": -166.0},
        "MSAVI": {"m": 479.0, "c": -382.0},
        "VARI": {"m": 193.0, "c": -2.08},
        "NDRE": {"m": 248.0, "c": -102.0},
        "CIRE": {"m": 13.3, "c": 5.61}
    },
    "Evenness": {
        "NDVI": {"m": 0.0132, "c": 0.813},
        "SAVI": {"m": 0.124, "c": 0.666},
        "MSAVI": {"m": 0.256, "c": 0.590},
        "VARI": {"m": 0.0151, "c": 0.820},
        "NDRE": {"m": -0.158, "c": 0.927},
        "CIRE": {"m": -0.0115, "c": 0.870}
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
                primary_vi_name = index_choice
                savi = ((b_nir - b_red) / (b_nir + b_red + 0.5 + 1e-8)) * 1.5
                msavi = 0.5 * (2 * b_nir + 1 - np.sqrt(abs((2 * b_nir + 1)**2 - 8 * (b_nir - b_red))))
                primary_vi_val = savi if index_choice == "SAVI" else msavi
                
            elif site_type == "Recreational / Moderate":
                index_choice = st.radio("Toggle Recommended Recreational Index:", ["NDVI", "VARI"], horizontal=True)
                primary_vi_name = index_choice
                ndvi = (b_nir - b_red) / (b_nir + b_red + 1e-8)
                vari_denom = (b_green + b_red - b_blue + 1e-8)
                vari = (b_green - b_red) / vari_denom if vari_denom != 0 else 0.0
                primary_vi_val = ndvi if index_choice == "NDVI" else vari
                
            else:
                index_choice = st.radio("Toggle Recommended Forest Index:", ["NDRE", "CIRE"], horizontal=True)
                primary_vi_name = index_choice
                ndre = (b_nir - b_re) / (b_nir + b_re + 1e-8)
                cire = (b_nir / (b_re + 1e-8)) - 1.0
                primary_vi_val = ndre if index_choice == "NDRE" else cire

            st.info(f"📊 **Active Predictor ({primary_vi_name}):** {round(primary_vi_val, 4)}")

            # --- Safety Threshold: Unblocked Warning ---
            if primary_vi_val < 0.05:
                st.warning("⚠️ **Low Spectral Signature Detected.** This pixel resembles shadow or exposed soil. Raw math is being processed below for testing.")
            
            # --- Step 5: Fungal Diversity Output (Linear Engine) ---
            st.markdown("#### Fungal Community Profiling Estimates")
            col_a, col_b = st.columns(2)
            
            shannon_m = fungal_models["Shannon"][primary_vi_name]["m"]
            shannon_c = fungal_models["Shannon"][primary_vi_name]["c"]
            shannon_val = (shannon_m * primary_vi_val) + shannon_c
            
            richness_m = fungal_models["Richness"][primary_vi_name]["m"]
            richness_c = fungal_models["Richness"][primary_vi_name]["c"]
            richness_val = (richness_m * primary_vi_val) + richness_c
            
            simpson_m = fungal_models["Simpson"][primary_vi_name]["m"]
            simpson_c = fungal_models["Simpson"][primary_vi_name]["c"]
            simpson_val = (simpson_m * primary_vi_val) + simpson_c
            
            evenness_m = fungal_models["Evenness"][primary_vi_name]["m"]
            evenness_c = fungal_models["Evenness"][primary_vi_name]["c"]
            evenness_val = (evenness_m * primary_vi_val) + evenness_c
            
            col_a.metric("Shannon (H')", round(shannon_val, 4))
            col_a.metric("Simpson (1-D)", round(simpson_val, 4))
            col_b.metric("Species Richness (S)", round(richness_val, 4))
            col_b.metric("Species Evenness (J')", round(evenness_val, 4))
else:
    # Default landing screen
    st.info("👈 Please define the ecosystem context and upload a multispectral TIFF file via the sidebar to begin.")
