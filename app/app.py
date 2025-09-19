import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from html import escape

from utils.preprocessing import extract_red_nir, compute_ndvi, normalize_image, smooth_image
from utils.data_loader import load_sensor_csv, read_image, generate_synthetic_field
from utils.analytics import soil_quality_rules, PestRiskModel


st.set_page_config(page_title="Smart Agro Dashboard", layout="wide")

# ---------- Sidebar Controls ----------
st.sidebar.title("Controls")

st.sidebar.markdown("**Mode**")
ui_mode = st.sidebar.radio("Choose mode", ["Farmer", "Advanced"], index=0)

st.sidebar.markdown("**Satellite / Drone Image**")
image_mode = st.sidebar.radio(
    "Image source",
    ["Synthetic", "Single RGB image", "Separate Red+NIR bands"],
    index=0,
    disabled=(ui_mode == "Farmer"),
)

# Upload widgets
rgb_image_file = None
red_band_file = None
nir_band_file = None
if image_mode == "Single RGB image":
    rgb_image_file = st.sidebar.file_uploader(
        "Upload RGB image (png/jpg/tif)", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )
elif image_mode == "Separate Red+NIR bands":
    red_band_file = st.sidebar.file_uploader(
        "Upload Red band (grayscale)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="red_band"
    )
    nir_band_file = st.sidebar.file_uploader(
        "Upload NIR band (grayscale)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="nir_band"
    )

smoothing = st.sidebar.slider("Smoothing (Gaussian kernel)", min_value=0, max_value=9, step=2, value=3, disabled=(ui_mode == "Farmer"))

st.sidebar.markdown("**Sensor Data**")
default_csv = os.path.join("app", "data", "sensors.csv")
custom_csv = st.sidebar.text_input("CSV path (optional)", value="", disabled=(ui_mode == "Farmer"))
csv_path = custom_csv.strip() if custom_csv.strip() else default_csv

st.sidebar.markdown("**Crop & Alerts**")
crop_choice = st.sidebar.selectbox(
    "Crop",
    ["Paddy (Rice)", "Wheat", "Maize", "Cotton", "Soybean", "Sugarcane", "Pulses"],
    index=0,
)
stage_choice = st.sidebar.selectbox(
    "Crop Stage",
    ["Nursery", "Vegetative", "Flowering"],
    index=1,
)
language = st.sidebar.selectbox("Language / भाषा / భాష / भाषा", ["English", "Hindi", "Telugu", "Marathi"]) 

# ---------- Load Data ----------
def _read_rgb_from_upload(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr

def _read_gray_from_upload(file) -> np.ndarray:
    img = Image.open(file).convert("L")
    arr = np.asarray(img).astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr

# Image
rgb_img = None
red_band = None
nir_band = None

if image_mode == "Synthetic":
    rgb_img = generate_synthetic_field(size=(256, 256), seed=42)
elif image_mode == "Single RGB image":
    if rgb_image_file is not None:
        rgb_img = _read_rgb_from_upload(rgb_image_file)
    else:
        st.warning("Please upload an RGB image or switch to Synthetic.")
        rgb_img = generate_synthetic_field(size=(256, 256), seed=42)
elif image_mode == "Separate Red+NIR bands":
    if red_band_file is not None and nir_band_file is not None:
        red_band = _read_gray_from_upload(red_band_file)
        nir_band = _read_gray_from_upload(nir_band_file)
        # Basic size check and resize NIR to match Red if needed
        if red_band.shape != nir_band.shape:
            st.info("Resizing NIR to match Red dimensions for NDVI alignment.")
            target_size = (red_band.shape[1], red_band.shape[0])  # (width, height)
            nir_band = np.array(
                Image.fromarray((nir_band * 255).astype(np.uint8)).resize(target_size, resample=Image.BILINEAR)
            ).astype(np.float32) / 255.0
    else:
        st.warning("Upload both Red and NIR band images, or switch to another mode.")
        rgb_img = generate_synthetic_field(size=(256, 256), seed=42)

if smoothing and smoothing > 0 and rgb_img is not None:
    rgb_img = smooth_image(rgb_img, ksize=max(1, smoothing))

# Sensors
sensor_df = None
sensor_error = None
try:
    sensor_df = load_sensor_csv(csv_path)
except Exception as e:
    sensor_error = str(e)

# ---------- Layout ----------
st.title("Smart Agro Monitoring Dashboard")
st.caption("Green = Good. Red = Risky. Simple alerts for quick action.")

# ---------- Localization (EN, HI, TE, MR) ----------
LANG_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Marathi": "mr",
}

# Stage adjustments and fertilizer hints
STAGES = {
    "Nursery": {"moist_adj": +5, "temp_hi_adj": -1, "fert_key": "fert_nursery"},
    "Vegetative": {"moist_adj": 0, "temp_hi_adj": 0, "fert_key": "fert_vegetative"},
    "Flowering": {"moist_adj": -2, "temp_hi_adj": -1, "fert_key": "fert_flowering"},
}

STRINGS = {
    "title": {
        "en": "Smart Agro Monitoring Dashboard",
        "hi": "स्मार्ट एग्रो डैशबोर्ड",
        "te": "స్మార్ట్ అగ్రో డ్యాష్‌బోర్డ్",
        "mr": "स्मार्ट अ‍ॅग्रो डॅशबोर्ड",
    },
    "caption": {
        "en": "Green = Good. Red = Risky. Simple alerts for quick action.",
        "hi": "हरा = अच्छा, लाल = जोखिम। त्वरित कार्य हेतु सरल अलर्ट।",
        "te": "ఆకుపచ్చ = మంచి, ఎరుపు = ప్రమాదం. తక్షణ చర్యలకు సరళ అలర్ట్లు.",
        "mr": "हिरवे = चांगले, लाल = धोका. त्वरित कृतीसाठी सोपे अलर्ट.",
    },
    "soil_status": {"en": "Soil Status", "hi": "मृदा स्थिति", "te": "మట్టి స్థితి", "mr": "मातीची स्थिती"},
    "pest_risk": {"en": "Pest Risk", "hi": "कीट जोखिम", "te": "కీటక ప్రమాదం", "mr": "किडीचा धोका"},
    "crop_health": {"en": "Crop Health (NDVI)", "hi": "फसल स्वास्थ्य (NDVI)", "te": "పంట ఆరోగ్యం (NDVI)", "mr": "पीक स्वास्थ्य (NDVI)"},
    "send_sms": {"en": "Send SMS Alert", "hi": "SMS अलर्ट भेजें", "te": "SMS అలర్ట్ పంపండి", "mr": "SMS सूचना पाठवा"},
    "details": {"en": "Details", "hi": "विवरण", "te": "వివరాలు", "mr": "तपशील"},
    "good": {"en": "Good", "hi": "अच्छा", "te": "మంచిది", "mr": "चांगले"},
    "okay": {"en": "Okay", "hi": "ठीक", "te": "సరే", "mr": "ठीक"},
    "low": {"en": "Low", "hi": "कम", "te": "తక్కువ", "mr": "कमी"},
    "risky": {"en": "Risky", "hi": "जोखिम", "te": "ప్రమాదం", "mr": "धोका"},
    "high": {"en": "High", "hi": "उच्च", "te": "అధిక", "mr": "जास्त"},
    "mean_ndvi": {"en": "Mean NDVI", "hi": "औसत NDVI", "te": "సగటు NDVI", "mr": "सरासरी NDVI"},
    "humidity": {"en": "Humidity", "hi": "आर्द्रता", "te": "ఆర్ద్రత", "mr": "आर्द्रता"},
    "temperature": {"en": "Temp", "hi": "तापमान", "te": "ఉష్ణోగ్రత", "mr": "तापमान"},
    "crop": {"en": "Crop", "hi": "फसल", "te": "పంట", "mr": "पीक"},
    "stage": {"en": "Stage", "hi": "अवस्था", "te": "దశ", "mr": "अवस्था"},
    "recommendations": {"en": "Recommendations", "hi": "सुझाव", "te": "సూచనలు", "mr": "शिफारसी"},
    "rec_irrigate": {
        "en": "Irrigate today – soil moisture is low.",
        "hi": "आज सिंचाई करें – मिट्टी में नमी कम है।",
        "te": "ఈ రోజు నీటిపారుదల చేయండి – మట్టిలో తేమ తక్కువగా ఉంది.",
        "mr": "आज पाणी द्या – मातीत ओलावा कमी आहे."
    },
    "rec_mulch": {
        "en": "High heat/dry air: mulch soil and avoid noon irrigation.",
        "hi": "अधिक गर्मी/सूखी हवा: मल्च करें और दोपहर की सिंचाई से बचें।",
        "te": "ఎక్కువ వేడి/ఎండ: మల్చ్ వాడండి, మధ్యాహ్నం నీరుపోవడం నివారించండి.",
        "mr": "उष्णता/कोरडे हवामान: मळणी करा, दुपारी पाणी देणे टाळा."
    },
    "rec_nutrient": {
        "en": "Low crop health: check irrigation and consider light nutrient spray.",
        "hi": "फसल स्वास्थ्य कम: सिंचाई जाँचें और हल्की पोषक स्प्रे पर विचार करें।",
        "te": "పంట ఆరోగ్యం తక్కువ: నీరుపోయడం చూసుకోండి, తేలికపాటి పోషక స్ప్రే పరిగణించండి.",
        "mr": "पीक स्वास्थ्य कमी: सिंचन तपासा आणि हलकी पोषक फवारणी करा."
    },
    "rec_monitor": {
        "en": "Monitor pest signs: inspect leaves (underside) for eggs/larvae.",
        "hi": "कीट संकेत देखें: पत्तों (नीचे की तरफ) पर अंडे/लार्वा जाँचें।",
        "te": "కీటక లక్షణాలు గమనించండి: ఆకుల దిగువ భాగం పరిశీలించండి.",
        "mr": "किडीची चिन्हे पाहा: पानांच्या खालच्या बाजूस तपासा."
    },
    "rec_biopesticide": {
        "en": "High pest risk: use traps and apply biopesticide (e.g., neem).",
        "hi": "उच्च कीट जोखिम: ट्रैप लगाएँ और बायोपेस्टिसाइड (जैसे नीम) लगाएँ।",
        "te": "పురుగు ప్రమాదం ఎక్కువ: ట్రాప్స్ వాడండి, బయోపెస్టిసైడ్ (నింబోళి) వాడండి.",
        "mr": "उच्च किडीचा धोका: सापळे लावा आणि जैविक कीटकनाशक (उदा. निम) वापरा."
    },
    "rec_good": {
        "en": "Status good: continue regular schedule.",
        "hi": "स्थिति अच्छी: नियमित कार्यक्रम जारी रखें।",
        "te": "స్థితి బాగుంది: సాధారణ షెడ్యూల్ కొనసాగించండి.",
        "mr": "स्थिती चांगली: नियमित वेळापत्रक सुरू ठेवा."
    },
    "fertilizer": {"en": "Fertilizer (hint)", "hi": "उर्वरक (संकेत)", "te": "ఎరువు (సూచన)", "mr": "खत (सूचना)"},
    "fert_nursery": {
        "en": "Nursery: keep nutrition light; avoid heavy urea.",
        "hi": "नर्सरी: हल्का पोषण रखें; अधिक यूरिया न दें।",
        "te": "నర్సరీ: తేలికపాటి పోషణ; ఎక్కువ యూరియా వద్దు.",
        "mr": "नर्सरी: हलके पोषण ठेवा; जास्त युरिया देऊ नका."
    },
    "fert_vegetative": {
        "en": "Vegetative: split urea application; DAP if needed.",
        "hi": "वेजिटेटिव: यूरिया खुराक विभाजित करें; जरूरत हो तो DAP दें।",
        "te": "వెజిటేటివ్: యూరియా వంతులుగా ఇవ్వండి; అవసరమైతే DAP.",
        "mr": "वेजिटेटिव: युरिया हप्त्यांमध्ये द्या; गरज असल्यास DAP."
    },
    "fert_flowering": {
        "en": "Flowering: avoid excess nitrogen; prefer foliar micronutrients.",
        "hi": "फ्लावरिंग: अधिक नाइट्रोजन से बचें; सूक्ष्म पोषक फोलियर दें।",
        "te": "పుష్పించడం: అధిక నైట్రోజన్ నివారించండి; మైక్రోన్యూట్రియంట్స్ స్ప్రే చేయండి.",
        "mr": "फुलोरा: जास्त नायट्रोजन टाळा; सूक्ष्म पोषक फवारणी करा."
    },
}

def t(key: str, lang_name: str) -> str:
    code = LANG_MAP.get(lang_name, "en")
    return STRINGS.get(key, {}).get(code, STRINGS.get(key, {}).get("en", key))

# ---------- Crop knowledge (simple heuristics for India) ----------
CROPS = {
    "Paddy (Rice)": {
        "moisture_min": 25,  # %
        "temp_range": (20, 35),
        "pests": {"en": "stem borer/brown planthopper", "hi": "तना छेदक/भूरा तिलचट्टा", "te": "స్టెమ్ బోరర్/బ్రౌన్ ప్లాంతాపర్", "mr": "खोडकिडा/तांबड्या पानावरचा कीडा"},
    },
    "Wheat": {
        "moisture_min": 20,
        "temp_range": (15, 30),
        "pests": {"en": "aphids/armyworm", "hi": "चेपा/आर्मी वर्म", "te": "ఆఫిడ్స్/ఆర్మీవరం", "mr": "मावा/आर्मीवर्म"},
    },
    "Maize": {
        "moisture_min": 20,
        "temp_range": (18, 35),
        "pests": {"en": "fall armyworm", "hi": "फॉल आर्मीवर्म", "te": "ఫాల్ ఆర్మీవారం", "mr": "फॉल आर्मीवर्म"},
    },
    "Cotton": {
        "moisture_min": 18,
        "temp_range": (20, 38),
        "pests": {"en": "bollworm/whitefly", "hi": "बोलवर्म/सफ़ेद मक्खी", "te": "బాల్‌వార్మ్/వైట్‌ఫ్లై", "mr": "बोंड अळी/पांढरी माशी"},
    },
    "Soybean": {
        "moisture_min": 22,
        "temp_range": (18, 35),
        "pests": {"en": "girdle beetle/aphids", "hi": "गर्डल बीटल/चेपा", "te": "గిర్డిల్ బీటిల్/ఆఫిడ్స్", "mr": "गर्डल भुंगा/मावा"},
    },
    "Sugarcane": {
        "moisture_min": 30,
        "temp_range": (20, 38),
        "pests": {"en": "early shoot borer/top borer", "hi": "अर्ली शूट बोरर/टॉप बोरर", "te": "అర్లీ షూట్ బోరర్/టాప్ బోరర్", "mr": "अर्ली शुट बोरर/टॉप बोरर"},
    },
    "Pulses": {
        "moisture_min": 18,
        "temp_range": (20, 35),
        "pests": {"en": "pod borer/aphids", "hi": "पॉड बोरर/चेपा", "te": "పాడ్ బోరర్/ఆఫిడ్స్", "mr": "फळकिडा/मावा"},
    },
}

def make_recommendations(ndvi_mean: float, assess, pr_label: str, latest_row: pd.Series, lang: str, crop: str | None = None, stage: str | None = None) -> list[str]:
    recs: list[str] = []
    moisture = latest_row.get("moisture", np.nan)
    humidity = latest_row.get("humidity", np.nan)
    temp = latest_row.get("temp", np.nan)
    if pd.notna(moisture) and moisture <= 1.0:
        moisture = moisture * 100.0

    crop_info = CROPS.get(crop or "", None)
    crop_moist_min = crop_info.get("moisture_min") if crop_info else 20
    crop_temp_lo, crop_temp_hi = crop_info.get("temp_range", (18, 35)) if crop_info else (18, 35)
    # Apply stage adjustments
    stage_info = STAGES.get(stage or "", {"moist_adj": 0, "temp_hi_adj": 0, "fert_key": None})
    crop_moist_min = crop_moist_min + stage_info.get("moist_adj", 0)
    crop_temp_hi = crop_temp_hi + stage_info.get("temp_hi_adj", 0)

    # Soil moisture recommendation
    if assess and ((assess.moisture_status == "Low") or (pd.notna(moisture) and moisture < crop_moist_min)):
        recs.append(t("rec_irrigate", lang))

    # Heat/dry air recommendation
    if pd.notna(temp) and pd.notna(humidity) and (temp >= crop_temp_hi or humidity < 30):
        recs.append(t("rec_mulch", lang))

    # NDVI-based crop stress
    if ndvi_mean < 0.2:
        recs.append(t("rec_nutrient", lang))
    elif ndvi_mean < 0.4 and len(recs) == 0:
        recs.append(t("rec_nutrient", lang))

    # Pest risk
    if pr_label == "High":
        # mention crop-specific pests
        pest_note = None
        if crop_info:
            pest_note = crop_info["pests"].get(LANG_MAP.get(lang, "en"), crop_info["pests"]["en"])
        recs.append(t("rec_monitor", lang) + (f" ({pest_note})" if pest_note else ""))
        recs.append(t("rec_biopesticide", lang))

    # Fertilizer hint for stage
    fert_key = stage_info.get("fert_key")
    if fert_key:
        recs.append(t("fertilizer", lang) + ": " + t(fert_key, lang))

    if not recs:
        recs.append(t("rec_good", lang))
    return recs

# Utilities to export sample images
def _save_numpy_image(path: str, arr: np.ndarray, mode: str | None = None):
    """Save a numpy array [0..1] or uint8 image to disk as PNG/JPEG.
    If 2D -> grayscale 'L'; if 3D shape (H,W,3) -> RGB.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = arr
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255).astype(np.uint8)
    if a.ndim == 2:
        img = Image.fromarray(a, mode=mode or "L")
    else:
        img = Image.fromarray(a, mode=mode or "RGB")
    img.save(path)

with st.expander("Sample Images (click to generate test files)", expanded=(ui_mode == "Advanced")):
    st.write("Generate a few ready-to-use images under `app/data/` for quick testing.")
    if st.button("Generate sample images", disabled=(ui_mode == "Farmer")):
        base = generate_synthetic_field(size=(256, 256), seed=42)
        # Create a stressed variant by reducing vegetation intensity
        stressed = np.clip(base * 0.7, 0.0, 1.0)

        red_band_syn, nir_band_syn = extract_red_nir(base)
        # Save files
        _save_numpy_image(os.path.join("app", "data", "synthetic_field.png"), base)
        _save_numpy_image(os.path.join("app", "data", "stressed_field.png"), stressed)
        _save_numpy_image(os.path.join("app", "data", "red_band.png"), red_band_syn, mode="L")
        _save_numpy_image(os.path.join("app", "data", "nir_band.png"), nir_band_syn, mode="L")
        st.success("Generated: synthetic_field.png, stressed_field.png, red_band.png, nir_band.png in app/data/")

def status_card(title: str, value: str, color: str, subtext: str = "", icon: str = ""):
    bg = "#e8f5e9" if color == "green" else "#ffebee"
    fg = "#1b5e20" if color == "green" else "#b71c1c"
    st.markdown(
        f"""
        <div style='border-radius:8px;padding:16px;background:{bg};border:1px solid {fg};'>
            <div style='font-size:18px;color:{fg};font-weight:600;'>{icon} {title}</div>
            <div style='font-size:28px;color:{fg};font-weight:800;margin-top:4px;'>{value}</div>
            <div style='font-size:14px;color:{fg};opacity:0.8;margin-top:4px;'>{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Compute NDVI and assessments once
if image_mode == "Separate Red+NIR bands" and red_band is not None and nir_band is not None:
    red_src, nir_src = red_band, nir_band
else:
    red_src, nir_src = extract_red_nir(rgb_img)

ndvi = compute_ndvi(nir_src, red_src)
ndvi_mean = float(np.nanmean(ndvi)) if np.isfinite(ndvi).any() else 0.0

latest = sensor_df.iloc[-1] if sensor_df is not None and not sensor_df.empty else pd.Series({"moisture": np.nan, "temp": np.nan, "humidity": np.nan})
assess = soil_quality_rules(latest) if sensor_df is not None and not sensor_df.empty else None

pr_model = PestRiskModel()
humidity_val = float(latest.get("humidity", 55.0)) if pd.notna(latest.get("humidity", np.nan)) else 55.0
temp_val = float(latest.get("temp", 28.0)) if pd.notna(latest.get("temp", np.nan)) else 28.0
pr_label, pr_proba = pr_model.predict_risk(ndvi_mean, humidity_val, temp_val)

if ui_mode == "Farmer":
    # Big banners
    colA, colB = st.columns(2)
    with colA:
        soil_color = "green" if assess and assess.overall == "Good" else "red"
        soil_value_key = "good" if assess and assess.overall == "Good" else "risky"
        soil_value = t(soil_value_key, language) if assess else "--"
        sub_m = t("good", language) if assess and assess.moisture_status == "Good" else t("low", language) if assess else "--"
        sub_t = t("good", language) if assess and assess.temp_status == "Good" else t("risky", language) if assess else "--"
        sub_h = t("good", language) if assess and assess.humidity_status == "Good" else t("low", language) if assess else "--"
        status_card(t("soil_status", language), soil_value, soil_color,
                    subtext=f"💧 {sub_m} · 🌡️ {sub_t} · 💨 {sub_h}", icon="💧")
    with colB:
        pest_color = "red" if pr_label == "High" else "green"
        pest_value = t("high", language) if pr_label == "High" else t("low", language)
        status_card(t("pest_risk", language), f"{pest_value}", pest_color, subtext=f"{pr_proba*100:.0f}%", icon="🐛")

    # NDVI quick glance
    colC, colD = st.columns([1, 1])
    with colC:
        health_key = "good" if ndvi_mean >= 0.4 else ("okay" if ndvi_mean >= 0.2 else "low")
        color = "green" if health_key in ("good", "okay") else "red"
        status_card(t("crop_health", language), t(health_key, language), color, subtext=f"{t('mean_ndvi', language)} {ndvi_mean:.2f}", icon="🌱")
    with colD:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.axis('off')
        st.pyplot(fig, clear_figure=True)

    # Recommendations
    recs = make_recommendations(ndvi_mean, assess, pr_label, latest, language, crop_choice, stage_choice)
    st.subheader(t("recommendations", language))
    for r in recs:
        st.markdown(f"- {r}")

    # Simple action button
    if st.button(t("send_sms", language)):
        msg = {
            "English": "SMS Alert: Moisture low, water crops today.",
            "Hindi": "SMS सूचना: नमी कम है, आज सिंचाई करें।",
            "Telugu": "SMS హెచ్చరిక: తేమ తక్కువగా ఉంది, ఈ రోజు నీళ్లు పోయండి.",
            "Marathi": "SMS इशारा: मातीतील ओलावा कमी आहे, आज पाणी द्या."
        }.get(language, "SMS Alert: Moisture low, water crops today.")
        st.info(msg)

    with st.expander(t("details", language)):
        # Show charts and previews only when needed
        if sensor_df is not None:
            disp_df = sensor_df.copy()
            if disp_df["moisture"].max() <= 1.0:
                disp_df["moisture"] = disp_df["moisture"] * 100.0
            disp_df = disp_df.melt(id_vars=["timestamp"], value_vars=["moisture", "temp", "humidity"], var_name="metric", value_name="value")
            fig_line = px.line(disp_df, x="timestamp", y="value", color="metric", markers=True)
            fig_line.update_layout(height=300, legend_title_text="")
            st.plotly_chart(fig_line, use_container_width=True)
        # Image previews
        if image_mode == "Separate Red+NIR bands" and red_band is not None and nir_band is not None:
            prev1, prev2 = st.columns(2)
            with prev1:
                st.image(red_src, caption="Red band", clamp=True)
            with prev2:
                st.image(nir_src, caption="NIR band", clamp=True)
        else:
            st.image((rgb_img * 255).astype(np.uint8), caption="Input image")
else:
    # Advanced: original detailed layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Crop Health (NDVI)")
        if image_mode == "Separate Red+NIR bands" and red_band is not None and nir_band is not None:
            red, nir = red_band, nir_band
            st.caption("Uploaded bands (left: Red, right: NIR)")
            prev1, prev2 = st.columns(2)
            with prev1:
                st.image(red, caption="Red band", clamp=True)
            with prev2:
                st.image(nir, caption="NIR band", clamp=True)
        else:
            red, nir = extract_red_nir(rgb_img)
            st.caption("Using RGB image to approximate Red/NIR bands.")
            st.image((rgb_img * 255).astype(np.uint8), caption="Input image")

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('NDVI (-1 to 1)')
        st.pyplot(fig, clear_figure=True)

        st.caption("NDVI = (NIR - Red) / (NIR + Red). Higher is healthier.")

    with col2:
        st.subheader("Sensor Trends")
        if sensor_df is not None:
            disp_df = sensor_df.copy()
            if disp_df["moisture"].max() <= 1.0:
                disp_df["moisture"] = disp_df["moisture"] * 100.0
            disp_df = disp_df.melt(id_vars=["timestamp"], value_vars=["moisture", "temp", "humidity"], var_name="metric", value_name="value")
            fig_line = px.line(disp_df, x="timestamp", y="value", color="metric", markers=True)
            fig_line.update_layout(height=360, legend_title_text="")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.error(sensor_error or "Sensor data unavailable.")

    # ---------- Risk & Assessments ----------
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Soil Quality Assessment")
        if sensor_df is not None and not sensor_df.empty:
            color = "green" if assess.overall == "Good" else "red"
            st.markdown(f"- Moisture: **{assess.moisture_status}**\n- Temperature: **{assess.temp_status}**\n- Humidity: **{assess.humidity_status}**")
            st.markdown(f"**Overall: <span style='color:{color}'>" + assess.overall + "</span>**", unsafe_allow_html=True)

            if st.button("Simulate SMS Alert"):
                msg = {
                    "English": "SMS Alert: Moisture low, water crops today.",
                    "Hindi": "SMS सूचना: नमी कम है, आज सिंचाई करें।",
                    "Telugu": "SMS హెచ్చరిక: తేమ తక్కువగా ఉంది, ఈ రోజు నీళ్లు పోయండి.",
                    "Marathi": "SMS इशारा: मातीतील ओलावा कमी आहे, आज पाणी द्या."
                }.get(language, "SMS Alert: Moisture low, water crops today.")
                st.info(msg)
        else:
            st.write("Waiting for sensor data...")

    with right:
        st.subheader("Pest Outbreak Risk (Demo Model)")
        label, proba = pr_label, pr_proba
        color = "red" if label == "High" else "green"
        st.markdown(f"Mean NDVI: **{ndvi_mean:.3f}** | Humidity: **{humidity_val:.1f}%** | Temp: **{temp_val:.1f}°C**")
        st.markdown(f"**Risk: <span style='color:{color}'>" + label + f" ({proba*100:.1f}%)</span>**", unsafe_allow_html=True)
        st.subheader(t("recommendations", language))
        for r in make_recommendations(ndvi_mean, assess, pr_label, latest, language, crop_choice, stage_choice):
            st.markdown(f"- {r}")

# ---------- Export: Printable advice ----------
def build_advice_html(lang: str, crop: str, stage: str, ndvi_mean: float, assess, pr_label: str, pr_proba: float, recs: list[str]) -> str:
    color = "#1b5e20" if (assess and assess.overall == "Good" and pr_label != "High") else "#b71c1c"
    items = "".join([f"<li>{escape(r)}</li>" for r in recs])
    soil_line = "Unknown" if not assess else f"Moisture {assess.moisture_status} · Temp {assess.temp_status} · Humidity {assess.humidity_status}"
    html = f"""
    <html><head><meta charset='utf-8'><title>Advice</title></head>
    <body style='font-family:Arial,Helvetica,sans-serif;padding:16px;'>
      <h2 style='margin:0;color:{color};'>{t('title', lang)}</h2>
      <p style='opacity:0.85;'>{t('crop', lang)}: {crop} &nbsp; | &nbsp; {t('stage', lang)}: {stage} &nbsp; | &nbsp; {t('mean_ndvi', lang)}: {ndvi_mean:.2f}</p>
      <p style='opacity:0.85;'>Soil: {soil_line} &nbsp; | &nbsp; {t('pest_risk', lang)}: {pr_label} ({pr_proba*100:.0f}%)</p>
      <h3>{t('recommendations', lang)}</h3>
      <ul>{items}</ul>
      <p style='margin-top:24px;opacity:0.6;font-size:12px;'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </body></html>
    """
    return html

def build_advice_text(lang: str, crop: str, stage: str, ndvi_mean: float, assess, pr_label: str, pr_proba: float, recs: list[str]) -> str:
    soil_line = "Unknown" if not assess else f"Moisture {assess.moisture_status} | Temp {assess.temp_status} | Humidity {assess.humidity_status}"
    lines = [
        f"{t('title', lang)}",
        f"{t('crop', lang)}: {crop}",
        f"{t('stage', lang)}: {stage}",
        f"{t('mean_ndvi', lang)}: {ndvi_mean:.2f}",
        f"Soil: {soil_line}",
        f"{t('pest_risk', lang)}: {pr_label} ({pr_proba*100:.0f}%)",
        f"{t('recommendations', lang)}:",
    ] + [f"- {r}" for r in recs]
    return "\n".join(lines)

# Place export buttons below Farmer recommendations
if ui_mode == "Farmer":
    recs_current = make_recommendations(ndvi_mean, assess, pr_label, latest, language, crop_choice, stage_choice)
    html_content = build_advice_html(language, crop_choice, stage_choice, ndvi_mean, assess, pr_label, pr_proba, recs_current)
    txt_content = build_advice_text(language, crop_choice, stage_choice, ndvi_mean, assess, pr_label, pr_proba, recs_current)
    colx, coly = st.columns(2)
    with colx:
        st.download_button(
            label="Download Advice (Printable HTML)",
            data=html_content,
            file_name=f"advice_{LANG_MAP.get(language,'en')}.html",
            mime="text/html",
        )
    with coly:
        st.download_button(
            label="Download Advice (TXT)",
            data=txt_content,
            file_name=f"advice_{LANG_MAP.get(language,'en')}.txt",
            mime="text/plain",
        )

st.divider()

with st.expander("How this works (for judges)"):
    st.markdown(
        """
        - Satellite/Drone image (or synthetic) is preprocessed with smoothing and NDVI computed using `app/utils/preprocessing.py`.
        - Sensor data is simulated via `app/data/sensors.csv`. You can replace with your own CSV.
        - Soil quality uses simple rules in `app/utils/analytics.py`.
        - Pest risk uses a tiny `DecisionTreeClassifier` trained on synthetic features (NDVI mean, humidity, temperature).
        - Dashboard is built with Streamlit for quick hackathon prototyping.
        """
    )
