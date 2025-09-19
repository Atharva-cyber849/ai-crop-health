# Smart Agro Monitoring Dashboard (Hackathon-Ready)

A lightweight, demo-friendly pipeline for remote crop monitoring using satellite/drone imagery and simulated IoT soil sensors. Built for quick hackathon demos with clear visuals and easy setup.

## Features
- Crop Health via NDVI heatmap (Green = Healthy, Red = Stressed)
- Sensor Trends (moisture, temperature, humidity) from CSV
- Soil Quality Assessment (rule-based, farmer-friendly Good/Risky labels)
- Pest Outbreak Risk (tiny DecisionTreeClassifier on NDVI+humidity+temp)
- SMS Alert Simulation in English/Hindi/Telugu/Marathi
- Streamlit dashboard for fast UX
- Farmer mode (simple banners, single action button) and Advanced mode (full controls & charts)

## Tech Stack
- Python, Streamlit
- numpy, pandas (data)
- opencv-python, scipy (preprocessing, NDVI)
- matplotlib, plotly (visualization)
- scikit-learn (simple ML)

## Project Structure
```
SIH2025/
├─ requirements.txt
├─ app/
│  ├─ app.py
│  ├─ __init__.py
│  ├─ data/
│  │  └─ sensors.csv
│  └─ utils/
│     ├─ preprocessing.py
│     ├─ data_loader.py
│     └─ analytics.py
```

## Quick Start
1) Create and activate a virtual environment (recommended)
- Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Run the dashboard
```
python -m streamlit run app/app.py
```

4) Open the URL printed by Streamlit (typically http://localhost:8501)

## Using the App
- Mode → choose `Farmer` for a simplified UI (big color banners, one SMS button) or `Advanced` for full controls.
- Image → use Synthetic, upload a single RGB image, or upload separate Red+NIR bands.
- Sensor data → reads `app/data/sensors.csv` by default. You can provide a custom CSV path.
- Tuning → adjust smoothing to reduce noise before NDVI computation (Advanced mode).
- Alerts → click “Send/Simulate SMS Alert” to show a localized message.

## Data Notes
- Satellite/Drone Data: For demos, synthetic image is generated. You can replace with small tiles from Sentinel‑2, Landsat, or Indian Pines (ensure Red & NIR bands are aligned or adapt `extract_red_nir()` in `preprocessing.py`).
- IoT Sensors: `app/data/sensors.csv` simulates moisture (0–100), temp (°C), humidity (%). You can stream real sensor CSV if available.

### Sample Sensor CSVs
- `app/data/sensors.csv` (balanced day, default)
- `app/data/sensors_dry.csv` (dry conditions: low moisture, high temp, low humidity)
- `app/data/sensors_wet.csv` (wet conditions: high moisture, high humidity)
- `app/data/sensors_heatwave.csv` (very high temp, low humidity)

### Sample Images
- In the app, expand “Sample Images (click to generate test files)” and press “Generate sample images”.
- Files are saved under `app/data/`:
  - `synthetic_field.png` (healthy)
  - `stressed_field.png` (reduced vegetation)
  - `red_band.png` and `nir_band.png` (grayscale bands)

## How It Works
- NDVI = (NIR − Red) / (NIR + Red) computed in `app/utils/preprocessing.py`.
- Soil rules in `app/utils/analytics.py` classify Good/Risky using thresholds.
- Pest risk model (`DecisionTreeClassifier`) is trained on synthetic samples (features: NDVI mean, humidity, temp) for demonstration.

## Customization
- Replace `sensors.csv` with your data (same columns: `timestamp, moisture, temp, humidity`).
- Plug actual satellite tiles and update `extract_red_nir()` to point to correct Red/NIR indices.
- For real SMS, integrate Twilio or any SMS API in `app/app.py` where the simulation message is generated.

## Hackathon Tips
- Keep it simple: synthetic data is fine for judging.
- Emphasize color coding: Green = Safe/Good, Red = Danger/Risky.
- Show end-to-end flow live: upload image → see NDVI → view sensor trends → see alerts.

## Troubleshooting
- If `streamlit` isn’t recognized, run with `python -m streamlit run app/app.py`.
- If imports fail like `No module named 'app.utils'`, they’re already fixed to use local `utils/`.
- If OpenCV blur errors occur, set smoothing to an odd number or 0 (handled in code).

## License
MIT
