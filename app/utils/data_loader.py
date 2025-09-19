import os
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, Optional


def load_sensor_csv(csv_path: str) -> pd.DataFrame:
    """Load simulated IoT soil sensor data from CSV.
    Expected columns: timestamp, moisture, temp, humidity
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]) if os.path.exists(csv_path) else None
    if df is None or df.empty:
        raise FileNotFoundError(f"Sensor CSV not found or empty at: {csv_path}")
    return df.sort_values("timestamp")


def read_image(image_path: str) -> np.ndarray:
    """Read an image using OpenCV. Returns array in RGB order (float32 0..1)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if bgr.ndim == 3:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = bgr  # grayscale
    rgb = rgb.astype(np.float32)
    # normalize to 0..1 for consistency if looks like 0..255
    if rgb.max() > 1.0:
        rgb /= 255.0
    return rgb


def generate_synthetic_field(size: Tuple[int, int] = (256, 256), seed: int = 42) -> np.ndarray:
    """Generate a synthetic RGB image representing fields with varying vegetation.
    Returns float32 RGB in 0..1.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    base = rng.normal(loc=0.4, scale=0.1, size=(h, w)).astype(np.float32)
    base = np.clip(base, 0.1, 0.8)

    # Create patches of healthier vegetation
    for _ in range(8):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rad = rng.integers(min(h, w)//12, min(h, w)//6)
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
        base[mask] += rng.uniform(0.1, 0.25)

    base = np.clip(base, 0.05, 0.95)

    # Map to RGB: stronger in G to reflect vegetation
    img = np.stack([
        base * 0.8,      # R
        base * 1.0,      # G
        base * 0.6       # B
    ], axis=-1)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img
