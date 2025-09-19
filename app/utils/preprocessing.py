import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize any image or band to 0..1 float32.
    Handles uint8/uint16/float. Clips to [0,1] for safety.
    """
    if img is None:
        raise ValueError("Input image is None")
    img = img.astype(np.float32)
    # If values look like 0..255 or 0..65535 scale accordingly
    vmax = np.percentile(img, 99.9)
    vmin = np.percentile(img, 0.1)
    if vmax - vmin <= 1e-6:
        vmax, vmin = img.max(), img.min()
    if vmax - vmin <= 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    norm = (img - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    return norm.astype(np.float32)


def smooth_image(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Gaussian smoothing using OpenCV. Preserves dtype.
    """
    # Ensure kernel size is a positive odd integer as required by OpenCV
    k = int(ksize)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    if img.ndim == 2:
        return cv2.GaussianBlur(img, (k, k), 0)
    else:
        # apply channel-wise
        channels = []
        for c in range(img.shape[2]):
            channels.append(cv2.GaussianBlur(img[..., c], (k, k), 0))
        return np.stack(channels, axis=-1)


def median_denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filter using SciPy; good for salt-and-pepper noise."""
    if img.ndim == 2:
        return ndi.median_filter(img, size=ksize)
    else:
        channels = []
        for c in range(img.shape[2]):
            channels.append(ndi.median_filter(img[..., c], size=ksize))
        return np.stack(channels, axis=-1)


def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute NDVI = (NIR - Red) / (NIR + Red).
    Expects bands already roughly aligned and normalized in 0..1.
    """
    nir_n = normalize_image(nir)
    red_n = normalize_image(red)
    denom = (nir_n + red_n).astype(np.float32)
    ndvi = (nir_n - red_n) / (denom + eps)
    # NDVI theoretically in [-1, 1]; clamp for visualization
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi.astype(np.float32)


def extract_red_nir(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract Red and NIR bands from an image.
    - If 4-channel, assume RGBA or RGB+NIR: use channel 0=R, 3=NIR (heuristic)
    - If 3-channel RGB: approximate NIR with the R or G channel; default to G for vegetation contrast
    - If single channel: treat as Red and synthesize NIR via slight smoothing + offset
    """
    if image.ndim == 2:
        red = image
        nir = np.clip(smooth_image(image, 5) + 0.05, 0, None)
        return red, nir
    if image.shape[2] >= 4:
        red = image[..., 2] if image.dtype == np.uint8 else image[..., 0]  # heuristic guard
        # Prefer last channel as NIR if present
        nir = image[..., -1]
        return red, nir
    # 3-channel RGB: use R as Red, G as a pseudo-NIR
    red = image[..., 2] if image.shape[2] == 3 else image[..., 0]
    nir = image[..., 1]
    return red, nir
