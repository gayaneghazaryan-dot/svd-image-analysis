# src/svd_utils.py
"""
Reusable SVD functions for image processing.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def svd_denoise(img_noisy, k=50):
    """Apply SVD to denoise an image."""
    U, Sigma, Vt = np.linalg.svd(img_noisy, full_matrices=False)
    U_k = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    Vt_k = Vt[:k, :]
    return U_k @ Sigma_k @ Vt_k

def svd_compress(img_clean, k=30):
    """Compress image using truncated SVD."""
    return svd_denoise(img_clean, k)

def compute_metrics(img_clean, img_reconstructed):
    """Compute PSNR and energy preserved."""
    psnr = peak_signal_noise_ratio(img_clean, img_reconstructed, data_range=1.0)
    U, Sigma, Vt = np.linalg.svd(img_clean, full_matrices=False)
    energy_ratio = np.sum(Sigma[:min(U.shape[1], Vt.shape[0])]**2) / np.sum(Sigma**2)
    return psnr, energy_ratio

def compression_ratio(m, n, k):
    """Compute storage compression ratio."""
    return (m * n) / (k * (m + n + 1))