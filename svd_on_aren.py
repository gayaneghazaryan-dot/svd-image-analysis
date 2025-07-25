r"""
SVD for Image Denoising and Compression
---------------------------------------
This script:
1. Loads an image from r"C:\Users\Gayane\Python\Aren.jpg"
2. Resizes to 800x600
3. Applies SVD for:
   a) Denoising (remove Gaussian noise)
   b) Compression (low-rank approximation)
4. Computes metrics: PSNR, CR, energy preservation
5. Visualizes results

Requirements:
    pip install numpy matplotlib scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color, transform
from skimage.metrics import peak_signal_noise_ratio
import os

# --- Configuration ---
image_path = r"C:\Users\Gayane\Python\Aren.jpg"
target_height = 800
target_width = 600
k_denoise = 50    # Rank for denoising
k_compress = 30  # Rank for compression
noise_sigma = 0.1  # Noise level

# Check if file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at: {image_path}\n"
                            f"Please check the path and filename.")

print("=== SVD: Image Denoising and Compression ===")
print(f"Loading image from: {image_path}")
print(f"Resizing to: ({target_height}, {target_width})")

# --- 1. Load and Resize Image ---
img_original = io.imread(image_path)

# Convert to grayscale
if img_original.ndim == 3:
    img_gray = color.rgb2gray(img_original)
else:
    img_gray = img_original

# Resize and normalize
img_resized = transform.resize(img_gray, (target_height, target_width), anti_aliasing=True)
img_clean = img_as_float(img_resized)
print(f"Image shape after resize: {img_clean.shape}")

# --- 2. Add Noise for Denoising Task ---
img_noisy = img_clean + np.random.normal(0, noise_sigma, img_clean.shape)
img_noisy = np.clip(img_noisy, 0.0, 1.0)

# --- 3. Compute SVD (used for both tasks) ---
U, Sigma, Vt = np.linalg.svd(img_noisy, full_matrices=False)
print(f"SVD computed. Total singular values: {len(Sigma)}")

# ================================
#   A. IMAGE DENOISING
# ================================

# Reconstruct with k_denoise
U_k_d = U[:, :k_denoise]
Sigma_k_d = np.diag(Sigma[:k_denoise])
Vt_k_d = Vt[:k_denoise, :]
img_denoised = U_k_d @ Sigma_k_d @ Vt_k_d
img_denoised = np.clip(img_denoised, 0.0, 1.0)

# PSNR for denoising
psnr_noisy = peak_signal_noise_ratio(img_clean, img_noisy, data_range=1.0)
psnr_denoised = peak_signal_noise_ratio(img_clean, img_denoised, data_range=1.0)

print("\n--- DENOISING RESULTS ---")
print(f"PSNR (Noisy):       {psnr_noisy:.2f} dB")
print(f"PSNR (Denoised, k={k_denoise}): {psnr_denoised:.2f} dB")
print(f"Improvement:        {psnr_denoised - psnr_noisy:+.2f} dB")

# ================================
#   B. IMAGE COMPRESSION
# ================================

# Use clean image for compression (more realistic)
U_c, Sigma_c, Vt_c = np.linalg.svd(img_clean, full_matrices=False)

# Reconstruct with k_compress
U_k_c = U_c[:, :k_compress]
Sigma_k_c = np.diag(Sigma_c[:k_compress])
Vt_k_c = Vt_c[:k_compress, :]
img_compressed = U_k_c @ Sigma_k_c @ Vt_k_c
img_compressed = np.clip(img_compressed, 0.0, 1.0)

# --- Compression Metrics ---
m, n = img_clean.shape
storage_full = m * n
storage_compressed = k_compress * (m + n + 1)
compression_ratio = storage_full / storage_compressed
energy_preserved = np.sum(Sigma_c[:k_compress]**2) / np.sum(Sigma_c**2)
psnr_compressed = peak_signal_noise_ratio(img_clean, img_compressed, data_range=1.0)

print("\n--- COMPRESSION RESULTS ---")
print(f"Original storage:     {storage_full:} values")
print(f"Compressed storage:   {storage_compressed:} values")
print(f"Compression Ratio:    {compression_ratio:.2f}x")
print(f"Energy Preserved:     {energy_preserved:.3f} ({energy_preserved*100:.1f}%)")
print(f"PSNR (Compressed):    {psnr_compressed:.2f} dB")

# ================================
#   C. VISUALIZATION
# ================================

plt.figure(figsize=(18, 10))

# Row 1: Denoising
plt.subplot(2, 3, 1)
plt.imshow(img_clean, cmap='gray')
plt.title('Original (Clean)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_noisy, cmap='gray')
plt.title(f'Noisy Image\n(PSNR = {psnr_noisy:.2f} dB)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_denoised, cmap='gray')
plt.title(f'Denoised (k={k_denoise})\n(PSNR = {psnr_denoised:.2f} dB)')
plt.axis('off')

# Row 2: Compression
plt.subplot(2, 3, 4)
plt.imshow(img_clean, cmap='gray')
plt.title('Original (Clean)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_compressed, cmap='gray')
plt.title(f'Compressed (k={k_compress})\n(PSNR = {psnr_compressed:.2f} dB)')
plt.axis('off')

# Singular values plot
plt.subplot(2, 3, 6)
plt.semilogy(Sigma_c, 'b-', linewidth=2, label='Singular Values')
plt.axvline(k_compress, color='r', linestyle='--', label=f'k={k_compress}')
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value (log scale)')
plt.title('Singular Value Decay')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle("SVD Applications: Denoising & Compression (Resized to 800×600)", fontsize=16)
plt.tight_layout()
plt.show()


# ================================
#   D. OPTIONAL: SAVE RESULTS
# ================================

# Uncomment to save compressed/denoised images
# output_dir = r"C:\Users\Gayane\Python"
# io.imsave(os.path.join(output_dir, "Aren_denoised_SVD.png"), (img_denoised * 255).astype(np.uint8))
# io.imsave(os.path.join(output_dir, "Aren_compressed_SVD.png"), (img_compressed * 255).astype(np.uint8))
# print(f"Denoised and compressed images saved to: {output_dir}")

