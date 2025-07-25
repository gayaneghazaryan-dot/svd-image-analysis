# --- Fix Python Path ---
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------
from src.svd_utils import svd_denoise, svd_compress, compute_metrics, compression_ratio
"""
SVD Denoising and Compression on Your Image: 'Aren.jpg'
✅ This script now works standalone — no import errors!
"""

# --- 🔧 AUTO-CONFIG: Fix Python Path (DO NOT REMOVE) ---
import sys
import os

# Get the directory of this script (supplementary/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (project root: svd-image-analysis/)
project_root = os.path.dirname(current_dir)

# Add project root to Python's module search path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Optional: Debug path (uncomment to check)
# print(f"🔧 Project root added to path: {project_root}")
# print(f"🔍 sys.path: {sys.path}")
# -------------------------------------------------------

# --- ✅ NOW SAFE TO IMPORT ---
try:
    from src.svd_utils import svd_denoise, svd_compress, compute_metrics, compression_ratio
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure:")
    print("   1. 'src/svd_utils.py' exists in the parent folder")
    print("   2. 'src/__init__.py' exists (empty file)")
    print("   3. You didn't rename the 'src' folder")
    sys.exit(1)

# --- 📷 CONFIGURE IMAGE PATH ---
# ✏️ EDIT THIS LINE TO POINT TO YOUR IMAGE:
image_path = r"C:\Users\Gayane\Python\Aren.jpg"

# Do not change below unless adjusting parameters
target_shape = (600, 800)     # Resize to 600x800
k_denoise = 50                # Rank for denoising
k_compress = 30               # Rank for compression
noise_sigma = 0.1             # Noise level (0.1 = 10% noise)

# --- 📂 LOAD & PREPARE IMAGE ---
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io, img_as_float, color, transform
    from skimage.metrics import peak_signal_noise_ratio
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("💡 Run: pip install numpy matplotlib scikit-image")
    sys.exit(1)

# Load image
try:
    img_original = io.imread(image_path)
except FileNotFoundError:
    print(f"❌ Image not found at: {image_path}")
    print("💡 Please update the 'image_path' variable to the correct location.")
    print("   Example: r'C:\\MyFolder\\Aren.jpg'")
    sys.exit(1)

# Convert to grayscale if needed
if img_original.ndim == 3:
    img_gray = color.rgb2gray(img_original)
else:
    img_gray = img_original

# Resize and normalize
img_resized = transform.resize(img_gray, target_shape, anti_aliasing=True)
img_clean = img_as_float(img_resized)

# --- 🌫️ ADD NOISE ---
img_noisy = np.clip(img_clean + np.random.normal(0, noise_sigma, img_clean.shape), 0, 1)
img_denoised = svd_denoise(img_noisy, k=k_denoise)
psnr_d, _ = compute_metrics(img_clean, img_denoised)
psnr_n = peak_signal_noise_ratio(img_clean, img_noisy, data_range=1.0)

# --- 📦 COMPRESS ---
img_compressed = svd_compress(img_clean, k=k_compress)
psnr_c, energy = compute_metrics(img_clean, img_compressed)
cr = compression_ratio(*img_clean.shape, k_compress)

# --- 🖼️ DISPLAY RESULTS ---
plt.figure(figsize=(18, 6))

titles = [
    'Original',
    f'Noisy (PSNR={psnr_n:.2f} dB)',
    f'Denoised (PSNR={psnr_d:.2f} dB)',
    f'Compressed (k={k_compress}, CR={cr:.2f}x)'
]

images = [img_clean, img_noisy, img_denoised, img_compressed]

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 4, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title, fontsize=10)
    plt.axis('off')

plt.suptitle("SVD on 'Aren.jpg': Denoising & Compression", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# --- 📊 PRINT METRICS ---
print("\n" + "="*50)
print("       ✅ SVD RESULTS FOR 'AREN.JPG'")
print("="*50)
print(f"📁 Image Size:       {img_clean.shape[0]} × {img_clean.shape[1]}")
print(f"⚙️  Denoising (k={k_denoise}): PSNR = {psnr_d:.2f} dB")
print(f"⚙️  Compression (k={k_compress}):")
print(f"   📉 Compression Ratio: {cr:.2f}×")
print(f"   🔋 Energy Preserved:  {energy:.1%}")
print(f"   📏 PSNR:              {psnr_c:.2f} dB")
print("="*50)