# 📘 On the Theory and Application of Singular Value Decomposition in Image Processing and Data Analysis 
### *Algorithms, Adaptive Methods, and Reproducible Experiments*


*(SVDlab: A Reproducible Toolkit Demonstrating the Theory and Application of SVD in Image Processing — Compression, Denoising, and PCA)*


Repository archive DOI will be added when available.

---

> **Ghazaryan, G., & Ghazaryan, A. (2026).**  
> *On the Theory and Application of Singular Value Decomposition in Image Processing and Data Analysis.*  
> Accepted for publication in Mathematical Problems of Computer Science (MPCS)


---

## 🔧 Installation

**Requirements**  
- Python ≥ 3.10 (tested on Python 3.13 under macOS, Linux, and Windows)  
- Recommended: create a virtual environment  

```bash
# Clone the repository
git clone https://github.com/gayaneghazaryan-dot/svd-image-analysis.git
cd svd-image-analysis

# Create an isolated environment and install dependencies
python3 -m venv myenv
source myenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Optional dependency for elbow detection
python3 -m pip install kneed
```

If `kneed` is not installed, the toolkit automatically falls back to the cumulative-energy rule.

---

## 🚀 Usage

The toolkit includes **four self-contained scripts**, each reproducing all figures and tables for one application.  
Results are saved to `results/Figures/` and `results/Tables/`.

```bash
# 1) Image compression
python3 code/svd_compression_merged.py

# 2) Image denoising
python3 code/svd_denoising.py

# 3) Decomposition benchmarking (SVD, EVD, QR)
python3 code/benchmark_and_plots.py

# 4) PCA with adaptive component selection
python3 code/pca_adaptive_combined.py
```

Each command reproduces the article’s results from a clean environment.

<details>
<summary>Optional command-line arguments</summary>

```bash
# Use another built-in image
python3 code/benchmark_and_plots.py --input=camera

# Resize before processing (HxW)
python3 code/benchmark_and_plots.py --resize=256x256

# Add Gaussian noise for denoising tests
python3 code/benchmark_and_plots.py --use_noisy --sigma=0.10

# Modify rank-selection parameters
python3 code/benchmark_and_plots.py --energy=0.995 --strategies energy elbow
```
</details>

---

## 📂 Outputs

```
results/
 ├── Figures/   → publication-ready plots  
 └── Tables/    → CSV/LaTeX tables with PSNR, SSIM, runtime, variance, etc.
```


All figures and tables in the MPCS manuscript can be reproduced automatically by the scripts above.

---

## ✨ Key Features

- **Adaptive rank selection** – hybrid cumulative-energy + Kneedle elbow detection  
- **Unified benchmarking** – consistent SVD, EVD, QR comparison  
- **Cross-domain coverage** – compression, denoising, and PCA  
- **Reproducibility by design** – deterministic runs, fixed seeds, pinned dependencies  

---

## 🔁 Reproducing the Paper’s Results

All figures and tables from the MPCS submission are generated automatically.  
After installation, simply run:

```bash
python3 code/svd_compression_merged.py
python3 code/svd_denoising.py
python3 code/benchmark_and_plots.py
python3 code/pca_adaptive_combined.py
```

This will regenerate the complete set of figures and tables referenced in the paper.

---

## 📜 License

Released under the **MIT License**.  
See [`LICENSE`](LICENSE) for details.

---

## 📖 Citation

If you use this toolkit, please cite:

> **Ghazaryan, G., & Ghazaryan, A. (2025).**  
> *On the Theory and Application of Singular Value Decomposition in Image Processing and Data Analysis.*  
> *Mathematical Problems of Computer Science (MPCS).*  
> (DOI will be provided upon publication.)

---

✦ *With only four commands, the entire paper—including all figures and tables—can be reproduced from scratch.*
