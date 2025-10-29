# 📘 Singular Value Decomposition in Image Processing and Data Analysis: Theory, Algorithms, and Reproducible Experiments

### *(SVDlab: A Reproducible Toolkit for SVD-based Image Compression, Denoising, and PCA with Adaptive Rank Selection)*  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17313445.svg)](https://doi.org/10.5281/zenodo.17313445)  
Archived at Zenodo: [https://doi.org/10.5281/zenodo.17313445](https://doi.org/10.5281/zenodo.17313445)

---

This repository accompanies the article:

> **Ghazaryan, G., & Ghazaryan, A. (2025).**  
> *Singular Value Decomposition in Image Processing and Data Analysis: Theory, Algorithms, and Reproducible Experiments.*  
> Submitted to *Mathematical Problems of Computer Science (MPCS)*, 2025.

---

## 🔧 Installation

Platform note:  
- On **macOS/Linux**, use `python3` (and `python3 -m pip …`).  
- On **Windows**, use `py -3` (or `python`) for all commands below.

```bash
# Clone the repository
git clone https://github.com/gayaneghazaryan-dot/svd-image-analysis.git
cd svd-image-analysis

# Create an isolated environment and install dependencies
python3 -m venv myenv
source myenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Optional dependency (for elbow detection)
python3 -m pip install kneed

```

If `kneed` is not installed, the code automatically falls back to the energy-based rule.  
Python ≥ 3.10 is required (tested on Python 3.13 under Windows/macOS/Linux).

---

## 🚀 Usage

The toolkit consists of four self-contained scripts.  
Each reproduces all figures and tables for one of the main tasks.  
Outputs are written to `results/Figures/` and `results/Tables/`.

```bash
# 1) Image compression
python3 code/svd_compression_merged.py

# 2) Image denoising
python3 code/svd_denoising.py

# 3) Factorization benchmarks
python3 code/benchmark_and_plots.py

# 4) PCA with adaptive component selection
python3 code/pca_adaptive_combined.py
```


Each command regenerates all required figures and tables, allowing reviewers to reproduce the article’s results from a clean environment.

<details> <summary>Optional: Advanced CLI arguments (not required for MPCS reproduction)</summary>

```bash
# Use a different built-in image
python3 code/benchmark_and_plots.py --input=camera

# Resize before processing (HxW)
python3 code/benchmark_and_plots.py --resize=256x256

# Add noise and benchmark on noisy input
python3 code/benchmark_and_plots.py --use_noisy --sigma=0.10

# Change adaptive energy threshold and strategies
python3 code/benchmark_and_plots.py --energy=0.995 --strategies energy elbow
```
</details>

---

## 📂 Outputs

```
results/
 ├── Figures/   → publication-quality plots
 └── Tables/    → CSV/LaTeX tables with PSNR, SSIM, runtime, PCA variance, etc.
```

Example outputs are provided under `examples/` for illustration:

- **examples/Figures/** — sample plots (compression, denoising, PCA, benchmarking)  
- **examples/Tables/** — representative CSV file with PSNR/SSIM metrics  

These examples demonstrate the structure and appearance of the automatically generated outputs.  
All full results can be reproduced by running the four Python scripts above.

---

## ✨ Features

- **Adaptive rank selection** — combines cumulative-energy thresholds with Kneedle-based elbow detection  
- **Unified benchmarking** — transparent comparisons of SVD, EVD, and QR under identical conditions  
- **Cross-domain applications** — supports image compression, denoising, and PCA-based dimensionality reduction  
- **Reproducibility by design** — deterministic results, fixed random seeds, and pinned dependencies  

---

## 🔁 Reproducing Figures and Tables

All figures and tables presented in the paper are generated automatically by the four Python scripts in the `code/` folder.  
Each script creates its own subdirectories under `results/Figures/` and `results/Tables/`, containing all publication-ready outputs in PDF, CSV, and LaTeX formats.

No pre-generated results are stored in the repository to ensure reproducibility and lightweight version control.  
After installation, simply run:

```bash
python3 code/svd_compression_merged.py
python3 code/svd_denoising.py
python3 code/benchmark_and_plots.py
python3 code/pca_adaptive_combined.py
```

This will regenerate the full set of figures and tables exactly as referenced in the MPCS manuscript.
Each run is deterministic and environment-controlled via fixed random seeds and version-pinned dependencies.

---

## 📜 License

This project is licensed under the **MIT License** – see the `LICENSE.txt` file for details.

---

## 📖 Citation

If you use this software, please cite:

> Ghazaryan, G., & Ghazaryan, A. (2025).  
> *Singular Value Decomposition in Image Processing and Data Analysis: Theory, Algorithms, and Reproducible Experiments.*  
> *Mathematical Problems of Computer Science (MPCS).*  
> DOI: [10.5281/zenodo.17313445](https://doi.org/10.5281/zenodo.17313445)


✦ With only four commands, the entire paper and all figures can be reproduced from scratch.

