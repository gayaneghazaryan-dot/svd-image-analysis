📘 SVD Image Analysis Toolkit

This repository provides a complete, reproducible implementation of Singular Value Decomposition (SVD) methods for image compression, denoising, benchmarking, and dimensionality reduction (PCA). It accompanies the article:

SVDlab: A Reproducible Toolkit for SVD-based Image Compression, Denoising, and PCA with Adaptive Rank Selection
Submitted to the Journal of Open Source Software (JOSS), 2025.

🔧 Installation

Platform note: On macOS/Linux use python3 (and python3 -m pip …).
On Windows use py -3 (or python) for all commands below.

Clone the repository and install dependencies:

git clone https://github.com/gayaneghazaryan-dot/svd-image-analysis.git
cd svd-image-analysis


Create an isolated environment and install requirements:

python3 -m venv myenv
source myenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt


Optional dependency (elbow selection):

python3 -m pip install kneed


If kneed is not installed, the code automatically falls back to the energy-based rule.

Python ≥ 3.10 is required (tested on Python 3.13, Windows/macOS/Linux).

🚀 Usage

The toolkit consists of four scripts. Each script is self-contained and regenerates all figures and tables for a specific task. Outputs are written to results/Figures/ and results/Tables/.

# 1) Compression (default built-in "astronaut" image)
python3 code/svd_compression_merged.py

# 2) Denoising (default built-in "astronaut" image with default noise level)
python3 code/svd_denoising.py

# 3) Factorization benchmarks and plots (default "astronaut" and built-in "camera")
python3 code/benchmark_and_plots.py
python3 code/benchmark_and_plots.py --input=camera

# 4) PCA with adaptive component selection (default Iris dataset)
python3 code/pca_adaptive_combined.py


Each command automatically regenerates all required figures and tables under results/, allowing reviewers or users to fully reproduce the article’s results from a clean environment.

Command-line options (examples)
# Use a different built-in image
python3 code/benchmark_and_plots.py --input=camera

# Resize before processing (HxW)
python3 code/benchmark_and_plots.py --resize=256x256

# Add noise and benchmark on noisy input
python3 code/benchmark_and_plots.py --use_noisy --sigma=0.10

# Change adaptive energy threshold and strategies
python3 code/benchmark_and_plots.py --energy=0.995 --strategies energy elbow

📂 Outputs

results/Figures/ → publication-quality plots

results/Tables/ → CSV/LaTeX tables with PSNR, SSIM, runtime, PCA variance, etc.

📊 Example Results

To provide an overview of typical outputs, a few representative figures and tables are included under the examples/ folder:

examples/Figures/ — sample plots illustrating:

Image compression and denoising results (astronaut_svd_comparison_995.pdf, astronaut_spectrum.png)

PCA visualization (pca_2d_scatter.pdf)

Benchmark performance (astronaut_psnr_vs_k_fixed_offset.pdf)

examples/Tables/ — one CSV file (astronaut_denoising_comparison.csv) showing quantitative PSNR and SSIM results.

These examples demonstrate the structure and appearance of the automatically generated outputs. All full results can be reproduced by running the four Python scripts as described above.

✨ Features

Adaptive rank selection — combines energy thresholding and elbow detection for robust SVD truncation.

Unified benchmarking — transparent comparisons of SVD, EVD, and QR under identical conditions.

Cross-domain applications — supports image compression, denoising, and PCA-based dimensionality reduction.

🔁 Reproducing Figures and Tables

All figures and tables presented in the paper are generated automatically by the four Python scripts located in the code/ folder. Each script creates its own subdirectories under results/Figures/ and results/Tables/, containing all publication-ready outputs in PDF, CSV, and LaTeX formats.

No pre-generated results are stored in the repository to ensure reproducibility and lightweight version control. After installation, simply run:

python3 code/svd_compression_merged.py
python3 code/svd_denoising.py
python3 code/benchmark_and_plots.py
python3 code/pca_adaptive_combined.py


This will regenerate the full set of figures and tables exactly as used in the manuscript. Each run is deterministic and environment-controlled via fixed random seeds and version-pinned dependencies.

📜 License

This project is licensed under the MIT License – see the LICENSE.txt
 file for details.

📖 Citation

If you use this software, please cite:

Ghazaryan, G., & Ghazaryan, A. (2025).
SVDlab: A Reproducible Toolkit for SVD-based Image Compression, Denoising, and PCA with Adaptive Rank Selection.
Journal of Open Source Software (JOSS).
DOI: to be assigned upon acceptance.

✦ With only four commands, the entire paper and all figures can be reproduced from scratch.
