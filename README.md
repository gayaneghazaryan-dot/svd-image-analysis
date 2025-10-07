📘 SVD Image Analysis Toolkit

This repository provides a complete, reproducible implementation of Singular Value Decomposition (SVD) methods for
image compression, denoising, benchmarking, and dimensionality reduction (PCA).
It accompanies the article:

SVDlab: A Reproducible Toolkit for SVD-based Image Compression, Denoising, and PCA with Adaptive Rank Selection
Submitted to the Journal of Open Source Software (JOSS), 2025.

🔧 Installation

Clone the repository and install dependencies:

git clone https://github.com/gayaneghazaryan-dot/svd-image-analysis.git
cd svd-image-analysis
pip install -r requirements.txt


💡 Tip:
All experiments were executed in a virtual environment named myenv.
To reproduce the same setup:

python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt


Python ≥ 3.9 is required (tested on Python 3.13, Windows/macOS/Linux).

🚀 Usage

The toolkit consists of four scripts.
Each script is self-contained and regenerates all figures and tables for a specific task.
Outputs are written to results/Figures/ and results/Tables/.

# 1) Compression (default built-in "astronaut" image)
python code/svd_compression_merged.py

# 2) Denoising (default built-in "astronaut" image with default noise level)
python code/svd_denoising.py

# 3) Factorization benchmarks and plots (default "astronaut" and built-in "camera")
python code/benchmark_and_plots.py
python code/benchmark_and_plots.py --input=camera

# 4) PCA with adaptive component selection (default Iris dataset)
python code/pca_adaptive_combined.py


Each command automatically regenerates all required figures and tables under results/,
allowing reviewers or users to fully reproduce the article’s results from a clean environment.

📂 Outputs

results/Figures/ → Publication-quality plots

results/Tables/ → CSV tables with PSNR, SSIM, runtime, PCA variance, etc.

📊 Example Results

To provide an overview of typical outputs, a few representative figures and tables are included under the examples/ folder:

examples/Figures/ — sample plots illustrating:

Image compression and denoising results (astronaut_svd_comparison_995.pdf, astronaut_spectrum.png)

PCA visualization (pca_2d_scatter.pdf)

Benchmark performance (astronaut_psnr_vs_k_fixed_offset.pdf)

examples/Tables/ — one CSV file (astronaut_denoising_comparison.csv) showing quantitative PSNR and SSIM results.

These examples demonstrate the structure and appearance of the automatically generated outputs.
All full results can be reproduced by running the four Python scripts as described above.

✨ Features

Adaptive rank selection — combines energy thresholding and elbow detection for robust SVD truncation.

Unified benchmarking — transparent comparisons of SVD, EVD, and QR under identical conditions.

Cross-domain applications — supports image compression, denoising, and PCA-based dimensionality reduction.

🔁 Reproducing Figures and Tables

All figures and tables presented in the paper are generated automatically by the four Python scripts located in the code/ folder.
Each script creates its own subdirectories under results/Figures/ and results/Tables/, containing all publication-ready outputs in PDF, CSV, and LaTeX formats.

No pre-generated results are stored in the repository to ensure reproducibility and lightweight version control.
After installation, simply run:

python code/svd_compression_merged.py
python code/svd_denoising.py
python code/benchmark_and_plots.py
python code/pca_adaptive_combined.py


This will regenerate the full set of figures and tables exactly as used in the manuscript.
Each run is deterministic and environment-controlled via fixed random seeds and version-pinned dependencies.

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

📖 Citation

If you use this software, please cite:

Ghazaryan, G., & Ghazaryan, A. (2025).
SVDlab: A Reproducible Toolkit for SVD-based Image Compression, Denoising, and PCA with Adaptive Rank Selection.
Journal of Open Source Software (JOSS).
DOI: to be assigned upon acceptance.

✦ With only four commands, the entire paper and all figures can be reproduced from scratch.
